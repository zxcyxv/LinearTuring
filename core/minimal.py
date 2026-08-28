"""최소 구조 — 근거 있는 부품만.  (2026-08-28)

블록 (세그당 K개, 전 블록 가중치 공유, 주기 1:1):
    h ← h + W_d[ (W_g h)/2 ⊙ (W_u h) ]           쌍선형 경계 (활성화 0개)                    [실측: 돌파의 변수]
    h ← h + β · E(x)                               입력 재주입                                 [실측: 끄면 붕괴]
    z_t = W_C h_t ∈ C^p,  ẑ = z/‖z‖               복소 주소, 위상 정규화                       [정리: |a| ≤ 1]
    a_tn = Σ_j e^{−α‖Δ‖} Re[ conj(ẑ_tj) e^{−i(ψ_j+θ_j·Δ)} ẑ_nj ]   부호 어텐션 (Gabor 커널)   [복소 회전 구조 — 유지]
    h ← Φ( h + Σ_n a_tn Wᵀ W h_n ),  Φ(h) = h/√(1+γ‖h‖²)          값 경로 (W,Wᵀ) + 4차 소산 [정리: 흡수구]
세그먼트 사이 detach (하네스).  로짓 = W_cls h.

버린 것과 이유:
    wc_skew (832² 반대칭 → exp)     5.5M 로 104×832 를 표현. ẑ 정규화가 있어 직교성이 사는 것 없음 → W_C 는 QR 로 행직교 (파라미터 104×832)
    w_ov, w_bo                       R1B8 경로에서 미사용 (유령 1.38M)
    d_t·b                            근거(a_tt∝‖z‖² 우함수항)가 ẑ 정규화로 소멸. 0차 편향만 잔존
    Λ (full)                         FHN 유비 외 근거 없음. 쌍선형이 이미 채널 혼합
    R, dt, rand_R, Strang 반스텝     R=1 에서 내용 없음. Φ 는 스텝당 1회
    puzzle_emb                       식별자 1개 → 전역 편향. 하네스 인터페이스 요구로 형식만 유지 (아래)
    q_head                           ACT 없음. 손실 호환용 상수 로짓 (파라미터 0)
"""
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, Optional

import pydantic
import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_                # URM
from models.sparse_embedding import CastedSparseEmbedding   # URM (하네스가 model.puzzle_emb.buffers() 를 요구)


@dataclass
class LTCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class LTConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0
    hidden_size: int = 832
    num_heads: int = 8
    loops: int = 16
    grid: int = 9
    blocks_per_seg: int = 8
    mlp_expansion: float = 4.0
    inj_gate_init: float = 0.25
    gamma_init: float = 0.1
    alpha_init: float = 0.1
    eps: float = 1e-4
    amp: bool = True
    forward_dtype: str = "float32"


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class LT_Inner(nn.Module):
    def __init__(self, config: LTConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g, d, H = config.seq_len, config.grid, config.hidden_size, config.num_heads
        assert T == g * g and d % H == 0 and (d // H) % 2 == 0
        self.d, self.H, self.dh = d, H, d // H
        self.p = self.dh // 2
        # 위치 (행, 열) 와 차 Δ
        u = torch.arange(T).float() // g; w = torch.arange(T).float() % g
        self.register_buffer("pos_u", u, persistent=False); self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("l1", (u[:, None] - u[None]).abs() + (w[:, None] - w[None]).abs(), persistent=False)
        # 임베딩 · 판독
        self.embed = nn.Embedding(config.vocab_size, d)
        self.w_cls = nn.Linear(d, config.vocab_size)
        self.inj_gate = nn.Parameter(torch.tensor(float(config.inj_gate_init)))
        # 복소 주소: W_C = [A;B] 행직교 (QR), ψ 위상차, θ 2D 파수, α 감쇠(헤드별)
        self.wc_raw = nn.Parameter(torch.randn(H, self.dh, d) / math.sqrt(d))
        self.psi = nn.Parameter(torch.rand(H, self.p) * 2 * math.pi - math.pi)
        self.theta = nn.Parameter((torch.rand(H, self.p, 2) * 2 - 1) * (math.pi / 2))
        self.alpha_raw = nn.Parameter(torch.full((H, 1), inv_softplus(config.alpha_init)))
        # 값 경로 (W, Wᵀ): 블록 항등 init
        w_sh = torch.zeros(H, self.dh, d)
        for m in range(H):
            w_sh[m, :, m * self.dh:(m + 1) * self.dh] = torch.eye(self.dh)
        self.w_sh = nn.Parameter(w_sh + 0.01 * torch.randn(H, self.dh, d) / math.sqrt(d))
        # 4차 소산
        self.gamma_raw = nn.Parameter(torch.tensor(inv_softplus(config.gamma_init)))
        # 쌍선형 경계 (down 영init → 시작 시 항등)
        inter = int(config.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.b_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()
        # 하네스 인터페이스 (내용상 전역 편향 1개)
        self.puzzle_emb_ndim = config.puzzle_emb_ndim
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)
        self.init_hidden = nn.Buffer(trunc_normal_init_(torch.empty(d, dtype=self.forward_dtype), std=1.0), persistent=True)

    # ---------------------------------------------------------------- 부품
    @property
    def gamma(self): return F.softplus(self.gamma_raw)
    @property
    def alpha(self): return F.softplus(self.alpha_raw)

    def W_C(self):
        """행직교 [A;B] (헤드별 dh×d): raw 의 전치를 QR 해 열직교 → 전치."""
        Q, _ = torch.linalg.qr(self.wc_raw.transpose(-1, -2))            # [H,d,dh]
        AB = Q.transpose(-1, -2)
        return AB[:, :self.p, :], AB[:, self.p:, :]

    def kernel(self):
        """decay_h [H,T,T], 위상각 A_t = ψ/2 + θ·pos_t (q), B_t = −ψ/2 + θ·pos_t (k) → cos/sin [T,H,p]."""
        decay_h = torch.exp(-self.alpha[:, 0, None, None] * self.l1)
        ppos = self.theta[..., 0, None] * self.pos_u + self.theta[..., 1, None] * self.pos_w   # [H,p,T]
        A = (ppos + self.psi[..., None] / 2).permute(2, 0, 1); B = (ppos - self.psi[..., None] / 2).permute(2, 0, 1)
        return decay_h, torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)

    def attn(self, h, AB, kc):
        """a_tn [B,H,T,T] = e^{−α‖Δ‖} · Re[conj(q̂_t) k̂_n],  q̂/k̂ = 위상 정규화 후 각각 회전."""
        A, Bm = AB; decay_h, cosA, sinA, cosB, sinB = kc
        x = torch.einsum('btd,hjd->bthj', h, A); y = torch.einsum('btd,hjd->bthj', h, Bm)
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt(); x = x / (nrm + self.config.eps); y = y / (nrm + self.config.eps)
        qx = x * cosA - y * sinA; qy = x * sinA + y * cosA
        kx = x * cosB - y * sinB; ky = x * sinB + y * cosB
        a = torch.einsum('bthj,bnhj->bhtn', qx, kx) + torch.einsum('bthj,bnhj->bhtn', qy, ky)
        return a * decay_h.unsqueeze(0)

    def phi(self, h):
        return h / torch.sqrt(1.0 + self.gamma * h.pow(2).sum(-1, keepdim=True))

    def step(self, h, AB, kc):
        a = self.attn(h, AB, kc)
        v = torch.einsum('btd,hcd->bthc', h, self.w_sh)                   # W h_n
        o = torch.einsum('bhtn,bnhc->bthc', a, v)                         # Σ_n a_tn v_n
        f = torch.einsum('bthc,hcd->btd', o, self.w_sh)                   # Wᵀ o
        return self.phi(h + f), a

    def boundary(self, h):
        g, u = self.b_gate_up(h).chunk(2, dim=-1)
        return h + self.b_down(0.5 * g * u)

    def injection(self, batch):
        inj = self.embed(batch["inputs"].to(torch.long))
        if self.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])
            pad = self.d - self.puzzle_emb_ndim
            if pad > 0: pe = F.pad(pe, (0, pad))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)
        return inj

    # ---------------------------------------------------------------- 세그먼트
    def empty_carry(self, batch_size):
        return LTCarry(current_hidden=torch.empty(batch_size, self.config.seq_len, self.d, dtype=self.forward_dtype))

    def reset_carry(self, reset_flag, carry):
        return replace(carry, current_hidden=torch.where(reset_flag.view(-1, 1, 1), self.init_hidden, carry.current_hidden))

    def forward(self, carry, batch):
        if self.config.amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nc, logits = self._forward(carry, batch)
            return replace(nc, current_hidden=nc.current_hidden.float()), logits.float()
        return self._forward(carry, batch)

    def _forward(self, carry, batch):
        h = carry.current_hidden; inj = self.injection(batch)
        AB = self.W_C(); kc = self.kernel()
        for _ in range(self.config.blocks_per_seg):
            h = self.boundary(h)
            h = h + self.inj_gate * inj
            h, _ = self.step(h, AB, kc)
        return replace(carry, current_hidden=h.detach()), self.w_cls(h)


class LT(nn.Module):
    """URM 하네스 인터페이스. ACT 없음 (halted = steps ≥ loops). q 로짓은 상수 (파라미터 0)."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LTConfig(**config_dict)
        self.inner = LT_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch):
        B = batch["inputs"].shape[0]
        return LTCarry(current_hidden=self.inner.empty_carry(B).current_hidden,
                       steps=torch.zeros((B,), dtype=torch.int32), halted=torch.ones((B,), dtype=torch.bool),
                       current_data={k: torch.empty_like(v) for k, v in batch.items()})

    def forward(self, carry, batch, compute_target_q: bool = False):
        inner = self.inner.reset_carry(carry.halted, carry)
        steps = torch.where(carry.halted, 0, carry.steps)
        data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        inner, logits = self.inner(inner, data)
        q = torch.full((logits.shape[0],), -5.0, device=logits.device, dtype=torch.float32)
        outputs = {"logits": logits, "q_halt_logits": q, "q_continue_logits": q}
        with torch.no_grad():
            steps = steps + 1; halted = steps >= self.config.loops
        return LTCarry(current_hidden=inner.current_hidden, steps=steps, halted=halted, current_data=data), outputs
