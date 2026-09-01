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
    coupling: Optional[torch.Tensor] = None       # [4단계] STDP 결합 기억 w [B,H,T,T]
    fresh: Optional[torch.Tensor] = None          # [B] bool — 이 퍼즐의 w 가 아직 초기화 전
    vprev: Optional[torch.Tensor] = None          # [causal] 직전 블록의 정규화 값 v̂ [B,T,H,C]
    gate: Optional[torch.Tensor] = None           # [gate] 직전 블록 메시지의 판별력에서 나온 굳힘 이득 K [B,T]


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
    addr_dim: int = 0           # [3단계] >0 이면 상태 = [주소 addr_dim | 값 나머지]. W_C 는 주소 블록만, 값 경로는 값 블록만 → row(W_C) ⟂ row(W), 스텝 안 a 불변 (STDP.md)
    stdp: bool = False          # [4단계] 결합 기억 w ← w + η(a−w), 결합 = (1−λ)a + λw. η,λ 헤드별 학습 (sigmoid). STDP.md §2
    stdp_target: str = "addr"  # faithful: Γ = a^β·⟨v̂_t,v̂_n⟩ — 가소성 창(β) × 동시활동. STDP 세 인자를 모두 담는 형태 (2026-08-31)
                                #   addr: Γ = a (주소 위상 커널, β 비대칭) / value: Γ = D(Δ)·⟨v̂_t,v̂_n⟩ (값 겹침만; 부호 없음 → 거부됨)
                                #   product: Γ = a_tn·⟨v̂_t,v̂_n⟩ — 결합 에너지 E(h,w)=−½Σ w a⟨Wh,Wh⟩+½Σw² 의 ∂/∂w. 그래프 부호 × 값 일치 (STDP.md)
                                #   causal: Γ = a_tn·⟨v̂_t,v̂_n⟩ + μ_h · a^β_tn·⟨Δv̂_t, v̂_n⟩ — 창의 비대칭(sin) 성분이 곱하는 항.
                                #     STDP 발화율 축약에서 창의 홀수 부분은 pre 활동 × post 활동의 시간 미분을 낳는다(Kempter–Gerstner–van Hemmen).
                                #     Δv̂_t = 이 블록의 값 변화 → "t 가 바뀌는 순간 서 있던 n" 이 결합에 남는다 = 의존 그래프 (STDP.md §6.13)
    stdp_eta_init: float = 0.1
    stdp_gain_init: float = 1.0  # [2026-08-31] 누적 이득 G (헤드별 학습, softplus). w ← (1−δ)w + δ·G·Γ 의 고정점 = G·⟨Γ⟩.
                                #   기존 EMA 는 고정점이 ⟨Γ⟩ 로 이득이 δ 와 무관하게 정확히 1 — 필터지 적분기가 아니다.
                                #   G=1 이면 기존과 비트 동치. δ=잊음률(eta_raw 재사용), G=축적량 으로 역할 분리.
    stdp_lam_init: float = 0.25
    stdp_lam_fixed: float = -1.0  # ≥0 이면 λ 를 이 값으로 고정(학습 안 함). 1.0 = 전달을 w 가 전담하는 STDP 충실형
    stdp_mu_init: float = 0.5
    stdp_diag: str = "keep"      # keep | zero(자기시냅스 제거) | only(관계항 제거) — 절제용, 기본은 무변경   # [causal] 인과 항의 헤드별 계수 μ_h 초기값 (학습; softplus 아님, 부호 자유)
    gate: bool = False          # [굳힘 게이트] 경계(추론) 항의 이득 = K(d) = (s·d)²/(1+(s·d)²)  — 칼만 이득 형.
                                #   d_t = (직전 블록 메시지가 만든 로짓의 top1−top2) / std_v  — 척도 불변 판별력(증거 정밀도의 대리).
                                #   K≡1 (현재) = 모든 메시지를 무한 정밀로 취급 = 전제 확인 없는 단정 규칙. STDP.md §6.13
    gate_s_init: float = 5.0   # K(d) 가 초기에 관대(≈0.85)하도록: 무작위 초기 모델의 d 중앙값 ≈0.46
    psi_zero: bool = False      # [2단계] ψ≡0 고정 → a_tn = a_nt (명제 7), 값 수송이 대칭 → 스텝이 E_adj 의 경사 (STDP.md §3)


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class LT_Inner(nn.Module):
    def __init__(self, config: LTConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g, d, H = config.seq_len, config.grid, config.hidden_size, config.num_heads
        assert T == g * g and d % H == 0 and (d // H) % 2 == 0
        self.split = config.addr_dim > 0
        self.d_a = config.addr_dim if self.split else d
        self.d_v = d - self.d_a if self.split else d
        assert self.d_a % H == 0 and self.d_v % H == 0 and (self.d_a // H) % 2 == 0
        self.d, self.H = d, H
        self.dh_a, self.dh_v = self.d_a // H, self.d_v // H     # 주소·값 헤드 폭 (분할 없으면 둘 다 d/H)
        self.dh = self.dh_v
        self.p = self.dh_a // 2
        # 위치 (행, 열) 와 차 Δ
        u = torch.arange(T).float() // g; w = torch.arange(T).float() % g
        self.register_buffer("pos_u", u, persistent=False); self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("l1", (u[:, None] - u[None]).abs() + (w[:, None] - w[None]).abs(), persistent=False)
        # 임베딩 · 판독
        self.embed = nn.Embedding(config.vocab_size, d)
        self.w_cls = nn.Linear(d, config.vocab_size)
        self.inj_gate = nn.Parameter(torch.tensor(float(config.inj_gate_init)))
        # 복소 주소: W_C = [A;B] 행직교 (QR), ψ 위상차, θ 2D 파수, α 감쇠(헤드별)
        self.wc_raw = nn.Parameter(torch.randn(H, self.dh_a, self.d_a) / math.sqrt(self.d_a))
        if config.psi_zero:
            self.register_buffer("psi", torch.zeros(H, self.p), persistent=False)
        else:
            self.psi = nn.Parameter(torch.rand(H, self.p) * 2 * math.pi - math.pi)
        self.theta = nn.Parameter((torch.rand(H, self.p, 2) * 2 - 1) * (math.pi / 2))
        self.alpha_raw = nn.Parameter(torch.full((H, 1), inv_softplus(config.alpha_init)))
        # 값 경로 (W, Wᵀ): 블록 항등 init
        w_sh = torch.zeros(H, self.dh_v, self.d_v)
        for m in range(H):
            w_sh[m, :, m * self.dh_v:(m + 1) * self.dh_v] = torch.eye(self.dh_v)
        self.w_sh = nn.Parameter(w_sh + 0.01 * torch.randn(H, self.dh_v, self.d_v) / math.sqrt(self.d_v))
        # [4단계] STDP 결합 기억: η, λ 헤드별 (sigmoid 재매개화)
        self.stdp = config.stdp
        if self.stdp:
            lg = lambda x: math.log(x / (1 - x))
            self.eta_raw = nn.Parameter(torch.full((H, 1, 1), lg(config.stdp_eta_init)))
            self.lam_raw = nn.Parameter(torch.full((H, 1, 1), lg(config.stdp_lam_init)))
            self.gain_raw = nn.Parameter(torch.full((H, 1, 1), inv_softplus(config.stdp_gain_init)))
            self.beta = nn.Parameter(torch.zeros(H, self.p))          # 위상 STDP 창의 비대칭 β_h (ψ 와 별개, 0 = 대칭 Hebb)
            if config.stdp_target in ("causal", "faithful"):
                self.beta.data.normal_(0.0, 0.5)                        # β=0 이면 a^β = a 라 창이 대칭 → 작은 무작위 비대칭에서 출발
            if config.stdp_target == "causal":
                self.mu = nn.Parameter(torch.full((H, 1, 1), float(config.stdp_mu_init)))   # 인과 항 계수 (부호 자유)
        assert not (config.gate and config.stdp_target == "causal"), "gate + causal 동시 사용은 미구현"
        self.gate_on = config.gate
        if self.gate_on:
            self.gate_s_raw = nn.Parameter(torch.tensor(inv_softplus(config.gate_s_init)))
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

    def kernel(self, psi=None):
        """decay_h [H,T,T], 위상각 A_t = ψ/2 + θ·pos_t (q), B_t = −ψ/2 + θ·pos_t (k) → cos/sin [T,H,p]. psi 를 주면 그 위상차로 (STDP 창 β 용)."""
        psi = self.psi if psi is None else psi
        decay_h = torch.exp(-self.alpha[:, 0, None, None] * self.l1)
        ppos = self.theta[..., 0, None] * self.pos_u + self.theta[..., 1, None] * self.pos_w   # [H,p,T]
        A = (ppos + psi[..., None] / 2).permute(2, 0, 1); B = (ppos - psi[..., None] / 2).permute(2, 0, 1)
        return decay_h, torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)

    def addr(self, h, AB):
        """정규화된 복소 주소 ẑ 의 (실부, 허부). **커널 위상(ψ/β)과 무관** — 한 블록에서 여러 커널로
        attn 을 부를 때 이것을 공유한다. 이 사영이 T×T einsum 보다 10배 비싸다 (7.18 vs 0.70 G-MAC)."""
        A, Bm = AB
        ha = h[..., :self.d_a] if self.split else h                         # 주소 블록만 읽음
        x = torch.einsum('btd,hjd->bthj', ha, A); y = torch.einsum('btd,hjd->bthj', ha, Bm)
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()
        return x / (nrm + self.config.eps), y / (nrm + self.config.eps)

    def attn_xy(self, xy, kc):
        """정규화된 주소에서 커널 kc 로 a 를 만든다 (회전 + 내적 + 감쇠)."""
        x, y = xy; decay_h, cosA, sinA, cosB, sinB = kc
        qx = x * cosA - y * sinA; qy = x * sinA + y * cosA
        kx = x * cosB - y * sinB; ky = x * sinB + y * cosB
        a = torch.einsum('bthj,bnhj->bhtn', qx, kx) + torch.einsum('bthj,bnhj->bhtn', qy, ky)
        return a * decay_h.unsqueeze(0)

    def attn(self, h, AB, kc):
        """a_tn [B,H,T,T] = e^{−α‖Δ‖} · Re[conj(q̂_t) k̂_n].  (분석 스크립트 호환용 래퍼)"""
        return self.attn_xy(self.addr(h, AB), kc)

    def phi(self, h):
        return h / torch.sqrt(1.0 + self.gamma * h.pow(2).sum(-1, keepdim=True))

    def step(self, h, AB, kc, w=None, fresh=None, kcb=None, vprev=None):
        """vprev: 직전 블록의 정규화 값 v̂ [B,T,H,C] (causal 전용). 반환에 현재 v̂ 를 함께 돌려준다."""
        xy = self.addr(h, AB)                                               # 주소 사영 — 블록당 1회
        a = self.attn_xy(xy, kc)
        hv = h[..., self.d_a:] if self.split else h
        v = torch.einsum('btd,hcd->bthc', hv, self.w_sh)                    # 값 사영 — 블록당 1회 (수송·agree 공용)
        if self.stdp:
            eta = torch.sigmoid(self.eta_raw); lam = torch.sigmoid(self.lam_raw) if self.config.stdp_lam_fixed < 0 else torch.full_like(self.lam_raw, float(self.config.stdp_lam_fixed))
            if self.config.stdp_target in ("value", "product", "causal", "faithful"):
                vv = v / (v.norm(dim=-1, keepdim=True) + self.config.eps)   # 재사용 (재계산 안 함)
                agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
                if self.config.stdp_target == "causal":
                    G = a * agree
                    if vprev is not None:
                        dv = vv - vprev                                                     # 이 블록의 값 변화 (post 의 시간 미분)
                        ab = self.attn_xy(xy, kcb) if kcb is not None else a                # 비대칭 창 a^β (사영 재사용)
                        G = G + self.mu * ab * torch.einsum('bthc,bnhc->bhtn', dv, vv)      # 인과 항: t 가 변할 때 서 있던 n
                elif self.config.stdp_target == "faithful":
                    # [2026-08-31] 충실형 STDP: 가소성 창(β) × 동시활동. 창 × pre × post 의 세 인자를 모두 담는다.
                    #   addr 은 창만, product 는 동시활동만(창은 전달용 ψ 를 빌려 씀) 담았다.
                    G = (self.attn_xy(xy, kcb) if kcb is not None else a) * agree   # 사영 재사용
                else:
                    G = agree * kc[0].unsqueeze(0) if self.config.stdp_target == "value" else a * agree   # product: 전달 창(ψ) × 값 일치
            else:
                G = self.attn_xy(xy, kcb) if kcb is not None else a      # STDP 창 Γ = cos(Δφ − θ·Δ − β)
            tgt = F.softplus(self.gain_raw) * G                      # 고정점 목표 = G·Γ (이득)
            if self.config.stdp_diag != "keep":
                # [2026-09-01 절제] STDP 에 자기시냅스(t=n)는 없다. Γ 단계에서 걸러야 w 대각이 항상 0 이다.
                eye = torch.eye(tgt.shape[-1], device=tgt.device, dtype=tgt.dtype)
                tgt = tgt * (eye if self.config.stdp_diag == "only" else (1 - eye))
            if w is None: w = tgt
            else:
                w = torch.where(fresh.view(-1, 1, 1, 1), tgt, w) if fresh is not None else w
                w = (1 - eta) * w + eta * tgt                        # 고정점 = G·⟨Γ⟩.  G=1 이면 기존 EMA 와 동일
            a = (1 - lam) * a + lam * w
        o = torch.einsum('bhtn,bnhc->bthc', a, v)                         # Σ_n a_tn v_n  (위에서 계산한 v 재사용)
        f = torch.einsum('bthc,hcd->btd', o, self.w_sh)                   # Wᵀ o
        if self.split: f = F.pad(f, (self.d_a, 0))                         # 주소 블록에는 0 → 스텝 안 a 불변
        hout = self.phi(h + f)
        if self.gate_on:                                                   # 메시지가 만드는 로짓 변화의 척도 불변 마진 → 다음 블록 굳힘 이득
            ell = torch.einsum('btd,vd->btv', f, self.w_cls.weight)
            t2 = ell.topk(2, dim=-1).values
            z = F.softplus(self.gate_s_raw) * (t2[..., 0] - t2[..., 1]) / (ell.std(-1) + self.config.eps)
            return hout, (w if self.stdp else a), z * z / (1.0 + z * z)
        if self.stdp and self.config.stdp_target == "causal":
            hv1 = hout[..., self.d_a:] if self.split else hout
            v1 = torch.einsum('btd,hcd->bthc', hv1, self.w_sh); v1 = v1 / (v1.norm(dim=-1, keepdim=True) + self.config.eps)
            return hout, w, v1
        return hout, (w if self.stdp else a)

    def boundary(self, h, gate=None):
        g, u = self.b_gate_up(h).chunk(2, dim=-1)
        delta = self.b_down(0.5 * g * u)
        return h + (delta if gate is None else gate.unsqueeze(-1) * delta)

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
        w = carry.coupling if self.stdp else None; fresh = carry.fresh if self.stdp else None
        kcb = self.kernel(self.beta) if self.stdp else None
        gate = carry.gate if self.gate_on else None
        if self.gate_on and fresh is not None and gate is not None: gate = torch.where(fresh.view(-1, 1), torch.zeros_like(gate), gate)
        causal = self.stdp and self.config.stdp_target == "causal"
        vprev = carry.vprev if causal else None
        if causal and vprev is not None and fresh is not None: vprev = torch.where(fresh.view(-1, 1, 1, 1), torch.zeros_like(vprev), vprev)
        for _ in range(self.config.blocks_per_seg):
            h = self.boundary(h, gate)                                     # 굳힘 = 직전 블록 증거의 판별력에 비례
            h = h + self.inj_gate * inj
            if self.gate_on:
                h, w, gate = self.step(h, AB, kc, w, fresh, kcb)
            elif causal:
                h, w, vprev = self.step(h, AB, kc, w, fresh, kcb, vprev)   # vprev = 직전 블록 값 (세그먼트 넘어 carry)
            else:
                h, w = self.step(h, AB, kc, w, fresh, kcb)
            fresh = None                                                   # 첫 블록에서만 초기화
        return replace(carry, current_hidden=h.detach(), coupling=(w.detach() if self.stdp else None), fresh=None,
                       vprev=(vprev.detach() if causal and vprev is not None else None),
                       gate=(gate.detach() if self.gate_on else None)), self.w_cls(h)


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
        inner = replace(inner, fresh=carry.halted.clone())
        steps = torch.where(carry.halted, 0, carry.steps)
        data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        inner, logits = self.inner(inner, data)
        q = torch.full((logits.shape[0],), -5.0, device=logits.device, dtype=torch.float32)
        outputs = {"logits": logits, "q_halt_logits": q, "q_continue_logits": q}
        with torch.no_grad():
            steps = steps + 1; halted = steps >= self.config.loops
        return LTCarry(current_hidden=inner.current_hidden, steps=steps, halted=halted, current_data=data, coupling=inner.coupling, vprev=inner.vprev, gate=inner.gate), outputs
