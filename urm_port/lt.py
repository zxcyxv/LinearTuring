"""LT — LinearTuring 코어의 URM 하네스 어댑터.

구조 (SUDOKU.md 채택 표):
  세그먼트(하네스 loop) = [ 경계 재부호화 W_O(contract) → 경계 입력 주입 → R 마이크로스텝 ]
  세그먼트 사이 detach (하네스가 수행) → 그래디언트 노출 = 블록 1개 (e^{λτ} 문제의 구조적 해법)
  ACT 제거: halted = (steps >= loops) 결정론. q_head 는 손실 호환용 보조로만 존치.
  삼중 루프(loops×H×L) → 이중(loops × R) 축소. H_cycles 워밍업 없음.
  puzzle_emb: 토큰 앞붙임 대신 주입 시 브로드캐스트 가산 (T=81·2D 위치 보존; 스도쿠는 식별자 1개라 전역 학습 바이어스)
"""
import sys
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import pydantic

sys.path.insert(0, "/workspace/LinearTuring")
from model1 import Model1                                   # noqa: E402
from models.common import trunc_normal_init_                # noqa: E402
from models.sparse_embedding import CastedSparseEmbedding   # noqa: E402


@dataclass
class LTCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class LTConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")   # causal 등 하네스 주입 키 허용
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0
    hidden_size: int
    num_heads: int
    R: int = 8                 # dt = 1/R (이산화 해상도)
    seg_steps: int = 0         # 세그먼트당 마이크로스텝 수 (0 = R, 즉 τ=1; 2R 이면 τ=2)
    loops: int = 16
    grid: int = 9
    ckpt: bool = True          # 마이크로스텝 gradient checkpointing (메모리 ÷R, 연산 ×~1.5)
    boundary_mlp: bool = False # 경계를 W_O 대신 g·h + SwiGLU(h) 로 — 내부 흐름은 순수 유지 (땜빵 검증용)
    mlp_expansion: float = 4.0
    forward_dtype: str = "float32"


class LT_Inner(nn.Module):
    def __init__(self, config: LTConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g = config.seq_len, config.grid
        assert T == g * g
        pos = torch.stack([torch.arange(T).float() // g, torch.arange(T).float() % g], 1)
        # 채택 레시피: sheaf(W,Wᵀ) + Λ full + 경계 W_O(수축 init) + b 유지 + 커널 lr ×1(하네스 단일 lr)
        self.core = Model1(d=config.hidden_size, H=config.num_heads, R=config.R,
                           n_classes=config.vocab_size, positions=pos, vocab=config.vocab_size,
                           pool=False, sheaf=True, lam_mode="full",
                           boundary_wo=True, wo_mode="contract",   # w_bo 파라미터 확보용 — 적용은 여기서 직접
                           alpha_per_head=True)                    # 감쇠 헤드 단위 → 고속 경로 (등록된 편차)
        self.puzzle_emb_ndim = config.puzzle_emb_ndim
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0,
                                                    cast_to=self.forward_dtype)
        self.q_head = nn.Linear(config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)
        # 주입 게이트 — carry 익사 수선 (step 15624 프로브: ‖W_O h‖/‖inj‖=0.47 → β=0.25 로 ~1.9 반전)
        self.inj_gate = nn.Parameter(torch.tensor(0.25))
        if config.boundary_mlp:
            # 경계 SwiGLU: down_proj 영 초기화 → 시작 시 항등성 유지(carry 익사 3회차 방지), 학습으로 성장
            d = config.hidden_size
            inter = int(config.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
            self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
            self.b_down = nn.Linear(inter, d, bias=False)
            with torch.no_grad():
                self.b_down.weight.zero_()
            self.b_carry = nn.Parameter(torch.tensor(0.6))    # 학습된 contract ρ 근방에서 출발
        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1.0),
            persistent=True)

    def empty_carry(self, batch_size: int) -> LTCarry:
        return LTCarry(current_hidden=torch.empty(
            batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype))

    def reset_carry(self, reset_flag: torch.Tensor, carry: LTCarry) -> LTCarry:
        return replace(carry, current_hidden=torch.where(
            reset_flag.view(-1, 1, 1), self.init_hidden, carry.current_hidden))

    def _injection(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inj = self.core.embed(batch["inputs"].to(torch.long))            # [B,T,d]
        if self.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])            # [B,ndim]
            pad = self.config.hidden_size - self.puzzle_emb_ndim
            if pad > 0:
                pe = F.pad(pe, (0, pad))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)                    # 브로드캐스트 가산
        return inj

    def forward(self, carry: LTCarry, batch: Dict[str, torch.Tensor]):
        m = self.core
        h = carry.current_hidden
        # ── 경계: 재부호화 + 입력 주입 ─────────────────────────────
        if self.config.boundary_mlp:
            gate, up = self.b_gate_up(h).chunk(2, dim=-1)
            h = self.b_carry * h + self.b_down(F.silu(gate) * up)
        else:
            h = h @ m.w_bo.t()
        h = h + self.inj_gate * self._injection(batch)
        # ── 내부 R 재귀 — 고속 경로: RoPE 화 위상 + 헤드 감쇠 마스크 (동치 검증됨) ──
        fc = m.kernel_fast(); AB = m.W_C(); dt = 1.0 / m.R
        nstep = self.config.seg_steps or self.config.R          # τ = nstep/R

        def micro(hh):
            hh = m.phi(hh, dt / 2)
            f, *_ = m.field(hh, None, None, None, AB, fast_ctx=fc)
            hh = hh + dt * f
            return m.phi(hh, dt / 2)

        for _ in range(nstep):
            if self.config.ckpt and self.training and torch.is_grad_enabled():
                h = checkpoint(micro, h, use_reentrant=False)
            else:
                h = micro(h)
        new_carry = replace(carry, current_hidden=h.detach())
        logits = m.w_cls(h)                                              # [B,T,vocab]
        q = self.q_head(h.mean(1)).to(torch.float32)
        return new_carry, logits, (q[..., 0], q[..., 1])


class LT(nn.Module):
    """URM 클래스 미러 — ACT 제거판 (halted = steps>=loops 결정론)."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LTConfig(**config_dict)
        self.inner = LT_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> LTCarry:
        batch_size = batch["inputs"].shape[0]
        return LTCarry(
            current_hidden=self.inner.empty_carry(batch_size).current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: LTCarry, batch: Dict[str, torch.Tensor], compute_target_q: bool = False):
        new_inner = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }
        new_inner, logits, (q_halt, q_cont) = self.inner(new_inner, new_current_data)
        outputs = {"logits": logits, "q_halt_logits": q_halt, "q_continue_logits": q_cont}
        with torch.no_grad():
            new_steps = new_steps + 1
            halted = new_steps >= self.config.loops        # ACT 없음 — 결정론적 소진
        return (
            LTCarry(current_hidden=new_inner.current_hidden, steps=new_steps,
                    halted=halted, current_data=new_current_data),
            outputs,
        )
