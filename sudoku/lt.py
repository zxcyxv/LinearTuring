"""LT — LinearTuring 코어(core/model1.py)를 URM 학습 하네스에 태우는 어댑터. R1B8 구조 단일판.

세그먼트(하네스 loop 1회) = [ 경계 → 입력 주입 → dt=1 완보 1스텝 ] × blocks_per_seg
  경계:  h ← h + W_d[ act(W_g h) ⊙ (W_u h) ]     act = g/2 (bilinear=True, 활성화 0개) 또는 silu
  주입:  h ← h + β·Embed(x)                        β 학습 스칼라 (inj_gate)
  완보:  h ← Φ_{1/2}( Φ_{1/2}(h) + f(Φ_{1/2}(h)) )   f = 코어 벡터장 (sheaf 값경로 + Λ full + b)
세그먼트 사이는 하네스가 h.detach() → BPTT 창 = 세그먼트 1개.
ACT 없음: halted = steps >= loops. q_head 는 하네스 손실 호환용 더미(영가중치·bias −5).

체크포인트 호환: 파라미터 이름은 원판(main 브랜치 urm_port/lt.py)과 동일. 코어의 w_ov·w_bo 는
이 경로에서 쓰이지 않는 유령 파라미터이나 strict 로드를 위해 남겨 둔다 (1.38M, 실사용 12.70M).
"""
import os
import sys
from typing import Dict, Optional
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import pydantic

_ROOT = os.environ.get("LT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "core"))
from model1 import Model1                                   # noqa: E402
from models.common import trunc_normal_init_                # noqa: E402  (URM)
from models.sparse_embedding import CastedSparseEmbedding   # noqa: E402  (URM)


@dataclass
class LTCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class LTConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")   # 하네스가 주입하는 causal 등 허용
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0
    hidden_size: int
    num_heads: int
    R: int = 1                  # dt = 1/R. R1B8 = 1 (완보). R>1 이면 블록당 R 마이크로스텝(τ=1 흐름)
    loops: int = 16             # 세그먼트 수
    grid: int = 9
    blocks_per_seg: int = 8     # 세그당 [경계→주입→스텝] 블록 수
    block_inj: bool = True      # 블록마다 재주입 (False 면 세그 첫 블록만)
    bilinear: bool = True       # 경계 게이트 g/2 (활성화 0개). False 면 silu
    mlp_expansion: float = 4.0
    inj_gate_init: float = 0.25
    ckpt: bool = False          # 스텝 단위 gradient checkpointing
    amp: bool = True            # 세그먼트 forward bf16 autocast, 로짓·carry 는 fp32
    forward_dtype: str = "float32"


class LT_Inner(nn.Module):
    def __init__(self, config: LTConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g, d = config.seq_len, config.grid, config.hidden_size
        assert T == g * g
        pos = torch.stack([torch.arange(T).float() // g, torch.arange(T).float() % g], 1)
        self.core = Model1(d=d, H=config.num_heads, R=config.R, n_classes=config.vocab_size,
                           positions=pos, vocab=config.vocab_size, pool=False,
                           sheaf=True, lam_mode="full", alpha_per_head=True,
                           boundary_wo=True, wo_mode="contract")   # w_bo: 체크포인트 호환용 유령
        self.puzzle_emb_ndim = config.puzzle_emb_ndim
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0,
                                                    cast_to=self.forward_dtype)
        self.q_head = nn.Linear(d, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)
        self.inj_gate = nn.Parameter(torch.tensor(float(config.inj_gate_init)))
        # 경계: down 영 초기화 → 시작 시 항등. b_carry 는 잔차 계수(1 고정, 호환용 파라미터)
        inter = int(config.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.b_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()
        self.b_carry = nn.Parameter(torch.tensor(1.0))
        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(d, dtype=self.forward_dtype), std=1.0), persistent=True)

    def empty_carry(self, batch_size: int) -> LTCarry:
        return LTCarry(current_hidden=torch.empty(
            batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype))

    def reset_carry(self, reset_flag: torch.Tensor, carry: LTCarry) -> LTCarry:
        return replace(carry, current_hidden=torch.where(
            reset_flag.view(-1, 1, 1), self.init_hidden, carry.current_hidden))

    def _act(self, g):
        return 0.5 * g if self.config.bilinear else F.silu(g)

    def _boundary(self, h):
        gate, up = self.b_gate_up(h).chunk(2, dim=-1)
        return self.b_carry * h + self.b_down(self._act(gate) * up)

    def _injection(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inj = self.core.embed(batch["inputs"].to(torch.long))            # [B,T,d]
        if self.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])            # [B,ndim]
            pad = self.config.hidden_size - self.puzzle_emb_ndim
            if pad > 0:
                pe = F.pad(pe, (0, pad))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)
        return inj

    def forward(self, carry: LTCarry, batch: Dict[str, torch.Tensor]):
        if self.config.amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                new_carry, logits, q = self._forward(carry, batch)
            return (replace(new_carry, current_hidden=new_carry.current_hidden.float()),
                    logits.float(), (q[0].float(), q[1].float()))
        return self._forward(carry, batch)

    def _forward(self, carry: LTCarry, batch: Dict[str, torch.Tensor]):
        m = self.core
        h = carry.current_hidden
        inj = self._injection(batch)
        fc = m.kernel_fast(); AB = m.W_C(); dt = 1.0 / m.R

        def micro(hh):
            hh = m.phi(hh, dt / 2)
            f, *_ = m.field(hh, None, None, None, AB, fast_ctx=fc)
            return m.phi(hh + dt * f, dt / 2)

        use_ckpt = self.config.ckpt and self.training and torch.is_grad_enabled()
        for blk in range(self.config.blocks_per_seg):
            h = self._boundary(h)
            if blk == 0 or self.config.block_inj:
                h = h + self.inj_gate * inj
            for _ in range(m.R):
                h = checkpoint(micro, h, use_reentrant=False) if use_ckpt else micro(h)
        new_carry = replace(carry, current_hidden=h.detach())
        logits = m.w_cls(h)                                              # [B,T,vocab]
        q = self.q_head(h.mean(1)).to(torch.float32)
        return new_carry, logits, (q[..., 0], q[..., 1])


class LT(nn.Module):
    """URM 모델 클래스 미러 — ACT 제거판 (halted = steps >= loops)."""
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
            halted = new_steps >= self.config.loops
        return (
            LTCarry(current_hidden=new_inner.current_hidden, steps=new_steps,
                    halted=halted, current_data=new_current_data),
            outputs,
        )


def load_lt(ckpt_path: str, device: str = "cuda", **overrides) -> LT:
    """체크포인트(.pt) 로부터 LT 를 구성·로드. config 는 함께 저장된 <run>_config.yaml 또는 기본값 R1B8.
    분석 스크립트 공용 진입점."""
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = st["model_state_dict"] if "model_state_dict" in st else st
    sd = {k.replace("_orig_mod.", "").split("model.", 1)[-1] if k.startswith(("model.", "_orig_mod.")) else k: v
          for k, v in sd.items()}
    cfg = dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832,
               hidden_size=832, num_heads=8, R=1, loops=16, blocks_per_seg=8, block_inj=True,
               bilinear=True, amp=False, ckpt=False, causal=False)
    cfg.update(overrides)
    with torch.device(device):
        m = LT(cfg)
    m.load_state_dict(sd, strict=True)
    return m.eval()
