# -*- coding: utf-8 -*-
"""LinearTuring / minimal(LT) 스도쿠 학습 — Kaggle 노트북 셀 1개용 단독 스크립트.

이 파일이 무엇인가
    저장소 `clean` 브랜치의 `core/minimal.py` 모델을 URM 하네스(`refs/URM/pretrain.py`) 없이
    **처음부터 끝까지** 학습시킨다. 모델·손실·옵티마이저·데이터·학습 루프·EMA·eval·체크포인트가
    전부 이 한 파일 안에 있고, 외부 저장소 import 가 없다. 노트북 셀에 통째로 붙여넣으면 돈다.

무엇을 재현하는가 (기준: `checkpoints/R1B8_min_faith_config.yaml` + `stdp_window=psi`)
    1 step = `model(carry, batch)` 1회 = 세그먼트 1개(블록 8개) fwd+bwd+opt.step().
    carry 는 스텝을 넘어 유지되고 세그먼트 끝에서 detach. ACT 없음 → halted = (steps >= loops=16).
    1 에폭 = 퍼즐 그룹 1,000개 × 그룹당 1샘플 = 1,000 예제, global_batch_size=128 → 7.8125 step/epoch.
    총 스텝 = int(epochs × 1000 / 128) = 390,625 (epochs=50000). 실제로 도는 것은 eval 구간 경계에서
    마지막 부분배치를 버리므로 200 iter × 1,953 = 390,600 (HANDOVER 의 '계획 390,600 step' 과 일치).

URM 하네스에서 무엇을 어떻게 옮겼는가
    - 모델: `core/minimal.py` 의 `LTConfig`/`LTCarry`/`LT_Inner`/`LT` **수식·초기화·draw 순서 그대로**.
      바뀐 것은 (a) `pydantic.BaseModel` → 평범한 `dataclass`(수치 무관, 의존성 축소),
      (b) URM import 두 개를 이 파일 안의 동등 구현으로 교체. 그 외 한 글자도 안 바꿨다.
    - `trunc_normal_init_`: `refs/URM/models/common.py` 에서 그대로 복사.
    - `CastedSparseEmbedding` + 전용 sign-SGD: `refs/URM/models/sparse_embedding.py` 를 **그대로 이식**.
      평범한 `nn.Parameter` 로 바꾸지 않은 근거는 §옵티마이저의 주석 참조 (동등하지 않다).
      단 `forward` 의 범위 검사 `if torch.any(...)` 만 제거 — 데이터 의존 분기라 dynamo 가 그래프를 끊는다.
    - `stablemax_cross_entropy` / `ACTLossHead`(accuracy·exact_accuracy 정의): `refs/URM/models/losses.py` 그대로.
    - `AdamATan2`: `sudoku/adam_atan2.py` 그대로 인라인.
    - weight decay 그룹 분리: `sudoku/urm_patches.diff` 의 `pretrain.py:229` 패치 그대로.
      (psi/theta/alpha_raw/gamma_raw/inj_gate/st_gain/gain_raw/eta_raw/lam_raw/beta/mu 와 1차원 파라미터는 wd=0.
       빠지면 wd=1.0 이 약한 그래디언트의 커널을 (1−lr·wd)^t 로 잠식해 학습이 망가진다.)
    - 학습 루프·loss_scale·all-reduce·EMA·lr 스케줄: `pretrain.py` 의 `train_batch`(:522)·`launch`(:838) 의미론 그대로.
    - 데이터: `refs/URM/puzzle_dataset.py` 의 에폭/배치 프로토콜 + `data/build_sudoku_dataset.py` 의 증강 규칙.
      증강 1,000배를 파일로 안 받고 학습 중 생성한다 (§데이터 주석에 동등성 논증).

Kaggle 제약 (자세한 것은 kaggle/예외사항.md)
    * `notebook_launcher` 는 **부모(노트북) 프로세스에서 CUDA 가 초기화되어 있으면 실패**한다
      (fork 로 자식을 띄우기 때문). 그래서 이 파일은 **import 시점에 `torch.cuda` 를 절대 건드리지 않는다.**
      GPU 를 쓰는 코드는 전부 `main()` 안에 있다. 셀에서 이 파일 앞에 `torch.cuda.*` 를 호출하지 말 것
      (`nvidia-smi` 는 별개 프로세스라 무해). 유일한 예외가 `run()` 의 `torch.cuda.device_count()` 인데,
      이것은 NVML 경로라 컨텍스트를 만들지 않는다(accelerate 자신도 같은 호출을 한다).
    * 12시간 세션 제한 → `CFG["max_hours"]` 가 지나면 체크포인트를 쓰고 정상 종료한다. 재개는 필수 기능.

사용
    # 노트북 셀: 이 파일 전체를 붙여넣고 실행. 필요하면 맨 위 CFG 만 고친다.
    # 셸:      python train_kaggle.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# import — 여기서 CUDA 를 건드리는 것은 하나도 없다 (notebook_launcher 제약)
# ─────────────────────────────────────────────────────────────────────────────
import copy
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset


# ─────────────────────────────────────────────────────────────────────────────
# CFG — 셀에서 한 줄로 바꾸는 곳
# ─────────────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = {                       # 판 사이의 차이는 이 세 플래그뿐
    "v1":   dict(legacy_gauge=True,  block_order="pre",  use_trace=False),   # 9/1 원본
    "v1.1": dict(legacy_gauge=False, block_order="pre",  use_trace=False),   # √d 고정 게이지. 학습·측정 완료 (최신)
    "v2":   dict(legacy_gauge=False, block_order="post", use_trace=False),   # v1.1 + post 순서. 아직 학습 안 함
    "v1.7": dict(legacy_gauge=False, block_order="post", use_trace=True),    # v2 + 주소 흔적 z. 학습 후반 불안정
}

CFG = dict(
    # ---- 데이터 (data/prep_dataset.py 가 만든 npz: 1k 퍼즐 × 증강 1000, 테스트 2048)
    data_npz=os.path.join(REPO, "data/sudoku_lt_1k.npz"),
    num_aug=1000,
    test_size=2048,
    # ---- 모델
    hidden_size=832, num_heads=8, loops=16, blocks_per_seg=8, num_layers=1, grid=9, vocab_size=11,
    mlp_expansion=4.0, alpha_init=0.1, dist_decay=True, eps=1e-4, psi_zero=False,
    puzzle_emb_ndim=832, forward_dtype="float32", amp=True,
    legacy_gauge=True, inj_gate_init=0.25, gamma_init=0.1,          # preset 이 덮어씀
    block_order="pre", use_trace=False, trace_rho_init=0.5,          # preset 이 덮어씀
    stdp=True, stdp_eta_init=0.05, stdp_gain_init=1.0, stdp_lam_init=0.25, stdp_gain_fixed=-1.0, stdp_lam_fixed=-1.0,
    # ---- 학습 (URM yaml 그대로)
    global_batch_size=128, epochs=50000, lr=1e-4, lr_min_ratio=1.0, lr_warmup_steps=2000, weight_decay=1.0,
    beta1=0.9, beta2=0.95, puzzle_emb_lr=1e-4, puzzle_emb_weight_decay=1.0, grad_accum_steps=1, q_weight=0.5,
    seed=0, ema=True, ema_rate=0.999, eval_interval=250, compile=True, inductor_no_persist=True,
    # ---- 실행/저장
    out_dir=None,                 # None → runs/<preset>
    resume_from=None,             # None → out_dir 에서 최신 step_*.pt 자동 재개
    require_resume=False,
    keep_last=2, save_every_steps=2000,
    milestone_every=10000,        # milestones/step_N.pt 영구 저장 + EMA 로 seg128 외삽 → milestones/extrap_step_N.txt
    milestone_extrap_segs=128, milestone_extrap_n=512,
    max_hours=1e9, max_steps=None, log_every=250, dataloader_workers=1,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. URM `models/common.py` — 그대로 복사
# ─────────────────────────────────────────────────────────────────────────────
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# 2. URM `models/sparse_embedding.py` — 그대로 이식 (범위 검사만 제거)
#
# 왜 평범한 nn.Parameter 로 바꾸지 않았는가 (판단 근거):
#   내용상으로는 `num_puzzle_identifiers=1` 이라 이 임베딩은 전역 편향 벡터 1개가 맞다. 그러나
#   **갱신 규칙이 다르다.** 원본에서 이 버퍼는 `CastedSparseEmbeddingSignSGD_Distributed` 가 맡고,
#       p ← p·(1 − lr·wd) − lr·sign(Σ_batch Σ_rank g)
#   즉 스텝 크기가 정확히 lr 인 sign-SGD 다. 반면 나머지 파라미터를 맡은 AdamATan2 는
#       p ← p·(1 − lr·wd) − lr·atan2(m̂, √v̂)
#   이고, 그래디언트 부호가 일정한 극한에서도 atan2(g,|g|) = π/4 ≈ 0.785 라 **스텝이 0.785배**이며
#   모멘텀(β₁=0.9)·2차 모멘트(β₂=0.95)의 과도 응답도 다르다. 따라서 nn.Parameter+AdamATan2 대체는
#   수치적으로 동등하지 않다 → 요구사항대로 **원본을 그대로 이식**했다.
#   (그래서 puzzle_emb_lr=1e-4 / puzzle_emb_weight_decay=1.0 도 원본처럼 이 전용 옵티마이저에 그대로 들어간다.)
#
# 제거한 것: forward 의 `if torch.any((inputs < 0) | (inputs >= num_embeddings))` 범위 검사.
#   데이터 의존 분기라 torch.compile 이 여기서 그래프를 9번 끊는다(블록 8개 + q). 값에는 영향이 없다
#   (모든 puzzle_identifier 가 0 이고 num_embeddings=1 이라 검사는 항상 통과).
# ─────────────────────────────────────────────────────────────────────────────
class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.num_embeddings = num_embeddings

        # Real Weights — Truncated LeCun normal init (init_std=0 → 전부 0)
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )
        # Local embeddings, with gradient, not persistent
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        # Local embedding IDs, not persistent (CUDA scatter/gather 가 long 을 요구)
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int64), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].to(self.cast_to)

        # Training mode, fill puzzle embedding from weights
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(self, params, world_size: int, lr=1e-3, weight_decay: float = 1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, world_size=world_size))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            local_weights_grad = None
            local_ids = None
            weights = None

            assert len(group["params"]) == 3
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False

            assert local_ids is not None
            assert weights is not None

            # Adam ≈ SignSGD if gradient is very sparse
            if local_weights_grad is not None:
                _sparse_emb_signsgd_dist(local_weights_grad, local_ids, weights,
                                         lr=group["lr"], weight_decay=group["weight_decay"],
                                         world_size=group["world_size"])


def _sparse_emb_signsgd_dist(local_weights_grad, local_ids, weights, lr: float, weight_decay: float, world_size: int):
    N, D = local_weights_grad.shape

    # All-gather (원본 그대로. 범위 검사 _validate_ids 는 .item() 동기화를 부르므로 제거 —
    # 식별자가 전부 0 이고 num_embeddings=1 이라 항상 통과하는 검사였다.)
    all_weights_grad = local_weights_grad
    all_ids = local_ids
    if world_size > 1:
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N, dtype=local_ids.dtype, device=local_ids.device)
        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids, local_ids)

    # Unique
    grad_ids, inv = all_ids.unique(return_inverse=True)
    inv = inv.to(torch.int64)
    grad_ids = grad_ids.to(torch.int64)

    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    # SignSGD with decoupled weight decay
    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
    weights[grad_ids] = p


# ─────────────────────────────────────────────────────────────────────────────
# 3. 모델 — core/minimal.py (clean 브랜치, 2026-09-04) 를 **그대로** 옮긴 것.
#    수식·초기화·파라미터 생성 순서(=RNG 소비 순서)를 한 글자도 바꾸지 않았다.
#    바꾼 것은 다음 세 줄뿐이며 전부 수치와 무관하다:
#      (a) LTConfig 의 베이스가 pydantic.BaseModel → dataclass
#      (b) LT.__init__ 의 LTConfig(**d) → LTConfig.from_dict(d)  (모르는 키 무시)
#      (c) trunc_normal_init_ / CastedSparseEmbedding 을 위에서 정의한 동등 구현으로 참조
#    아래 원문 주석(설계 근거)도 그대로 보존한다.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LTCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None
    coupling: Optional[torch.Tensor] = None       # STDP 결합 기억 w [B,H,T,T]
    fresh: Optional[torch.Tensor] = None          # [B] bool — 이 퍼즐의 w·z 가 아직 초기화 전
    trace: Optional[torch.Tensor] = None          # 주소 흔적 z [B,T,H,p,2] (use_trace 일 때만)


@dataclass
class LTConfig:
    """모델 설정. 판(preset) 간 차이는 legacy_gauge · block_order · use_trace 세 플래그뿐이다.
      v1   = legacy_gauge=True,  block_order="pre",  use_trace=False   (9/1 원본)
      v1.1 = legacy_gauge=False, block_order="pre",  use_trace=False   (최신 학습판)
      v2   = legacy_gauge=False, block_order="post", use_trace=False   (미학습)
      v1.7 = legacy_gauge=False, block_order="post", use_trace=True
    """
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0
    hidden_size: int = 832
    num_heads: int = 8
    loops: int = 16
    grid: int = 9
    blocks_per_seg: int = 8     # 세그먼트당 블록 수 (× num_layers)
    num_layers: int = 1         # 가중치 벌 수. 블록 k → layers[k % num_layers]
    mlp_expansion: float = 4.0
    alpha_init: float = 0.1
    legacy_gauge: bool = False  # True: 주입 계수·γ 를 학습 스칼라로 (9/1 원본). False: √d 고정, γ=1/d, 임베딩 init 1/√d
    inj_gate_init: float = 0.25
    gamma_init: float = 0.1
    dist_decay: bool = True     # 거리 감쇠 e^{−α‖Δ‖₁} (헤드별 α)
    eps: float = 1e-4
    amp: bool = True
    forward_dtype: str = "float32"
    psi_zero: bool = False      # ψ ≡ 0 고정 (읽기 커널 대칭)
    stdp: bool = True           # 결합 기억 w ← (1−η)w + η·g·a^β·agree,  읽기 a_eff = (1−λ)a + λw
    stdp_eta_init: float = 0.05
    stdp_gain_init: float = 1.0
    stdp_lam_init: float = 0.25
    stdp_gain_fixed: float = -1.0   # ≥0 이면 g 를 이 값으로 고정
    stdp_lam_fixed: float = -1.0    # ≥0 이면 λ 를 이 값으로 고정
    block_order: str = "pre"    # pre: 경계 → 주입 → 스텝(Φ 포함) | post: 주입 → 스텝 → 경계 → Φ
    use_trace: bool = False     # 쓰기 창의 주소를 흔적 z ← μ⊙z + √(1−ρ²)⊙u 로 (μ_j = ρ_j e^{iω_j})
    trace_rho_init: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "LTConfig":
        known = {f.name for f in fields(cls)}
        dd = {k: v for k, v in d.items() if k in known}
        if "use_trace" not in d and "trace_rho_init" in d:     # v1.6/v1.7 체크포인트 cfg 호환 (use_trace 키가 없던 시절)
            dd["use_trace"] = True
        return cls(**dd)


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class LTLayer(nn.Module):
    """가중치 한 벌. 파라미터 이름은 모든 기존 체크포인트와 동일하다."""

    def __init__(self, config: "LTConfig", H, d, dh, p) -> None:
        super().__init__()
        self.wc_raw = nn.Parameter(torch.randn(H, dh, d) / math.sqrt(d))
        if config.psi_zero:
            self.register_buffer("psi", torch.zeros(H, p), persistent=False)
        else:
            self.psi = nn.Parameter(torch.rand(H, p) * 2 * math.pi - math.pi)
        self.theta = nn.Parameter((torch.rand(H, p, 2) * 2 - 1) * (math.pi / 2))
        self.alpha_raw = nn.Parameter(torch.full((H, 1), inv_softplus(config.alpha_init)))
        w_sh = torch.zeros(H, dh, d)
        for m in range(H):
            w_sh[m, :, m * dh:(m + 1) * dh] = torch.eye(dh)
        self.w_sh = nn.Parameter(w_sh + 0.01 * torch.randn(H, dh, d) / math.sqrt(d))
        if config.use_trace:
            _lg = math.log(config.trace_rho_init / (1 - config.trace_rho_init))
            self.mu_rho_raw = nn.Parameter(torch.full((H, p), _lg))
            self.mu_omega = nn.Parameter((torch.rand(H, p) * 2 - 1) * (math.pi / 2))
        if config.stdp:
            lg = lambda x: math.log(x / (1 - x))
            self.eta_raw = nn.Parameter(torch.full((H, 1, 1), lg(config.stdp_eta_init)))
            self.lam_raw = nn.Parameter(torch.full((H, 1, 1), lg(config.stdp_lam_init)))
            self.gain_raw = nn.Parameter(torch.full((H, 1, 1), inv_softplus(config.stdp_gain_init)))
            self.beta = nn.Parameter(torch.zeros(H, p))       # 쓰기 창의 위상 (ψ 와 별개)
            self.beta.data.normal_(0.0, 0.5)
        inter = int(config.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.b_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()

    @property
    def alpha(self): return F.softplus(self.alpha_raw)


class LT_Inner(nn.Module):
    def __init__(self, config: LTConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g, d, H = config.seq_len, config.grid, config.hidden_size, config.num_heads
        assert T == g * g and d % H == 0 and (d // H) % 2 == 0
        self.d, self.H = d, H
        self.dh = d // H
        self.p = self.dh // 2
        u = torch.arange(T).float() // g; w = torch.arange(T).float() % g
        self.register_buffer("pos_u", u, persistent=False); self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("l1", (u[:, None] - u[None]).abs() + (w[:, None] - w[None]).abs(), persistent=False)
        self.embed = nn.Embedding(config.vocab_size, d)
        if config.legacy_gauge:
            self.embed_scale = nn.Parameter(torch.tensor(float(config.inj_gate_init)))
            self.gamma_raw = nn.Parameter(torch.tensor(inv_softplus(config.gamma_init)))
        else:
            self.gamma = 1.0 / d
            self.embed_scale = math.sqrt(d)
            with torch.no_grad():
                trunc_normal_init_(self.embed.weight, std=1.0 / self.embed_scale)
        self.w_cls = nn.Linear(d, config.vocab_size)
        assert config.num_layers >= 1
        assert config.block_order in ("pre", "post"), f"block_order: pre | post (받은 값 {config.block_order})"
        self.stdp = config.stdp
        self.use_trace = config.use_trace
        self.layers = nn.ModuleList([LTLayer(config, H, d, self.dh, self.p) for _ in range(config.num_layers)])
        if config.num_layers > 1:
            sd0 = self.layers[0].state_dict()
            for Lx in self.layers[1:]:
                Lx.load_state_dict({k: v.clone() for k, v in sd0.items()})
        self.puzzle_emb_ndim = config.puzzle_emb_ndim
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)
        self.init_hidden = nn.Buffer(trunc_normal_init_(torch.empty(d, dtype=self.forward_dtype), std=1.0), persistent=True)

    # ---------------------------------------------------------------- 부품 (L = 레이어)
    def W_C(self, L):
        """행직교 [A;B] (헤드별 dh×d): raw 의 전치를 QR 해 열직교 → 전치."""
        Q, _ = torch.linalg.qr(L.wc_raw.transpose(-1, -2))               # [H,d,dh]
        AB = Q.transpose(-1, -2)
        return AB[:, :self.p, :], AB[:, self.p:, :]


    def kernel(self, L, psi=None):
        """decay_h [H,T,T], 위상각 A_t = ψ/2 + θ·pos_t (q), B_t = −ψ/2 + θ·pos_t (k) → cos/sin [T,H,p]. psi 를 주면 그 위상차로 (STDP 창 β 용)."""
        psi = L.psi if psi is None else psi
        decay_h = torch.exp(-L.alpha[:, 0, None, None] * self.l1) if self.config.dist_decay else torch.ones_like(self.l1).expand(L.alpha.shape[0], -1, -1)
        ppos = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w         # [H,p,T]
        A = (ppos + psi[..., None] / 2).permute(2, 0, 1); B = (ppos - psi[..., None] / 2).permute(2, 0, 1)
        return decay_h, torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)


    def attn_xy(self, xy, kc):
        """정규화된 주소에서 커널 kc 로 a 를 만든다 (회전 + 내적 + 감쇠)."""
        x, y = xy; decay_h, cosA, sinA, cosB, sinB = kc
        qx = x * cosA - y * sinA; qy = x * sinA + y * cosA
        kx = x * cosB - y * sinB; ky = x * sinB + y * cosB
        a = torch.einsum('bthj,bnhj->bhtn', qx, kx) + torch.einsum('bthj,bnhj->bhtn', qy, ky)
        return a * decay_h.unsqueeze(0)


    def phi(self, h):
        g = F.softplus(self.gamma_raw) if hasattr(self, "gamma_raw") else self.gamma
        return h / torch.sqrt(1.0 + g * h.pow(2).sum(-1, keepdim=True))


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


    def addr_raw(self, h, AB):
        """정규화 전 복소 주소 u = W_C h 의 (실부, 허부) [B,T,H,p]."""
        A, Bm = AB
        return torch.einsum('btd,hjd->bthj', h, A), torch.einsum('btd,hjd->bthj', h, Bm)

    def _unit(self, x, y):
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()
        return x / (nrm + self.config.eps), y / (nrm + self.config.eps)

    def addr(self, h, AB):
        """정규화된 순간 복소 주소 û (분석 스크립트 호환)."""
        return self._unit(*self.addr_raw(h, AB))

    def trace_step(self, L, ux, uy, ztr, fresh):
        """흔적 z ← μ⊙z + √(1−ρ²)⊙u,  μ_j = ρ_j e^{iω_j}.  풀면 z_j(k) = Σ_m ρ_j^m e^{iω_j m} u_j(k−m)."""
        rho = torch.sigmoid(L.mu_rho_raw)
        cw, sw = torch.cos(L.mu_omega), torch.sin(L.mu_omega)
        gin = torch.sqrt(torch.clamp(1.0 - rho * rho, min=1e-6))
        if ztr is None:
            zx, zy = gin * ux, gin * uy
        else:
            zx0, zy0 = ztr[..., 0], ztr[..., 1]
            if fresh is not None:
                m = fresh.view(-1, 1, 1, 1)
                zx0 = torch.where(m, torch.zeros_like(zx0), zx0)
                zy0 = torch.where(m, torch.zeros_like(zy0), zy0)
            zx = rho * (cw * zx0 - sw * zy0) + gin * ux
            zy = rho * (sw * zx0 + cw * zy0) + gin * uy
        return zx, zy, torch.stack((zx, zy), dim=-1)

    def step(self, L, h, AB, kc, w=None, fresh=None, kcb=None, ztr=None, apply_phi=True):
        """한 블록의 어텐션+수송.  읽기 a(û, ψ) · 쓰기 창 a^β(û 또는 ẑ, β) × agree · w 누적 · a_eff 로 v 수송."""
        ux, uy = self.addr_raw(h, AB)
        uh = self._unit(ux, uy)                                             # 순간 주소 û
        a = self.attn_xy(uh, kc)                                            # 읽기 커널 (ψ, 거리감쇠)
        v = torch.einsum('btd,hcd->bthc', h, L.w_sh)                        # 값 사영
        ztr_new = None
        if self.stdp:
            if self.use_trace:
                zx, zy, ztr_new = self.trace_step(L, ux, uy, ztr, fresh)
                win = self.attn_xy(self._unit(zx, zy), kcb)                 # 쓰기 창 on 흔적 ẑ
            else:
                win = self.attn_xy(uh, kcb)                                 # 쓰기 창 on 순간 û
            vv = v / (v.norm(dim=-1, keepdim=True) + self.config.eps)
            agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
            G = win * agree
            gain = F.softplus(L.gain_raw) if self.config.stdp_gain_fixed < 0 else float(self.config.stdp_gain_fixed)
            tgt = gain * G
            eta = torch.sigmoid(L.eta_raw)
            lam = torch.sigmoid(L.lam_raw) if self.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(self.config.stdp_lam_fixed))
            if w is None:
                w = tgt
            else:
                w = torch.where(fresh.view(-1, 1, 1, 1), tgt, w) if fresh is not None else w
                w = (1 - eta) * w + eta * tgt
            a = (1 - lam) * a + lam * w
        o = torch.einsum('bhtn,bnhc->bthc', a, v)
        f = torch.einsum('bthc,hcd->btd', o, L.w_sh)
        hout = self.phi(h + f) if apply_phi else (h + f)
        return hout, w, ztr_new

    def boundary(self, L, h):
        g, u = L.b_gate_up(h).chunk(2, dim=-1)
        return h + L.b_down(0.5 * g * u)

    def _forward(self, carry, batch):
        h = carry.current_hidden; inj = self.injection(batch)
        ABs = [self.W_C(L) for L in self.layers]
        kcs = [self.kernel(L) for L in self.layers]
        kcbs = [self.kernel(L, L.beta) if self.stdp else None for L in self.layers]
        w = carry.coupling if self.stdp else None; fresh = carry.fresh if self.stdp else None
        ztr = carry.trace if self.use_trace else None
        pre = self.config.block_order == "pre"
        for _ in range(self.config.blocks_per_seg):
            for li, L in enumerate(self.layers):
                AB, kc, kcb = ABs[li], kcs[li], kcbs[li]
                if pre:
                    h = self.boundary(L, h)
                h = h + self.embed_scale * inj
                h, w, ztr = self.step(L, h, AB, kc, w, fresh, kcb, ztr, apply_phi=pre)
                if not pre:
                    h = self.boundary(L, h)
                    h = self.phi(h)
                fresh = None
        return replace(carry, current_hidden=h.detach(), coupling=(w.detach() if w is not None else None),
                       trace=(ztr.detach() if ztr is not None else None), fresh=None), self.w_cls(h)


class LT(nn.Module):
    """URM 하네스 인터페이스. ACT 없음 (halted = steps ≥ loops). q 로짓은 상수."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LTConfig.from_dict(config_dict)
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
        return LTCarry(current_hidden=inner.current_hidden, steps=steps, halted=halted, current_data=data,
                       coupling=inner.coupling, trace=inner.trace), outputs


# ─────────────────────────────────────────────────────────────────────────────
# 4. 손실 — refs/URM/models/losses.py 그대로.
#    accuracy / exact_accuracy 의 정의를 바꾸지 않았으므로 eval 수치가 저장소 기록과 직접 비교 가능하다.
#    떼어낸 것: moe_aux_loss · router_metrics · profile 분기 (LT 의 outputs 에는 존재하지 않는 키다),
#              return_raw_outputs (미사용). 손실·지표 계산에는 손대지 않았다.
# ─────────────────────────────────────────────────────────────────────────────
IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1),
                           ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, q_weight: float = 0.5):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.q_weight = q_weight   # [절제] 0 이면 q(BCE) 손실 제거 — q_head 동결

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(self, return_keys: Set[str], **model_kwargs):
        # Model logits: B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),

                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses — FIXME(원문): Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"],
                                                         seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({"lm_loss": lm_loss.detach(), "q_halt_loss": q_halt_loss.detach()})

        # Q continue (bootstrapping target loss) — LT 는 target_q_continue 를 내지 않는다
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        returned_outputs: Dict[str, torch.Tensor] = {}
        for k in return_keys:
            if k in outputs:
                returned_outputs[k] = outputs[k].detach()

        total_loss = lm_loss + self.q_weight * (q_halt_loss + q_continue_loss)
        return new_carry, total_loss, metrics, returned_outputs, new_carry.halted.all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. 옵티마이저 — sudoku/adam_atan2.py 그대로 인라인
# ─────────────────────────────────────────────────────────────────────────────
class AdamATan2(Optimizer):
    """수식 (Everett et al. 2024 정준형):
        p ← p · (1 − lr·wd)              # AdamW 식 분리 감쇠
        p ← p − lr · atan2(m̂, √v̂)       # ε 없는 스케일 불변 갱신
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]; lr = g["lr"]; wd = g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1; t = st["step"]
                m, v = st["m"], st["v"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mh = m / (1 - b1 ** t)
                vh = v / (1 - b2 ** t)
                p.mul_(1 - lr * wd)
                p.add_(torch.atan2(mh, vh.sqrt()), alpha=-lr)


# wd 제외 키 — sudoku/urm_patches.diff 의 pretrain.py:229 패치와 **문자열까지 동일**.
# 동역학 상수(ψ·θ·α·γ·게이트)와 가소성 파라미터(η·λ·G·β·μ), 그리고 1차원 이하 파라미터는 wd=0.
# 근거(패치 주석): wd=1.0 이 약한 그래디언트의 커널을 (1−lr·wd)^t 로 잠식 → loss 역행.
NO_DECAY_KEYS = ("psi", "theta", "alpha_raw", "gamma_raw", "inj_gate", "st_gain",
                 "gain_raw", "eta_raw", "lam_raw", "beta", "mu")


def _is_no_decay(name: str, p: torch.Tensor) -> bool:
    return p.ndim <= 1 or name.endswith(".b") or any(k in name for k in NO_DECAY_KEYS)


def create_optimizers(model: nn.Module, cfg: dict, world_size: int):
    """pretrain.py:224-248 의 옵티마이저 구성 그대로 (use_muon=False 경로)."""
    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),   # weights / local_weights / local_ids 3개
            lr=0,                               # 스케줄러가 매 스텝 채운다
            weight_decay=cfg["puzzle_emb_weight_decay"],
            world_size=world_size,
        ),
        AdamATan2(
            [{"params": [p for n, p in model.named_parameters() if _is_no_decay(n, p)],
              "weight_decay": 0.0},
             {"params": [p for n, p in model.named_parameters() if not _is_no_decay(n, p)],
              "weight_decay": cfg["weight_decay"]}],
            lr=0,                               # 스케줄러가 매 스텝 채운다
            weight_decay=cfg["weight_decay"],
            betas=(cfg["beta1"], cfg["beta2"]),
        ),
    ]
    optimizer_lrs = [cfg["puzzle_emb_lr"], cfg["lr"]]
    return optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(current_step: int, *, base_lr: float, num_warmup_steps: int,
                                          num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5):
    """pretrain.py:252 그대로. min_ratio=1.0 이면 warmup 후 상수 lr."""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


# ─────────────────────────────────────────────────────────────────────────────
# 6. EMA — pretrain.py:30 의 EMAHelper 그대로 (shadow ← μ·shadow + (1−μ)·param).
#    다른 점 하나: eval 시 원본은 `copy.deepcopy(model)` 로 사본을 떠 EMA 를 심는다.
#    여기서는 파라미터를 제자리 교환(raw 백업 → shadow 복사 → eval → 복원)한다.
#    수치는 동일하고(같은 텐서 값을 읽는다) 메모리와 재컴파일을 아낀다 — torch.compile 된 모듈을
#    deepcopy 하면 그래프를 새로 뜬다. Kaggle 에서는 이게 실제로 문제가 된다.
# ─────────────────────────────────────────────────────────────────────────────
class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class _EMASwap:
    """with 블록 동안 모델 파라미터를 EMA shadow 로 바꿔 끼운다."""
    def __init__(self, module, ema: Optional[EMAHelper]):
        self.module, self.ema, self.backup = module, ema, {}

    def __enter__(self):
        if self.ema is None:
            return self.module
        with torch.no_grad():
            for name, param in self.module.named_parameters():
                if param.requires_grad and name in self.ema.shadow:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.ema.shadow[name].data)
        return self.module

    def __exit__(self, *exc):
        if self.ema is None:
            return False
        with torch.no_grad():
            for name, param in self.module.named_parameters():
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup.clear()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 7. 데이터 — 1,000개만 올리고 증강은 여기서 만든다
#
# (a) 증강 규칙: refs/URM/data/build_sudoku_dataset.py 의 `shuffle_sudoku` 를 그대로 옮겼다.
#     자릿수 치환(1..9 의 순열, 0=빈칸 불변) · 전치(p=0.5) · 밴드 셔플(3! × 밴드 안 3!) ·
#     스택 셔플(3! × 스택 안 3!). 적용 순서도 원본과 같다: 전치 → 위치 매핑 → 자릿수 매핑.
#     `_apply_aug` 가 원본 `apply_transformation` 과 **같은 파라미터에서 같은 출력**을 냄을
#     kaggle/README 없이도 재확인할 수 있게 self-test 로 남겨 두었다(파일 끝 `--selftest`).
#
# (b) 왜 "고정 풀" 인가: 원본 데이터셋은 퍼즐 1개당 증강 1,000개를 **빌드 시점에 한 번 뽑아 고정**한다.
#     그래서 aug_idx=k 를 다시 뽑으면 항상 같은 판이 나오고, 50,000 에폭 동안 각 증강은 ~50회 반복된다.
#     매번 새 증강을 뽑으면(무한 풀) 과적합 압력이 달라져 저장소 기록(76k·123k step 곡선)과 비교가 깨진다.
#     따라서 여기서는 (퍼즐 id, aug_idx) 로 결정되는 Philox 스트림에서 변환을 뽑아 **고정 풀을 재현**한다.
#     같은 (id, aug_idx) → 항상 같은 판. 세션을 나눠 재개해도 같다.
#
# (c) 에폭 프로토콜: refs/URM/puzzle_dataset.py `_iter_train` 그대로.
#     그룹 1,000개(= 원본 퍼즐 1,000개)를 매 에폭 섞고, 그룹마다 [0, num_aug] 에서 1개를 균등추출한다
#     (0 = 무증강 원본). 그룹당 예제 1개이므로 에폭당 정확히 1,000 예제. 배치는 에폭 경계를 넘어
#     연속으로 채우고, 마지막 부분배치는 버린다. RNG 소비 순서까지 URM 과 맞춰 두었으므로
#     (seed 가 같으면) "몇 번째 스텝에 어느 퍼즐의 몇 번 증강이 오는가" 가 URM 런과 동일하다.
#
# (d) 유출: 증강은 train 에만 건다(원본 `num_augments = num_aug if set_name=="train" else 0`).
#     test 는 다른 CSV 에서 온 앞 2,048개를 손대지 않고 쓴다. 2026-09-01/README §7 의 점검 결론이
#     그대로 유지되는 근거: 그 점검은 "테스트 퍼즐이 학습 퍼즐의 **궤도**(자릿수 9! × 전치 2 ×
#     밴드 1296 × 스택 1296) 안에 있는가" 를 물어 전수 검증으로 0 을 얻었다. 궤도는 대표원을 어떻게
#     뽑든 같은 집합이므로, 우리가 어떤 증강을 새로 뽑아도 그 판은 이미 "테스트와 서로소" 로 검증된
#     궤도 안에 머문다. 즉 증강 방법을 파일→즉석 생성으로 바꿔도 유출 성질은 불변이다.
# ─────────────────────────────────────────────────────────────────────────────
_ROW_OF = np.arange(81) // 9
_COL_OF = np.arange(81) % 9


def _draw_aug_params(rng):
    """build_sudoku_dataset.shuffle_sudoku 의 파라미터 추출부 — 순서·의미 동일."""
    digit_map = np.pad(rng.permutation(np.arange(1, 10)), (1, 0))       # 0(빈칸)은 불변
    transpose_flag = rng.random() < 0.5
    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])
    return digit_map, transpose_flag, row_perm, col_perm


def _apply_aug(x: np.ndarray, digit_map, transpose_flag, row_perm, col_perm) -> np.ndarray:
    """원본 `apply_transformation` 과 동일: 전치 → 81→81 위치 매핑 → 자릿수 매핑."""
    mapping = row_perm[_ROW_OF] * 9 + col_perm[_COL_OF]
    if transpose_flag:
        x = x.T
    new_board = x.flatten()[mapping].reshape(9, 9)
    return digit_map[new_board]


class SudokuTrainDataset(IterableDataset):
    """URM PuzzleDataset(_iter_train) 을 스도쿠 1k-aug-N 구조에 특화해 옮긴 것.

    구조상 group_indices = arange(0, n*(1+num_aug)+1, 1+num_aug), puzzle_indices = arange(...),
    mean_puzzle_examples = 1 이므로 원본의 인덱스 배열을 파일로 들고 다닐 필요가 없다.
    """

    def __init__(self, inputs: np.ndarray, labels: np.ndarray, *, seed: int, num_aug: int,
                 global_batch_size: int, rank: int, world_size: int, epochs_per_iter: int,
                 start_iter: int, total_iters: int, skip_batches: int = 0):
        super().__init__()
        self.inputs, self.labels = inputs, labels
        self.seed, self.num_aug = seed, num_aug
        self.gbs, self.rank, self.world_size = global_batch_size, rank, world_size
        self.local_bs = global_batch_size // world_size
        self.epochs_per_iter = epochs_per_iter
        self.start_iter, self.total_iters, self.skip_batches = start_iter, total_iters, skip_batches
        self.n_groups = len(inputs)
        self.gsize = 1 + num_aug
        # 증강 풀 키: seed 로만 갈린다 (rank·세션과 무관해야 "고정 풀" 이 된다)
        self._aug_key0 = np.uint64(0x9E3779B97F4A7C15 ^ np.uint64(seed))

    def _augmented(self, gid: int, aug_idx: int):
        if aug_idx == 0:                       # 원본과 동일하게 첫 인덱스는 무증강
            return self.inputs[gid], self.labels[gid]
        key = np.array([self._aug_key0, np.uint64(gid * self.gsize + aug_idx)], dtype=np.uint64)
        rng = np.random.Generator(np.random.Philox(key=key))
        prm = _draw_aug_params(rng)
        return _apply_aug(self.inputs[gid], *prm), _apply_aug(self.labels[gid], *prm)

    def __iter__(self):
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        assert wi is None or wi.num_workers == 1, "Multithreaded data loading is not currently supported."

        skip = self.skip_batches
        for it in range(self.start_iter, self.total_iters):
            # URM: self._iters += 1 후 Philox(seed + self._iters) — iter 0 이 seed+1 을 쓴다
            rng = np.random.Generator(np.random.Philox(seed=self.seed + it + 1))
            group_order = np.concatenate([rng.permutation(self.n_groups) for _ in range(self.epochs_per_iter)])

            start_index = 0
            while start_index < group_order.size:
                # ---- _sample_batch: 그룹을 하나씩 집어 배치를 채운다 (퍼즐 크기 1 → 그룹당 정확히 1개)
                gsel, asel = [], []
                while (start_index < group_order.size) and (len(gsel) < self.gbs):
                    g = int(group_order[start_index])
                    pid = int(rng.integers(g * self.gsize, (g + 1) * self.gsize))
                    start_index += 1
                    rng.choice(1, 1, replace=False)     # URM 의 퍼즐 내 예제 선택 — 스트림 동기화용
                    gsel.append(g); asel.append(pid - g * self.gsize)

                if len(gsel) < self.gbs:               # Drop last batch (URM 과 동일)
                    break

                if skip > 0:                            # 재개 fast-forward: RNG 만 소비하고 보드는 안 만든다
                    skip -= 1
                    continue

                off = self.rank * self.local_bs
                inp = np.empty((self.local_bs, 81), dtype=np.int32)
                lab = np.empty((self.local_bs, 81), dtype=np.int32)
                for j in range(self.local_bs):
                    bi, bl = self._augmented(gsel[off + j], asel[off + j])
                    # `_seq_to_numpy` 의 arr + 1 (PAD=0, 빈칸=1, 숫자 1..9 → 2..10)
                    inp[j] = bi.reshape(81).astype(np.int32) + 1
                    lab[j] = bl.reshape(81).astype(np.int32) + 1
                # ignore_label_id=0 이지만 라벨은 전부 ≥1 이므로 -100 치환 대상이 없다 (URM 도 동일).
                yield it, {
                    "inputs": torch.from_numpy(inp),
                    "labels": torch.from_numpy(lab),
                    "puzzle_identifiers": torch.zeros(self.local_bs, dtype=torch.int32),
                }


def eval_batches(inputs: np.ndarray, labels: np.ndarray, global_batch_size: int, rank: int, world_size: int):
    """URM `_iter_test` 그대로: 전역 배치 단위로 자르고 각 랭크가 자기 구간만 든다. 증강 없음."""
    local_bs = global_batch_size // world_size
    total = len(inputs)
    start = 0
    while start < total:
        end = min(total, start + global_batch_size)
        ls = start + rank * local_bs
        le = min(start + (rank + 1) * local_bs, end)
        if le > ls:
            yield {
                "inputs": torch.from_numpy(inputs[ls:le].reshape(le - ls, 81).astype(np.int32) + 1),
                "labels": torch.from_numpy(labels[ls:le].reshape(le - ls, 81).astype(np.int32) + 1),
                "puzzle_identifiers": torch.zeros(le - ls, dtype=torch.int32),
            }
        start += global_batch_size


# ─────────────────────────────────────────────────────────────────────────────
# 8. 체크포인트 — 저장·탐색·재개
#
#   저장 내용은 저장소 규약(sudoku/urm_patches.diff 의 save_train_state extra)과 같다:
#     model_state_dict   = EMA 가중치 (평가·분석용, 저장소 체크포인트와 같은 의미)
#     raw_model_state_dict = 원시 가중치 (재개는 반드시 이것으로)
#     ema_shadow         = EMA shadow
#   재개 시 **assign=False** 로 로드한다. assign=True 는 Parameter 객체를 새로 만들어 이미 만들어진
#   옵티마이저가 옛 객체를 참조하게 만들고 → grad None → 전 파라미터 스킵 = 무학습 (업스트림 버그,
#   HANDOVER §3 첫 줄). 여기서는 처음부터 assign=False 만 쓴다.
# ─────────────────────────────────────────────────────────────────────────────
_CKPT_RE = re.compile(r"step_(\d+)\.pt$")


def find_latest_checkpoint(*dirs) -> Optional[str]:
    cands: List[Tuple[int, str]] = []
    for d in dirs:
        if not d:
            continue
        if os.path.isfile(d):
            return d
        for p in glob.glob(os.path.join(d, "**", "step_*.pt"), recursive=True):
            m = _CKPT_RE.search(os.path.basename(p))
            if m:
                cands.append((int(m.group(1)), p))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]


def strip_prefix(sd: dict) -> dict:
    """저장소 체크포인트(`_orig_mod.model.inner...`)도 읽을 수 있게 접두를 정리한다."""
    out = {}
    for k, v in sd.items():
        for pre in ("_orig_mod.", ):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def save_checkpoint(out_dir: str, step: int, base: nn.Module, optimizers, ema: Optional[EMAHelper],
                    iter_id: int, batch_in_iter: int, cfg: dict, keep_last: int):
    os.makedirs(out_dir, exist_ok=True)
    raw = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    if ema is not None:
        with _EMASwap(base, ema):
            ema_sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    else:
        ema_sd = raw
    state = {
        "step": step,
        "iter_id": iter_id,
        "batch_in_iter": batch_in_iter,
        "model_state_dict": ema_sd,                 # EMA (저장소 규약)
        "raw_model_state_dict": raw,                # 재개용 원시
        "ema_shadow": ({k: v.detach().cpu() for k, v in ema.state_dict().items()} if ema is not None else None),
        "optimizer_states": [o.state_dict() for o in optimizers],
        "rng_state": torch.random.get_rng_state(),
        "cfg": cfg,
    }
    if torch.cuda.is_available():
        try:
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            state["cuda_rng_state"] = torch.cuda.get_rng_state()

    tmp = os.path.join(out_dir, f".step_{step}.pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(out_dir, f"step_{step}.pt"))     # 중간에 끊겨도 반쪽 파일이 안 남는다

    # 용량 관리 — 최신 keep_last 개만 남긴다 (/kaggle/working 는 ~20GB)
    # [2026-09-04] `step_*.pt` 글롭은 `step_final.pt` 같은 이름도 잡는데 정규식은 숫자만 받는다.
    # 무방비로 .group(1) 을 부르면 AttributeError 로 **학습이 통째로 죽는다** (재현 확인).
    # find_latest_checkpoint 는 `if m:` 로 막고 있는데 여기만 빠져 있었다.
    files = []
    for p in glob.glob(os.path.join(out_dir, "step_*.pt")):
        m = _CKPT_RE.search(os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    for _, p in files[:-keep_last] if keep_last > 0 else []:
        try:
            os.remove(p)
        except OSError:
            pass
    return os.path.join(out_dir, f"step_{step}.pt")


def load_checkpoint(path: str, base: nn.Module, optimizers, device, load_optimizer: bool = True):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("raw_model_state_dict") or ck["model_state_dict"]
    base.load_state_dict(strip_prefix(sd), strict=True, assign=False)   # assign=False 필수 (위 주석)
    if load_optimizer and ck.get("optimizer_states") is not None:
        if len(ck["optimizer_states"]) == len(optimizers):
            for o, s in zip(optimizers, ck["optimizer_states"]):
                o.load_state_dict(s)
        else:
            print("[LT] 옵티마이저 개수 불일치 — 상태 로드 생략", flush=True)
    if ck.get("rng_state") is not None:
        torch.random.set_rng_state(torch.as_tensor(ck["rng_state"], device="cpu").to(torch.uint8))
    return ck


# ─────────────────────────────────────────────────────────────────────────────
# 9. 평가 — pretrain.py `evaluate` 그대로 (evaluators/save_preds 는 스도쿠 경로에서 미사용이라 제거)
#    원본 [EVAL] 프린트는 dist.reduce **전에** 찍혀 랭크 0 의 지역 수치만 보여준다(패치의 결함).
#    여기서는 reduce 후에 찍는다 → 다중 GPU 에서도 exact 가 /2048 로 나온다. 1 GPU 면 완전히 같다.
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _eval_pass(model, base, eval_in, eval_lb, cfg, rank, world_size, device):
    metric_keys: List[str] = []
    metric_values = None
    for batch in eval_batches(eval_in, eval_lb, cfg["global_batch_size"], rank, world_size):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.device(device):
            carry = base.initial_carry(batch)
        while True:
            carry, loss, metrics, _, all_finish = model(carry=carry, batch=batch, return_keys=set())
            if all_finish:
                break
        if metric_values is None:
            metric_keys = list(sorted(metrics.keys()))
            metric_values = torch.zeros(len(metric_keys), dtype=torch.float32, device=device)
        metric_values += torch.stack([metrics[k].to(torch.float32) for k in metric_keys])
    return metric_keys, metric_values


def evaluate(model, base, eval_in, eval_lb, cfg, rank, world_size, device, step: int,
             ema: Optional[EMAHelper]):
    with _EMASwap(base, ema):        # eval 은 EMA 가중치로 (yaml: ema=true)
        base.eval()
        try:
            keys, values = _eval_pass(model, base, eval_in, eval_lb, cfg, rank, world_size, device)
        finally:
            base.train()
    if values is None:
        return None
    if world_size > 1:
        dist.reduce(values, dst=0)
    if rank != 0:
        return None
    d = dict(zip(keys, values.cpu().numpy()))
    n = max(float(d.get("count", 1.0)), 1.0)
    print(f"[EVAL] step {step}  acc {d.get('accuracy', 0.0) / n:.4f}  "
          f"exact {int(d.get('exact_accuracy', 0))}/{int(n)}", flush=True)
    return {k: (v / n if k != "count" else v) for k, v in d.items()}


def extrapolate(base, eval_in, eval_lb, cfg, rank, world_size, device, step: int,
                ema: Optional[EMAHelper], segs: int, out_txt: str):
    """[2026-09-05] 세그먼트 외삽 — 가중치 고정(EMA), 추론에서 세그먼트만 segs 까지.
    2026-09-05/analysis/seg_extrap_kaggle.py 와 같은 프로토콜·같은 채점:
      acc = 81칸 중 맞은 비율의 퍼즐 평균, exact = 81칸 전부 맞은 퍼즐 수, churn = 직전 세그 대비 답 바뀐 칸 비율.
    모든 랭크가 호출해야 한다 (evaluate 와 같은 분담 + dist.reduce). 결과 txt 는 랭크 0 만 쓴다.
    컴파일된 model 이 아니라 eager `base` 로 돈다 — loops 를 바꿔도 재컴파일이 없다."""
    lt = base.model                               # ACTLossHead 안의 LT
    loops0 = lt.config.loops
    n_local = 0
    sums = torch.zeros(3, segs, dtype=torch.float64, device=device)     # [acc합, exact합, churn합] × seg
    cnt = torch.zeros(1, dtype=torch.float64, device=device)
    t0 = time.time()
    with _EMASwap(base, ema):
        base.eval()
        try:
            lt.config.loops = segs + 1            # halted 가 segs 안에서 켜지지 않도록
            with torch.no_grad():
                for batch in eval_batches(eval_in, eval_lb, cfg["global_batch_size"], rank, world_size):
                    if cfg.get("milestone_extrap_n") and n_local >= cfg["milestone_extrap_n"]:
                        break
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with torch.device(device):
                        carry = lt.initial_carry(batch)
                    y = batch["labels"]
                    mask = y != IGNORE_LABEL_ID
                    denom = mask.sum(-1).clamp_min(1)
                    prev = None
                    for si in range(segs):
                        carry, out = lt(carry, batch)
                        pred = out["logits"].argmax(-1)
                        ok = mask & (pred == y)
                        sums[0, si] += (ok.sum(-1).double() / denom).sum()
                        sums[1, si] += (ok.sum(-1) == mask.sum(-1)).sum()
                        if prev is not None:
                            sums[2, si] += (pred != prev).double().mean(-1).sum()
                        prev = pred
                    n_local += y.shape[0]
                    cnt += y.shape[0]
        finally:
            lt.config.loops = loops0
            base.train()
    if world_size > 1:
        dist.reduce(sums, dst=0); dist.reduce(cnt, dst=0)
    if rank != 0:
        return None
    N = int(cnt.item())
    acc = (sums[0] / max(N, 1)).cpu().numpy(); exact = sums[1].cpu().numpy(); churn = (sums[2] / max(N, 1)).cpu().numpy()
    base_a, base_e = float(acc[15]), int(exact[15])
    be = int(np.argmax(exact)); ba = int(np.argmax(acc))
    lines = [f"# seg_extrap  step={step}  weights=ema  segs={segs}  n={N}  elapsed={time.time()-t0:.0f}s",
             f"# cfg: d={cfg['hidden_size']} H={cfg['num_heads']} blocks={cfg['blocks_per_seg']} layers={cfg['num_layers']} "
             f"order={cfg['block_order']} gauge={'legacy' if cfg['legacy_gauge'] else 'sqrt_d'} trace={cfg.get('use_trace', False)} lam_fixed={cfg['stdp_lam_fixed']} gain_fixed={cfg['stdp_gain_fixed']}",
             f"{'seg':>5} {'acc':>8} {'exact':>6} {'exact%':>8} {'churn':>8}"]
    for si in range(segs):
        lines.append(f"{si+1:5d} {acc[si]:8.4f} {int(exact[si]):6d} {100*exact[si]/max(N,1):8.2f} {churn[si]:8.5f}" + ("  <-train" if si == 15 else ""))
    lines += ["",
              f"train seg16: acc {base_a:.4f}  exact {base_e}/{N} ({100*base_e/max(N,1):.2f}%)",
              f"best acc:    seg {ba+1}  {acc[ba]:.4f}  ({acc[ba]-base_a:+.4f})",
              f"best exact:  seg {be+1}  {int(exact[be])}/{N} ({100*exact[be]/max(N,1):.2f}%)  ({int(exact[be])-base_e:+d}, {100*(exact[be]-base_e)/max(N,1):+.2f}%p)"]
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    tmp = out_txt + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_txt)
    print(f"[EXTRAP] step {step}  seg16 {100*base_e/max(N,1):.2f}%  best seg{be+1} {100*exact[be]/max(N,1):.2f}%  "
          f"({100*(exact[be]-base_e)/max(N,1):+.2f}%p)  churn@{segs} {churn[-1]:.5f}  → {out_txt}", flush=True)
    return dict(acc=acc.tolist(), exact=exact.tolist(), churn=churn.tolist())


def _milestone(out_dir, ts, base, optimizers, ema, cur_iter, cfg, rank, world_size, device, te_in, te_lb):
    """[2026-09-05] milestone_every 마다: (1) 랭크 0 이 milestones/step_N.pt 영구 저장 (keep_last 정리 없음),
    (2) 모든 랭크가 외삽에 참여 → milestones/extrap_step_N.txt. 실패해도 학습은 계속 (예외를 삼키고 로그)."""
    mdir = os.path.join(out_dir, "milestones")
    if rank == 0:
        try:
            p = save_checkpoint(mdir, ts.step, base, optimizers, ema, cur_iter, ts.batch_in_iter, cfg, keep_last=0)
            print(f"[LT] milestone 저장 step={ts.step} → {p}", flush=True)
        except Exception as e:
            print(f"[LT] milestone 저장 실패: {type(e).__name__}: {e}", flush=True)
    if world_size > 1:
        dist.barrier()
    try:
        extrapolate(base, te_in, te_lb, cfg, rank, world_size, device, ts.step, ema,
                    int(cfg["milestone_extrap_segs"]), os.path.join(mdir, f"extrap_step_{ts.step}.txt"))
    except Exception as e:
        if rank == 0:
            print(f"[LT] milestone 외삽 실패: {type(e).__name__}: {e}", flush=True)
        raise            # 집합 통신 도중 실패면 랭크 간 불일치라 계속 갈 수 없다 — 죽기 전 저장은 바깥 except 가 한다


# ─────────────────────────────────────────────────────────────────────────────
# 10. 학습 루프 — pretrain.py `train_batch`(:522) · `launch`(:838) 의미론 그대로
#
# 손실 스케일과 all-reduce (다중 GPU 에서 URM 과 같은 값이 되도록):
#   URM 은 랭크마다 자기 몫(local_bs)에 대해 lm_loss 를 **합**으로 구하고 `loss/global_batch_size` 를
#   backward 한 뒤, 파라미터 그래디언트를 `dist.all_reduce`(기본 op=SUM, **평균이 아니다**) 한다.
#   → 최종 그래디언트 = ∇[ Σ_{전역 배치} ℓ / global_batch_size ]. 정확히 우리가 원하는 값이다.
#   accelerate 의 `prepare()`(DDP)는 그래디언트를 **평균**내므로 그대로 쓰면 1/world_size 로 어긋난다.
#   그래서 여기서는 **모델을 accelerate 로 감싸지 않고**(prepare 호출 없음) URM 의 수동 all_reduce 를
#   그대로 쓴다. accelerate 는 프로세스 기동·프로세스 그룹 초기화·디바이스 배치에만 쓴다.
#   부수 효과로 DDP 의 unused-parameter 문제도 사라진다 — stdp_window="psi" 면 `beta` 가 forward 에
#   등장하지 않아 DDP 는 find_unused_parameters 없이는 죽는다. 수동 경로는 grad=None 을 0 으로 대신
#   all_reduce 해 랭크 간 집합 호출 횟수만 맞추면 되므로(원본과 동일) 문제가 없다.
#
# carry 는 프로세스마다 자기 몫(local_bs)만 들고 있으면 된다 — 레인이 랭크를 넘나들지 않는다.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainState:
    step: int = 0
    iter_id: int = 0
    batch_in_iter: int = 0
    carry: Any = None
    last_fwdbwd_s: float = 0.0      # LT_TIME=1 일 때만 채워진다
    last_allreduce_s: float = 0.0


_TIME_SYNC = os.environ.get("LT_TIME", "") == "1"


def _tstr(ts) -> str:
    """LT_TIME=1 일 때만 스텝 시간 분해를 로그에 붙인다."""
    if not _TIME_SYNC:
        return ""
    return f"  [fwd+bwd {ts.last_fwdbwd_s*1e3:.0f}ms · allreduce {ts.last_allreduce_s*1e3:.0f}ms]"   # LT_TIME=1 → 스텝을 fwd+bwd / allreduce 로 분해


def train_batch(model, base, ts: TrainState, batch, cfg, optimizers, optimizer_lrs,
                total_steps: int, rank: int, world_size: int, device):
    accum_steps = max(1, cfg["grad_accum_steps"])
    gbs = cfg["global_batch_size"]
    if ts.step >= total_steps:
        return None

    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    if ts.carry is None:
        with torch.device(device):
            ts.carry = base.initial_carry(batch)

    # compute_target_q 는 LT.forward 가 받기만 하고 쓰지 않는다(ACT 없음). 넘기지 않는 이유:
    # 4스텝마다 값이 바뀌는 bool 인자는 dynamo 가 특수화해 같은 모델의 그래프를 하나 더 뜨게 한다.
    # 값에 영향이 없음이 코드상 자명하므로 생략한다.
    if _TIME_SYNC:
        torch.cuda.synchronize(); _tf = time.perf_counter()
    ts.carry, loss, metrics, _, _ = model(carry=ts.carry, batch=batch, return_keys=set())

    loss_scale = 1.0 / (gbs * accum_steps)
    (loss_scale * loss).backward()
    if _TIME_SYNC:
        torch.cuda.synchronize(); ts.last_fwdbwd_s = time.perf_counter() - _tf

    # Allreduce (URM: 평균이 아니라 합)
    # [2026-09-04] URM 원본은 파라미터마다 all_reduce 를 따로 부른다(우리 모델에서 25회). 집합통신
    #   하나하나가 전 랭크 동기화 장벽이라 스텝마다 GPU 파이프라인이 25번 비워지고, 배치가 작을수록
    #   (캐글 32/GPU) 커널 런치 오버랩이 깨져 통신량보다 그 손실이 커진다.
    #   → 평탄 버퍼 하나로 묶어 **1회**만 부른다. 합 리듀스라 수치는 동일하다.
    #   grad 가 None 이면 실제 `.grad` 에 0 을 붙인다(원본은 임시 텐서를 리듀스해 버려 의미가 없었다).
    ts.last_allreduce_s = 0.0
    if world_size > 1:
        grads = []
        for param in base.parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            grads.append(param.grad)
        if _TIME_SYNC:
            torch.cuda.synchronize(); _ta = time.perf_counter()
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        for g, src in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(src)
        if _TIME_SYNC:
            torch.cuda.synchronize(); ts.last_allreduce_s = time.perf_counter() - _ta

    lr_this_step = None
    for optim, base_lr in zip(optimizers, optimizer_lrs):
        lr_this_step = cosine_schedule_with_warmup_lr_lambda(
            current_step=ts.step, base_lr=base_lr,
            num_warmup_steps=round(cfg["lr_warmup_steps"]), num_training_steps=total_steps,
            min_ratio=cfg["lr_min_ratio"])
        for pg in optim.param_groups:
            pg["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    ts.step += 1

    # Reduce metrics
    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k].to(torch.float32) for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)
        if rank == 0:
            mv = metric_values.cpu().numpy()
            red = {k: mv[i] for i, k in enumerate(metric_keys)}
            count = max(red.get("count", 0), 1)
            out = {k: (v / gbs if k.endswith("loss") else v / count) for k, v in red.items()}
            out["lr"] = lr_this_step
            # count 는 halted 레인 수다. ACT 가 없으므로 halted 는 loops(=16) 스텝마다 한 번만 참이라
            # 대부분의 스텝에서 count=0 이고 accuracy/exact 는 0/1=0 이 된다(URM 도 동일 — HANDOVER §3
            # '재개 로그의 train acc 0.0000'). 원시 count 를 같이 돌려 호출측이 유효한 스텝만 로그하게 한다.
            out["_count_raw"] = float(red.get("count", 0.0))
            return out
    return None


def resolve_out_dir(cfg) -> str:
    if cfg["out_dir"]:
        return cfg["out_dir"]
    return "/kaggle/working/checkpoints" if os.path.isdir("/kaggle/working") else os.path.abspath("./lt_checkpoints")


def _find_npz(path: str) -> str:
    if os.path.exists(path):
        return path
    for alt in (os.path.join(REPO, "data/sudoku_lt_1k.npz"), os.path.join(REPO, "kaggle/upload/sudoku_lt_1k.npz")):
        if os.path.exists(alt):
            return alt
    raise FileNotFoundError(f"데이터 npz 를 못 찾았다: {path} — data/prep_dataset.py 로 만들어라")

def load_data(cfg):
    path = _find_npz(cfg["data_npz"])
    z = np.load(path)
    tr_in, tr_lb = z["train_inputs"], z["train_labels"]
    te_in, te_lb = z["test_inputs"][: cfg["test_size"]], z["test_labels"][: cfg["test_size"]]
    assert tr_in.ndim == 3 and tr_in.shape[1:] == (9, 9), tr_in.shape
    return tr_in, tr_lb, te_in, te_lb


def _init_pg_from_env():
    """런처가 심어 둔 환경변수(RANK/WORLD_SIZE/LOCAL_RANK)만으로 프로세스 그룹을 세운다."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, ws = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        torch.cuda.set_device(local)
        return rank, ws, torch.device("cuda", local)
    return rank, ws, torch.device("cpu")


def init_distributed(cfg):
    """torchrun 이 심은 RANK/WORLD_SIZE 가 있으면 프로세스 그룹을 세우고, 없으면 단일 프로세스."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return _init_pg_from_env()
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    cfg = dict(CFG)
    t_start = time.time()

    rank, world_size, device = init_distributed(cfg)
    is_main = (rank == 0)
    gbs = cfg["global_batch_size"]
    assert gbs % world_size == 0, f"global_batch_size {gbs} 가 프로세스 수 {world_size} 로 안 나뉜다"
    local_bs = gbs // world_size

    torch.random.manual_seed(cfg["seed"] + rank)      # pretrain.py:846
    np.random.seed(cfg["seed"] + rank)

    tr_in, tr_lb, te_in, te_lb = load_data(cfg)
    n_groups = len(tr_in)

    # ---- 총 스텝 (pretrain.py:270) — mean_puzzle_examples = 1
    total_steps = int(cfg["epochs"] * n_groups * 1.0 / (gbs * max(1, cfg["grad_accum_steps"])))
    assert cfg["epochs"] % cfg["eval_interval"] == 0, "eval_interval 은 epochs 의 약수여야 한다"
    total_iters = cfg["epochs"] // cfg["eval_interval"]

    # ---- 모델
    model_cfg = dict(cfg, batch_size=local_bs, seq_len=cfg["grid"] ** 2,
                     num_puzzle_identifiers=1)
    with torch.device(device):
        base = ACTLossHead(LT(model_cfg), "stablemax_cross_entropy", q_weight=cfg["q_weight"])
    base.train()
    if world_size > 1:      # 랭크마다 시드가 달라 초기값이 다르다 → 랭크 0 으로 통일 (pretrain.py:186)
        with torch.no_grad():
            for p in list(base.parameters()) + list(base.buffers()):
                dist.broadcast(p, src=0)

    optimizers, optimizer_lrs = create_optimizers(base, cfg, world_size)

    # ---- 재개
    out_dir = resolve_out_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    if is_main:
        # [2026-09-04] 경로 사고가 두 번 나서 시작 시점에 못 박아 둔다.
        try:
            _t = os.path.join(out_dir, ".write_test")
            with open(_t, "w") as _f:
                _f.write("ok")
            os.remove(_t)
            _w = "쓰기 가능"
        except Exception as _e:
            _w = f"★ 쓰기 불가: {type(_e).__name__}: {_e}"
        print(f"[LT] out_dir = {out_dir}  ({_w})", flush=True)
        print(f"[LT] 캐글 output 은 /kaggle/working 아래만 저장된다 — out_dir 이 그 밖이면 세션 종료 시 사라진다.",
              flush=True) if not out_dir.startswith("/kaggle/working") and os.path.isdir("/kaggle/working") else None
    ts = TrainState()
    ckpt = None
    resume_path = find_latest_checkpoint(out_dir, cfg["resume_from"])

    if resume_path:
        if is_main:
            print(f"[LT] 체크포인트 재개: {resume_path}", flush=True)
        ckpt = load_checkpoint(resume_path, base, optimizers, device)
        ts.step = int(ckpt.get("step", 0))
        ts.iter_id = int(ckpt.get("iter_id", 0))
        ts.batch_in_iter = int(ckpt.get("batch_in_iter", 0))

    elif is_main:
        print(f"[LT] ※ 체크포인트 없음 — step 0 부터 시작한다. 이어서 돌릴 의도였다면 "
              f"CFG['resume_from'] 에 이전 세션 output 경로를 지정하라 (예외사항.md §4).", flush=True)
        if cfg["require_resume"]:
            raise RuntimeError("require_resume=True 인데 재개할 체크포인트를 못 찾았다 — 12시간을 헛돌지 않도록 중단한다")

    ema = None
    if cfg["ema"]:
        ema = EMAHelper(mu=cfg["ema_rate"])
        ema.register(base)
        if ckpt is not None and ckpt.get("ema_shadow"):
            ema.load_state_dict({k: v.to(device) for k, v in ckpt["ema_shadow"].items()})
            if is_main:
                print("[LT] EMA shadow 복원", flush=True)
    del ckpt

    # [2026-09-04] 4xL4 가 1xL4 보다 느렸던 진짜 원인.
    #   local_bs=32 에서 step 이 614ms 인데 all_reduce 는 18ms 뿐이었다(통신 문제가 아니었다).
    #   프로파일 결과 Gabor 커널 a_tn 의 역전파로 생성된 persistent reduction 하나
    #   (triton_per_fused__to_copy_add_cos_div_mul_neg_sin_sum) 가 CUDA 시간의 84% 를 먹었다:
    #     bs48  3.32 ms/call  →  bs32  228.39 ms/call   (같은 커널, 69배)
    #   축약 크기가 그대로인데 병렬 블록만 줄어 점유율이 무너지는 전형적 케이스다. Inductor 가
    #   persistent 대신 looped reduction 을 쓰게 하면 사라진다 (로컬 L4 d832 N2 실측):
    #     bs32 compile 기본 544.66 ms → persistent_reductions=False 94.25 ms (5.8배)
    #   coordinate_descent_tuning=True 도 94.58 ms 로 같은 곳에 도달하나 컴파일이 훨씬 길다.
    if cfg["compile"] and cfg["inductor_no_persist"]:
        import torch._inductor.config as _ic
        _ic.triton.persistent_reductions = False
        if is_main:
            print("[LT] inductor: persistent_reductions=False (소배치 축약 병리 회피)", flush=True)
    model = torch.compile(base, dynamic=False) if cfg["compile"] else base

    if is_main:
        nparam = sum(p.numel() for p in base.parameters())
        print(f"[LT] world_size={world_size} local_bs={local_bs} params={nparam:,} "
              f"total_steps={total_steps} total_iters={total_iters} out={out_dir}", flush=True)
        print(f"[LT] 시작 step={ts.step} iter={ts.iter_id} batch_in_iter={ts.batch_in_iter}", flush=True)

    # ---- 데이터로더 (남은 구간만)
    ds = SudokuTrainDataset(tr_in, tr_lb, seed=cfg["seed"], num_aug=cfg["num_aug"],
                            global_batch_size=gbs, rank=rank, world_size=world_size,
                            epochs_per_iter=cfg["eval_interval"], start_iter=ts.iter_id,
                            total_iters=total_iters, skip_batches=ts.batch_in_iter)
    loader = DataLoader(ds, batch_size=None, num_workers=cfg["dataloader_workers"],
                        prefetch_factor=(8 if cfg["dataloader_workers"] > 0 else None),
                        pin_memory=True, persistent_workers=False)

    pbar = None
    if is_main:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=total_steps, initial=ts.step, dynamic_ncols=True)
        except Exception:
            pbar = None

    max_steps = cfg["max_steps"] if cfg["max_steps"] else total_steps
    stop = False
    last_full = None      # 마지막으로 halted 였던 스텝의 학습 지표 (위 주석 참조)

    def _do_eval_and_save(iter_done: int, batch_in_iter: int):
        evaluate(model, base, te_in, te_lb, cfg, rank, world_size, device, ts.step, ema)
        if is_main:
            save_checkpoint(out_dir, ts.step, base, optimizers, ema, iter_done, batch_in_iter,
                            cfg, cfg["keep_last"])

    def _should_stop() -> bool:
        """모든 랭크가 같은 결정을 내려야 한다 — 랭크 0 의 판정을 브로드캐스트."""
        flag = torch.tensor([1 if (time.time() - t_start) > cfg["max_hours"] * 3600 else 0],
                            device=device, dtype=torch.int32)
        if world_size > 1:
            dist.broadcast(flag, src=0)
        return bool(flag.item())

    cur_iter = ts.iter_id
    try:
        for it, batch in loader:
            if it != cur_iter:                      # iter 경계 = eval_interval 에폭 완주
                _do_eval_and_save(cur_iter, 0)
                cur_iter = it
                ts.batch_in_iter = 0
                if stop:
                    break

            metrics = train_batch(model, base, ts, batch, cfg, optimizers, optimizer_lrs,
                                  total_steps, rank, world_size, device)
            if ema is not None:
                ema.update(base)
            ts.batch_in_iter += 1
            ts.iter_id = cur_iter

            if is_main:
                if pbar is not None:
                    pbar.update(1)
                if metrics is not None and metrics.get("_count_raw", 0.0) > 0:
                    last_full = (ts.step, metrics)       # halted 스텝의 지표만 보관
                if metrics is not None and ts.step % cfg["log_every"] == 0:
                    lm = float(metrics.get("lm_loss", float("nan")))
                    if last_full is None:
                        print(f"[LT] step {ts.step}  lm_loss {lm:.4f}{_tstr(ts)}", flush=True)
                    else:
                        fs, fm = last_full
                        ex = float(fm.get("exact_accuracy", 0.0))
                        print(f"[LT] step {ts.step}  lm_loss {lm:.4f}{_tstr(ts)}  "
                              f"acc {float(fm.get('accuracy', 0)):.4f}  exact {ex:.4f} (~{ex * gbs:.0f}/{gbs}) "
                              f"[halt step {fs}]", flush=True)

            if cfg["save_every_steps"] and ts.step % cfg["save_every_steps"] == 0 and is_main:
                save_checkpoint(out_dir, ts.step, base, optimizers, ema, cur_iter, ts.batch_in_iter,
                                cfg, cfg["keep_last"])
            if cfg.get("milestone_every") and ts.step % cfg["milestone_every"] == 0:
                _milestone(out_dir, ts, base, optimizers, ema, cur_iter, cfg, rank, world_size, device, te_in, te_lb)

            if ts.step >= max_steps:
                stop = True
            elif ts.step % 50 == 0 and _should_stop():
                stop = True
                if is_main:
                    print(f"[LT] max_hours {cfg['max_hours']}h 도달 — 저장 후 종료", flush=True)
            if stop:
                break

    except BaseException as _e:
        # [2026-09-04] 8시간째 OOM·NaN 으로 죽으면 마지막 저장 이후가 전부 날아간다.
        #   죽기 전에 한 번 더 저장하고 예외는 그대로 올린다 (KeyboardInterrupt 포함이라 BaseException).
        if is_main:
            try:
                _p = save_checkpoint(out_dir, ts.step, base, optimizers, ema, cur_iter,
                                     ts.batch_in_iter, cfg, cfg["keep_last"])
                print(f"[LT] 예외({type(_e).__name__}) — step={ts.step} 저장 후 종료 → {_p}", flush=True)
            except Exception as _se:
                print(f"[LT] 예외 중 저장도 실패: {type(_se).__name__}: {_se}", flush=True)
        raise
    # ---- 마지막 eval + 저장
    evaluate(model, base, te_in, te_lb, cfg, rank, world_size, device, ts.step, ema)
    if is_main:
        p = save_checkpoint(out_dir, ts.step, base, optimizers, ema, cur_iter, ts.batch_in_iter,
                            cfg, cfg["keep_last"])
        print(f"[LT] 종료 step={ts.step} → {p}", flush=True)
        if pbar is not None:
            pbar.close()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# 11. 기동 — notebook_launcher
#     주의: 이 함수를 부르기 전에 노트북에서 torch.cuda 를 건드리면 fork 가 깨진다.
# ─────────────────────────────────────────────────────────────────────────────
def run():
    """단일 프로세스 또는 torchrun 자식으로 main() 을 돈다."""
    return main()


# ─────────────────────────────────────────────────────────────────────────────
# 12. self-test — GPU 없이 도는 검증 (증강 규칙이 원본과 같은지)
# ─────────────────────────────────────────────────────────────────────────────
def selftest():
    """`_apply_aug` 가 build_sudoku_dataset.shuffle_sudoku 의 apply_transformation 과
    **같은 파라미터에서 같은 출력**을 내는지 확인한다. 원본 함수를 여기에 다시 적어 대조한다."""
    def ref_shuffle(board, solution):
        digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
        transpose_flag = np.random.rand() < 0.5
        bands = np.random.permutation(3)
        row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
        stacks = np.random.permutation(3)
        col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
        mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

        def apply_transformation(x):
            if transpose_flag:
                x = x.T
            new_board = x.flatten()[mapping].reshape(9, 9).copy()
            return digit_map[new_board]

        return (apply_transformation(board), apply_transformation(solution),
                (digit_map, transpose_flag, row_perm, col_perm))

    rng = np.random.default_rng(0)
    base_sol = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)], dtype=np.uint8)
    ok = 0
    for s in range(200):
        np.random.seed(s)
        mask = rng.random((9, 9)) < 0.5
        board = np.where(mask, base_sol, 0).astype(np.uint8)
        rb, rs, prm = ref_shuffle(board, base_sol)
        assert np.array_equal(_apply_aug(board, *prm), rb), f"입력판 불일치 seed={s}"
        assert np.array_equal(_apply_aug(base_sol, *prm), rs), f"정답판 불일치 seed={s}"
        # 변환 후에도 유효한 스도쿠여야 한다
        assert np.all(np.sort(rs, axis=1) == np.arange(1, 10))
        assert np.all(np.sort(rs, axis=0).T == np.arange(1, 10))
        for br in range(3):
            for bc in range(3):
                assert np.array_equal(np.sort(rs[br * 3:br * 3 + 3, bc * 3:bc * 3 + 3].ravel()), np.arange(1, 10))
        assert np.all((rb == 0) | (rb == rs)), "단서가 정답과 불일치"
        ok += 1
    print(f"[selftest] 증강 규칙 원본과 일치: {ok}/200 케이스")

    # 고정 풀: 같은 (퍼즐, aug_idx) 는 항상 같은 판
    ds = SudokuTrainDataset(base_sol[None], base_sol[None], seed=0, num_aug=1000,
                            global_batch_size=1, rank=0, world_size=1, epochs_per_iter=1,
                            start_iter=0, total_iters=1)
    a1 = ds._augmented(0, 7)[0]
    a2 = ds._augmented(0, 7)[0]
    a3 = ds._augmented(0, 8)[0]
    assert np.array_equal(a1, a2) and not np.array_equal(a1, a3)
    assert np.array_equal(ds._augmented(0, 0)[0], base_sol), "aug_idx=0 은 무증강이어야 한다"
    print("[selftest] 고정 증강 풀 OK")


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="LT 스도쿠 학습 (URM 하네스). 판은 --preset 으로 고른다.")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="v1.1")
    ap.add_argument("--out_dir"); ap.add_argument("--resume_from"); ap.add_argument("--data")
    ap.add_argument("--milestone_every", type=int); ap.add_argument("--max_steps", type=int)
    ap.add_argument("--epochs", type=int); ap.add_argument("--seed", type=int)
    ap.add_argument("--no_compile", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    CFG.update(PRESETS[a.preset])
    if a.out_dir: CFG["out_dir"] = a.out_dir
    if CFG["out_dir"] is None: CFG["out_dir"] = os.path.join(REPO, "runs", a.preset.replace(".", "_"))
    if a.resume_from: CFG["resume_from"] = a.resume_from
    if a.data: CFG["data_npz"] = a.data
    if a.milestone_every is not None: CFG["milestone_every"] = a.milestone_every
    if a.max_steps is not None: CFG["max_steps"] = a.max_steps
    if a.epochs is not None: CFG["epochs"] = a.epochs
    if a.seed is not None: CFG["seed"] = a.seed
    if a.no_compile: CFG["compile"] = False
    print(f"[LT] preset={a.preset}  {PRESETS[a.preset]}  out_dir={CFG['out_dir']}", flush=True)
    return run()


if __name__ == "__main__":
    _cli()
