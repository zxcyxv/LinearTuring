"""분석 스크립트 공용: 경로·데이터·수동 전개.

환경변수  LT_ROOT (저장소 루트, 기본 = 이 파일의 상위)   URM_DIR (기본 $LT_ROOT/refs/URM)
          SUDOKU_TEST (기본 $URM_DIR/data/sudoku-extreme-1k-aug-1000/test)
"""
import os, sys
import numpy as np
import torch

ROOT = os.environ.get("LT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URM = os.environ.get("URM_DIR", os.path.join(ROOT, "refs", "URM"))
TEST = os.environ.get("SUDOKU_TEST", os.path.join(URM, "data", "sudoku-extreme-1k-aug-1000", "test"))
for p in (URM, os.path.join(ROOT, "sudoku"), os.path.join(ROOT, "core")):
    sys.path.insert(0, p)
from lt import load_lt   # noqa: E402,F401

CKPT_DEFAULT = os.path.join(ROOT, "checkpoints", "R1B8_bilin_r2_step123039.pt")


def load_test(n=None, device="cuda"):
    inp = torch.from_numpy(np.load(f"{TEST}/all__inputs.npy")).to(torch.int32)
    lab = torch.from_numpy(np.load(f"{TEST}/all__labels.npy")).to(torch.int32)
    depth = np.load(f"{TEST}/cell_depth.npy") if os.path.exists(f"{TEST}/cell_depth.npy") else None
    if n is not None:
        inp, lab = inp[:n], lab[:n]
        depth = depth[:n] if depth is not None else None
    return inp.to(device), lab.to(device), depth


def make_batch(inp, lab, device="cuda"):
    return dict(inputs=inp, labels=lab,
                puzzle_identifiers=torch.zeros(len(inp), dtype=torch.int32, device=device))


def peer_mask():
    """[81,81] 행·열·박스 피어 (자기 제외)."""
    r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
    pm = (r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])
    np.fill_diagonal(pm, False)
    return pm


def rollout(m, batch, act=None, hook=None):
    """LT_Inner 를 블록 단위로 수동 전개. act: 게이트 함수 치환 (None = 모델 설정).
    hook(loop, blk, stage, h, a) 는 stage ∈ {'pre','post_bnd','post_inj','post_step'} 마다 호출.
    반환: 최종 h."""
    inner = m.inner; core = inner.core; cfg = m.config
    fc = core.kernel_fast(); AB = core.W_C(); dt = 1.0 / core.R
    B = batch["inputs"].shape[0]
    h = inner.init_hidden.expand(B, cfg.seq_len, -1).clone()
    inj = inner._injection(batch)
    for loop in range(cfg.loops):
        for blk in range(cfg.blocks_per_seg):
            if hook: hook(loop, blk, "pre", h, None)
            g, u = inner.b_gate_up(h).chunk(2, -1)
            gv = inner._act(g) if act is None else act(g)
            h = inner.b_carry * h + inner.b_down(gv * u)
            if hook: hook(loop, blk, "post_bnd", h, None)
            if blk == 0 or cfg.block_inj:
                h = h + inner.inj_gate * inj
            if hook: hook(loop, blk, "post_inj", h, None)
            a = None
            for _ in range(core.R):
                hh = core.phi(h, dt / 2)
                f, a, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc)
                h = core.phi(hh + dt * f, dt / 2)
            if hook: hook(loop, blk, "post_step", h, a)
    return h


def logits(m, h):
    return m.inner.core.w_cls(h)
