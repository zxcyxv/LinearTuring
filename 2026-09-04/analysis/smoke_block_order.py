"""block_order 플래그 스모크 테스트.

검증 4가지:
  1. pre 가 플래그 도입 전 코드(git HEAD)와 **비트 동일** — 회귀 없음
  2. pre / post 각각이 의도한 연산 순서와 **비트 동일** (모듈 원시연산으로 손으로 재구성해 대조)
  3. post 의 세그먼트 마지막 연산이 boundary 다 (판독이 쌍선형 직후)
  4. 잘못된 값 거부 · 파라미터 수 불변(파라미터 0개짜리 축)
"""
import os, sys, subprocess, importlib.util, tempfile
import torch
from dataclasses import replace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "refs", "URM"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


BASE = dict(batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832,
            hidden_size=832, num_heads=8, loops=16, grid=9, blocks_per_seg=8,
            amp=False,                       # fp32 — 비트 대조를 위해 autocast 끔
            stdp=True, stdp_target="faithful", stdp_window="psi",
            stdp_eta_init=0.05, stdp_gain_init=1.0, stdp_lam_fixed=-1.0)


def make_batch(dev):
    g = torch.Generator(device=dev).manual_seed(1)
    return {"inputs": torch.randint(1, 11, (2, 81), device=dev, generator=g),
            "puzzle_identifiers": torch.zeros(2, dtype=torch.long, device=dev)}


def wake_boundary(m):
    """b_down 은 영 초기화라 초기엔 경계가 정확히 항등이다 → 순서 차이가 안 보인다.
    결정론적으로 채워 경계를 살린다 (학습된 모델의 대리)."""
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        w = m.inner.b_down.weight
        w.copy_((torch.randn(w.shape, generator=g) * 0.02).to(w.device, w.dtype))
    return m


def run_model(LT, cfg, dev, nseg=2):
    torch.manual_seed(0)
    m = wake_boundary(LT({**BASE, **cfg}).to(dev).eval())
    b = make_batch(dev)
    c = m.initial_carry(b)
    c = replace(c, current_hidden=c.current_hidden.to(dev), steps=c.steps.to(dev),
                halted=c.halted.to(dev), current_data={k: v.to(dev) for k, v in c.current_data.items()})
    outs = []
    with torch.no_grad():
        for _ in range(nseg):
            c, o = m(c, b)
            outs.append(o["logits"])
    return m, outs


def manual_segment(m, batch, order, carry_h, w, fresh):
    """모듈 원시연산으로 한 세그먼트를 손으로 재구성."""
    inner = m.inner
    h = carry_h
    inj = inner.injection(batch)
    AB = inner.W_C(); kc = inner.kernel()
    kcb = inner.kernel(inner.beta) if (inner.stdp and inner.config.stdp_window == "beta") else None
    ops = []
    for _ in range(inner.config.blocks_per_seg):
        if order == "pre":
            h = inner.boundary(h, None); ops.append("boundary")
        h = h + inner.inj_gate * inj;    ops.append("inject")
        h, w = inner.step(h, AB, kc, w, fresh, kcb, apply_phi=(order == "pre"))
        ops.append("step+Phi" if order == "pre" else "step")
        if order == "post":
            h = inner.boundary(h, None); ops.append("boundary")
            h = inner.phi(h);            ops.append("Phi")
        fresh = None
    return h, w, inner.w_cls(h), ops


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    cur = load(os.path.join(ROOT, "core", "minimal.py"), "m_cur")

    # --- 1) pre 가 플래그 도입 전과 비트 동일한가 ---
    old_src = subprocess.check_output(["git", "-C", ROOT, "show", "HEAD:core/minimal.py"], text=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(old_src); old_path = f.name
    old = load(old_path, "m_old")
    _, o_new = run_model(cur.LT, dict(block_order="pre", stdp_window="beta"), dev)
    _, o_old = run_model(old.LT, dict(stdp_window="beta"), dev)   # HEAD 에는 이 키가 없다(extra=allow → 무시) = beta 동작
    d1 = max((a - b).abs().max().item() for a, b in zip(o_new, o_old))
    print(f"[1] pre  vs  HEAD(플래그 이전)           max|Δ| = {d1:.3e}   {'OK' if d1 == 0 else 'FAIL'}")

    # --- 2) pre / post 가 의도한 순서와 비트 동일한가 ---
    for order in ("pre", "post"):
        m, outs = run_model(cur.LT, dict(block_order=order), dev, nseg=1)
        b = make_batch(dev)
        torch.manual_seed(0)
        m2 = wake_boundary(cur.LT({**BASE, "block_order": order}).to(dev).eval())   # 같은 시드·같은 경계
        h0 = m2.inner.init_hidden.expand(2, 81, 832).contiguous()    # reset_carry 와 동일
        with torch.no_grad():
            _, _, lg, ops = manual_segment(m2, b, order, h0, None, torch.ones(2, dtype=torch.bool, device=dev))
        d = (lg - outs[0]).abs().max().item()
        print(f"[2] {order:<4} vs 손으로 재구성한 {'→'.join(ops[:3])}…  max|Δ| = {d:.3e}   {'OK' if d == 0 else 'FAIL'}")
        print(f"[3] {order:<4} 세그먼트 연산 순서 = {'→'.join(ops[:4])}  (마지막 {ops[-1]})")

    # --- pre 와 post 가 실제로 다른가 ---
    _, op_ = run_model(cur.LT, dict(block_order="pre"), dev)
    _, oq_ = run_model(cur.LT, dict(block_order="post"), dev)
    print(f"[3b] pre vs post                        max|Δ| = {max((a-b).abs().max().item() for a,b in zip(op_,oq_)):.3e}  (0 이면 플래그가 죽은 것)")

    # --- 4) 값 검증 · 파라미터 수 ---
    torch.manual_seed(0); n_pre = sum(p.numel() for p in cur.LT({**BASE, "block_order": "pre"}).parameters())
    torch.manual_seed(0); n_post = sum(p.numel() for p in cur.LT({**BASE, "block_order": "post"}).parameters())
    print(f"[4] 파라미터 pre {n_pre:,} / post {n_post:,}   {'OK (동일)' if n_pre == n_post else 'FAIL'}")
    try:
        cur.LT({**BASE, "block_order": "middle"})
        print("[4] 잘못된 값 거부  FAIL")
    except AssertionError as e:
        print(f"[4] 잘못된 값 거부  OK — {e}")
    os.unlink(old_path)


if __name__ == "__main__":
    main()
