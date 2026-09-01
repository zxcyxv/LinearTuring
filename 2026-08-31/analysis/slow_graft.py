"""stdp1 에 '느린 기억'을 소수 성분으로 얹는다 — 123k 그래프트와 구조적으로 같은 형태.

  123k:  전달 = (1−λ)·a_보정됨      + λ·w_새것        λ 작게 → +52
  여기:  전달 = (1−μ)·w_fast(보정됨) + μ·w_slow(새것)   μ 작게 → ?

  w_fast ← w_fast + η_학습(Γ − w_fast)      (학습된 그대로, 전달의 주 성분)
  w_slow ← w_slow + η_slow(Γ − w_slow)      (새로 얹는 장기 성분)
"""
import argparse, json, os, sys, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "analysis"))
from common import ROOT, URM, load_test, make_batch                       # noqa: E402
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2048); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--segs", type=int, default=16)
ap.add_argument("--mu",   type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.4])
ap.add_argument("--slow", type=float, nargs="+", default=[0.1, 0.05, 0.02])
ap.add_argument("--out", default=os.path.join(ROOT, "2026-08-31", "results", "json", "slow_graft.json"))
A = ap.parse_args(); torch.set_grad_enabled(False)

inp, lab, _ = load_test(A.n); N = len(inp); GT = (lab - 2).long(); BL = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu",
                weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=A.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832,
                hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda()
m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; ETA = torch.sigmoid(inner.eta_raw).float(); K = 8

def run(mu, slow):
    es = torch.tensor(float(slow), device="cuda"); ex = 0; cell = 0.0; nb = 0
    for b in range(0, N, A.bs):
        x = inp[b:b + A.bs]; n_ = len(x); gt = GT[b:b + n_]; blb = BL[b:b + n_]
        h = inner.init_hidden.expand(n_, 81, -1).clone(); wf = ws = None
        for s in range(A.segs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
                for _ in range(K):
                    h = inner.boundary(h); h = h + inner.inj_gate * inj
                    a = inner.attn(h, AB, kc)
                    v = torch.einsum('btd,hcd->bthc', h, inner.w_sh)
                    vn = v / (v.norm(dim=-1, keepdim=True) + eps)
                    Gm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn)
                    wf = Gm if wf is None else wf + ETA * (Gm - wf)
                    ws = Gm if ws is None else ws + es  * (Gm - ws)
                    aeff = wf if mu == 0.0 else (1 - mu) * wf + mu * ws
                    o = torch.einsum('bhtn,bnhc->bthc', aeff.to(v.dtype), v)
                    h = inner.phi(h + torch.einsum('bthc,hcd->btd', o, inner.w_sh))
                lg = inner.w_cls(h).float()[:, :, 2:11]
            h = h.float()
        P = lg.argmax(-1)
        ex += int(((P == gt) | ~blb).all(1).sum()); cell += float((P == gt)[blb].float().sum()); nb += int(blb.sum())
    return ex, cell / nb

print(f"n={N}  segs={A.segs}  학습된 η 평균 {float(ETA.mean()):.3f}\n", flush=True)
base, cb = run(0.0, 0.1); print(f"기준 μ=0 (w_fast 단독)        완답 {base:5d}  셀 {cb:.4f}\n", flush=True)
res = {"base": base}
for slow in A.slow:
    for mu in A.mu:
        if mu == 0.0: continue
        ex, c = run(mu, slow)
        print(f"μ={mu:<5g} η_slow={slow:<6g}       완답 {ex:5d} ({ex-base:+5d})  셀 {c:.4f}", flush=True)
        res[f"mu={mu} slow={slow}"] = dict(exact=ex, cell=round(c, 4))
os.makedirs(os.path.dirname(A.out), exist_ok=True); json.dump(dict(res, n=N, segs=A.segs), open(A.out, "w"), indent=1)
print("\nsaved", A.out)
