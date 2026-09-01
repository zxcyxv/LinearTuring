"""stdp1 에 추론 시점에서 λ·η 를 풀어본다.  Γ 형태·가중치 불변, 전달 혼합비와 기억 속도만 바꾼다.
   결합 = (1−λ)·a + λ·w ,   w ← w + η(Γ−w) ,  Γ = a·⟨v̂,v̂⟩ (학습된 형태 그대로)
   학습은 λ=1 고정 + 헤드별 η 로 됐다. 기록(stdp_infer, 구판 123k)은 λ≈0.25·η≈0.05 가 최적이라 한다.
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
ap.add_argument("--lam", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
ap.add_argument("--eta", type=str, nargs="+", default=["learned", "0.1", "0.05"])
ap.add_argument("--out", default=os.path.join(ROOT, "2026-08-31", "results", "json", "lam_eta_probe.json"))
A = ap.parse_args(); torch.set_grad_enabled(False)

inp, lab, _ = load_test(A.n); N = len(inp); GT = (lab - 2).long(); BL = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu",
                weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=A.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832,
                hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda()
m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; ETA = torch.sigmoid(inner.eta_raw).float(); K = 8
print(f"n={N}  segs={A.segs}  학습된 η 평균 {float(ETA.mean()):.3f}  (학습은 λ=1 고정)\n", flush=True)

def run(lam, eta):
    e = ETA if eta is None else torch.tensor(float(eta), device="cuda")
    ex = 0; cell = 0.0; nb = 0
    for b in range(0, N, A.bs):
        x = inp[b:b + A.bs]; n_ = len(x); gt = GT[b:b + n_]; blb = BL[b:b + n_]
        h = inner.init_hidden.expand(n_, 81, -1).clone(); w = None
        for s in range(A.segs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
                for _ in range(K):
                    h = inner.boundary(h); h = h + inner.inj_gate * inj
                    a = inner.attn(h, AB, kc)
                    v = torch.einsum('btd,hcd->bthc', h, inner.w_sh)
                    vn = v / (v.norm(dim=-1, keepdim=True) + eps)
                    Gm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn)
                    w = Gm if w is None else w + e * (Gm - w)
                    aeff = (1 - lam) * a + lam * w
                    o = torch.einsum('bhtn,bnhc->bthc', aeff.to(v.dtype), v)
                    h = inner.phi(h + torch.einsum('bthc,hcd->btd', o, inner.w_sh))
                lg = inner.w_cls(h).float()[:, :, 2:11]
            h = h.float()
        P = lg.argmax(-1)
        ex += int(((P == gt) | ~blb).all(1).sum()); cell += float((P == gt)[blb].float().sum()); nb += int(blb.sum())
    return ex, cell / nb

res = {}
base = None
for lam in A.lam:
    for et in A.eta:
        e = None if et == "learned" else float(et)
        ex, c = run(lam, e)
        tag = f"λ={lam:<5g} η={et:<8s}"
        if base is None: base = ex
        print(f"{tag}  완답 {ex:5d} ({ex-base:+5d})  셀 {c:.4f}", flush=True)
        res[tag.strip()] = dict(exact=ex, cell=round(c, 4), lam=lam, eta=et)
os.makedirs(os.path.dirname(A.out), exist_ok=True); json.dump(dict(res, n=N, segs=A.segs), open(A.out, "w"), indent=1)
print("\nsaved", A.out)
