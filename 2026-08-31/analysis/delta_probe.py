"""사전 점검 — 방향만 누적해도 되는가 (SPEC.md §1 판별자).  추론만, G=1 고정, Γ 형태 불변.

  w ← (1−δ)w + δ·Γ      δ = 학습된 η (기준) 또는 상수
  (iii) 조건: 같은 δ 로 step `restart` 에서 w 를 리셋 = 초기 무지 구간을 배제

판정:
  (ii) > (i)         → 방향만으로 누적이 된다. 2단계(권위 가중) 불필요
  (iii) > (ii)       → 초기 구간이 오염원. 2단계 지시
  (iii) ≈ (ii) < (i) → 긴 기억 자체가 해롭다(지연). 2단계로도 안 고쳐짐
"""
import argparse, json, os, sys, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "analysis"))
from common import ROOT, URM, load_test, make_batch                     # noqa: E402
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=256); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--segs", type=int, default=16)
ap.add_argument("--out", default=os.path.join(ROOT, "2026-08-31", "results", "json", "delta_probe.json"))
A = ap.parse_args(); torch.set_grad_enabled(False)

inp, lab, _ = load_test(A.n); N = len(inp); GT = (lab - 2).long(); BL = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu",
                weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=A.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832,
                hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda()
m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; ETA = torch.sigmoid(inner.eta_raw).float(); K = 8

def run(delta, restart=None):
    """delta: None = 학습된 η [H,1,1] / float = 상수.  restart: 그 스텝에서 w 리셋."""
    d = ETA if delta is None else torch.tensor(float(delta), device="cuda")
    ex = np.zeros(A.segs); ever = np.zeros(A.segs)
    for b in range(0, N, A.bs):
        x = inp[b:b + A.bs]; n_ = len(x); gt = GT[b:b + n_]; blb = BL[b:b + n_]
        h = inner.init_hidden.expand(n_, 81, -1).clone(); w = None
        solved = torch.zeros(n_, A.segs, dtype=torch.bool, device="cuda")
        for s in range(A.segs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
                for k in range(K):
                    if restart is not None and s * K + k == restart: w = None
                    h = inner.boundary(h); h = h + inner.inj_gate * inj
                    a = inner.attn(h, AB, kc)
                    v = torch.einsum('btd,hcd->bthc', h, inner.w_sh)
                    vn = v / (v.norm(dim=-1, keepdim=True) + eps)
                    Gm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn)
                    w = Gm if w is None else (1 - d) * w + d * Gm            # G=1
                    o = torch.einsum('bhtn,bnhc->bthc', w.to(v.dtype), v)
                    h = inner.phi(h + torch.einsum('bthc,hcd->btd', o, inner.w_sh))
                lg = inner.w_cls(h).float()[:, :, 2:11]
            h = h.float(); ok = ((lg.argmax(-1) == gt) | ~blb).all(1)
            solved[:, s] = ok; ex[s] += int(ok.sum())
        ever += solved.cumsum(1).bool().sum(0).cpu().numpy()
    return ex.astype(int), ever.astype(int)

marks = [x for x in (1, 2, 4, 8, 12, 16) if x <= A.segs]
CONDS = [("(i)  δ=학습된 η", None, None), ("     δ=1.0 (기억없음)", 1.0, None), ("     δ=0.5", 0.5, None),
         ("(ii) δ=0.10", 0.10, None), ("(ii) δ=0.05", 0.05, None), ("(ii) δ=0.02", 0.02, None),
         ("(iii)δ=0.10 @64리셋", 0.10, 64), ("(iii)δ=0.05 @64리셋", 0.05, 64), ("(iii)δ=0.02 @64리셋", 0.02, 64)]
print(f"n={N}  segs={A.segs}  G=1 고정  (학습된 η 평균 {float(ETA.mean()):.3f})\n")
print(f"{'조건':<22}" + "".join(f"{'seg'+str(x):>8}" for x in marks) + f"{'한번이라도':>10}")
res = {}
for name, d, rs in CONDS:
    ex, ev = run(d, rs)
    print(f"{name:<22}" + "".join(f"{ex[x-1]:>8}" for x in marks) + f"{ev[-1]:>10}")
    res[name.strip()] = dict(exact=ex.tolist(), ever=ev.tolist(), delta=d, restart=rs)
os.makedirs(os.path.dirname(A.out), exist_ok=True); json.dump(res, open(A.out, "w"), indent=1)
print("\nsaved", A.out)
