"""양보 순서가 carry(h)의 크기에 있는가 — 칸 범주별 ‖h‖·‖Wh‖·상대 갱신량의 세그먼트별 추이.

결합이 대칭이어도 f_t = Σ_n a_tn (WᵀW) h_n 이므로
  메시지 크기 ∝ 보내는 쪽 ‖h_n‖,  저항 ∝ 받는 쪽 ‖h_t‖  →  상대 영향비 = (‖h_n‖/‖h_t‖)²
a_tn 은 칸 크기에 불변이므로(검증됨) 크기는 커널이 아니라 응답만 바꾼다.
범주: 단서칸 / 빈칸-최종정답 / 빈칸-최종오답
"""
import argparse, json, os, sys, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "analysis"))
from common import ROOT, URM, load_test, make_batch                    # noqa: E402
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=128); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--segs", type=int, default=16)
ap.add_argument("--out", default=os.path.join(ROOT, "2026-08-31", "results", "json", "carry_order.json"))
a_ = ap.parse_args(); torch.set_grad_enabled(False)

inp, lab, _ = load_test(a_.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu",
                weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=a_.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832,
                hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda()
m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; eta = torch.sigmoid(inner.eta_raw).float(); K = 8; S = a_.segs
gam = float(torch.nn.functional.softplus(inner.gamma_raw)); print(f"γ={gam:.4f}  흡수구 반지름 1/√γ = {1/gam**0.5:.3f}\n")

CATS = ["단서", "빈칸-정답", "빈칸-오답"]
rec = {c: {k: np.zeros(S) for k in ("hn", "vn", "relf")} for c in CATS}; cnt = np.zeros(S)
exact = 0
for b in range(0, N, a_.bs):
    x = inp[b:b + a_.bs]; n_ = len(x); gt = G[b:b + n_]; blb = bl[b:b + n_]
    h = inner.init_hidden.expand(n_, 81, -1).clone(); w = None
    snap = []
    for s in range(S):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for k in range(K):
                h = inner.boundary(h); h = h + inner.inj_gate * inj
                a = inner.attn(h, AB, kc)
                v = torch.einsum('btd,hcd->bthc', h, inner.w_sh)
                vn_ = v / (v.norm(dim=-1, keepdim=True) + eps)
                agree = torch.einsum('bthc,bnhc->bhtn', vn_, vn_)
                Gm = a * agree; w = Gm if w is None else w + eta * (Gm - w)
                o = torch.einsum('bhtn,bnhc->bthc', w.to(v.dtype), v)
                f = torch.einsum('bthc,hcd->btd', o, inner.w_sh)
                hprev = h; h = inner.phi(h + f)
            lg = inner.w_cls(h).float()[:, :, 2:11]
        h = h.float()
        snap.append((h.norm(dim=-1).clone(),                                  # ‖h_t‖
                     v.float().norm(dim=-1).mean(-1).clone(),                 # ‖W h_t‖ 헤드평균
                     (f.float().norm(dim=-1) / (hprev.float().norm(dim=-1) + 1e-9)).clone()))  # ‖f‖/‖h‖
    P = lg.argmax(-1); ok_cell = (P == gt)
    exact += int((ok_cell | ~blb).all(1).sum())
    masks = {"단서": ~blb, "빈칸-정답": blb & ok_cell, "빈칸-오답": blb & ~ok_cell}
    for s, (hn, vn2, rf) in enumerate(snap):
        cnt[s] += 1
        for c, msk in masks.items():
            if msk.sum() == 0: continue
            rec[c]["hn"][s] += float(hn[msk].mean()); rec[c]["vn"][s] += float(vn2[msk].mean()); rec[c]["relf"][s] += float(rf[msk].mean())

print(f"완답 {exact}/{N}\n")
marks = [0, 1, 3, 7, 11, 15]; marks = [x for x in marks if x < S]
for key, title in (("hn", "‖h_t‖  (carry 크기 = 저항)"), ("vn", "‖W h_t‖  (보내는 메시지 크기)"), ("relf", "‖f‖/‖h‖  (상대 갱신량 = 얼마나 움직였나)")):
    print(f"=== {title} ===")
    print(f"{'범주':<12}" + "".join(f"{'seg'+str(s+1):>10}" for s in marks))
    for c in CATS:
        print(f"{c:<12}" + "".join(f"{rec[c][key][s]/cnt[s]:>10.4f}" for s in marks))
    print()
r = {c: {k: (rec[c][k] / cnt).tolist() for k in rec[c]} for c in CATS}
h1 = rec["단서"]["hn"][-1] / cnt[-1]; h2 = rec["빈칸-오답"]["hn"][-1] / cnt[-1]
print(f"최종 세그먼트 크기비  단서/빈칸-오답 = {h1/h2:.3f}   →  상대 영향비 (제곱) = {(h1/h2)**2:.3f}")
os.makedirs(os.path.dirname(a_.out), exist_ok=True); json.dump(dict(r, gamma=gam, exact=exact, n=N), open(a_.out, "w"), indent=1)
print("saved", a_.out)
