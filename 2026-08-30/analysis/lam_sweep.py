"""기저+가소성 분리(λ) 스윕 — Backpropamine 의 `w_ij + α_ij·Hebb_ij` 구조를 추론 시점에 되살려 본다.
  전달 결합 = (1−λ)·a + λ·w,  w ← w + η(Γ−w),  Γ = a·⟨v̂,v̂⟩
  stdp1 은 λ=1 (전달을 w 가 전담, 기저 a 제거) 로 학습됨. λ<1 은 순간 성분 a 를 되살려 기억의 지연을 상쇄한다.
사용: python 2026-08-30/analysis/lam_sweep.py [--lam 1.0 0.75 0.5 0.25] [--reset 0 1] [--segs 32]"""
import argparse, json, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--segs", type=int, default=32); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lam", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25]); ap.add_argument("--reset", type=int, nargs="+", default=[0, 1])
ap.add_argument("--scale", type=int, default=0, help="1이면 a 를 w 의 RMS 에 맞춰 정규화한 뒤 섞는다 (논문의 학습되는 α 의 추론 시점 대용)")
ap.add_argument("--marks", type=int, nargs="+", default=[8, 16, 32, 64, 128]); ap.add_argument("--out", default=os.path.join(ROOT, "2026-08-30", "results", "json", "lam_sweep.json"))
args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=args.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
               stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; eta = torch.sigmoid(inner.eta_raw).float(); K = 8; S = args.segs; marks = [k for k in args.marks if k <= S]
def run(lam, reset):
    ex = np.zeros(S); conf = np.zeros(S); solved = torch.zeros(N, S, dtype=torch.bool, device="cuda")
    for b in range(0, N, args.bs):
        x = inp[b:b + args.bs]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); w = None
        for s in range(S):
            if reset: w = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
                for _ in range(K):
                    h = inner.boundary(h); h = h + inner.inj_gate * inj
                    a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
                    Gm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn); w = Gm if w is None else w + eta * (Gm - w)
                    if args.scale:                                                  # a 의 척도를 w 에 맞춤 (a 는 값일치로 안 눌린 원 커널이라 그냥 섞으면 총 크기가 커진다)
                        r = (w.float().pow(2).mean().sqrt() / (a.float().pow(2).mean().sqrt() + eps)).to(a.dtype)
                        aeff = (1 - lam) * (a * r) + lam * w
                    else:
                        aeff = (1 - lam) * a + lam * w                              # 기저 a + 가소성 w
                    o = torch.einsum('bhtn,bnhc->bthc', aeff.to(vv.dtype), vv); f = torch.einsum('bthc,hcd->btd', o, inner.w_sh)
                    h = inner.phi(h + f)
                lg = inner.w_cls(h).float()[:, :, 2:11]
            h = h.float(); P = lg.argmax(-1); ok = ((P == G[b:b + n]) | ~bl[b:b + n]).all(1); solved[b:b + n, s] = ok; ex[s] += int(ok.sum())
            fin = torch.where(bl[b:b + n], P, x - 2); conf[s] += float(((fin[:, :, None] == fin[:, None, :]) & pm[None]).sum() // 2)
    return ex.astype(int), solved.cumsum(1).bool().sum(0).cpu().numpy(), conf / N
res = {}
for reset in args.reset:
    for lam in args.lam:
        ex, ever, conf = run(lam, reset); tag = f"λ={lam:g} {'reset' if reset else 'carry'}{' scaled' if args.scale else ''}"
        res[tag] = dict(exact=ex.tolist(), ever=ever.tolist(), conf=conf.round(2).tolist())
        print(f"{tag:<16s} 완답 " + " ".join(f"{k}:{ex[k-1]}" for k in marks) + f" | 한번이라도 {ever[-1]} | 충돌/퍼즐 {conf[-1]:.1f}", flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(res, open(args.out, "w"))
