"""논문(Differentiable plasticity/Backpropamine)의 충실한 이식 시험 — 추론 시점, 학습 없음.
  논문 구조: 연결 = (부호·위상을 담은) 기저 + α·(순수 pre·post 흔적).  흔적은 단독으로 결합이 되지 않는다.
  우리 대응:  결합 = a_tn + α·r·E_tn ,  E ← (1−η)E + η·⟨v̂_t,v̂_n⟩ ,  r = RMS(a)/RMS(E) (척도 맞춤)
  현재 모델은 결합 = EMA(a·agree) 로 학습됨 — 흔적에 그래프를 접어 넣은 하이브리드. 위가 논문 형태.
  α<0 이면 "일관되게 같은 값을 든 쌍일수록 더 밀어낸다" = 위반 구동 압력. 그 부호를 손으로 고르지 않고 스윕한다.
사용: python 2026-08-30/analysis/agree_trace.py [--alpha -1 -0.5 -0.25 0.25 0.5] [--segs 32]"""
import argparse, json, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--segs", type=int, default=32); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--alpha", type=float, nargs="+", default=[-1.0, -0.5, -0.25, 0.25, 0.5]); ap.add_argument("--reset", type=int, default=1)
ap.add_argument("--marks", type=int, nargs="+", default=[8, 16, 32]); ap.add_argument("--out", default=os.path.join(ROOT, "2026-08-30", "results", "json", "agree_trace.json"))
args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=args.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
               stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; eta = torch.sigmoid(inner.eta_raw).float(); K = 8; S = args.segs; marks = [k for k in args.marks if k <= S]
def run(alpha):
    ex = np.zeros(S); conf = np.zeros(S)
    for b in range(0, N, args.bs):
        x = inp[b:b + args.bs]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); E = None
        for s in range(S):
            if args.reset: E = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
                for _ in range(K):
                    h = inner.boundary(h); h = h + inner.inj_gate * inj
                    a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
                    agree = torch.einsum('bthc,bnhc->bhtn', vn, vn)                  # 순수 pre·post 흔적 (그래프 없음)
                    E = agree if E is None else E + eta * (agree - E)
                    r = (a.float().pow(2).mean().sqrt() / (E.float().pow(2).mean().sqrt() + eps)).to(a.dtype)
                    aeff = a + alpha * r * E                                          # 기저(부호·위상) + α·흔적
                    o = torch.einsum('bhtn,bnhc->bthc', aeff.to(vv.dtype), vv); f = torch.einsum('bthc,hcd->btd', o, inner.w_sh)
                    h = inner.phi(h + f)
                lg = inner.w_cls(h).float()[:, :, 2:11]
            h = h.float(); P = lg.argmax(-1); ex[s] += int(((P == G[b:b + n]) | ~bl[b:b + n]).all(1).sum())
            fin = torch.where(bl[b:b + n], P, x - 2); conf[s] += float(((fin[:, :, None] == fin[:, None, :]) & pm[None]).sum() // 2)
    return ex.astype(int), conf / N
res = {}
for al in args.alpha:
    ex, conf = run(al); res[f"α={al:g}"] = dict(exact=ex.tolist(), conf=conf.round(2).tolist())
    print(f"α={al:>5g}  완답 " + " ".join(f"{k}:{ex[k-1]}" for k in marks) + f" | 충돌/퍼즐 {conf[-1]:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(res, open(args.out, "w"))
