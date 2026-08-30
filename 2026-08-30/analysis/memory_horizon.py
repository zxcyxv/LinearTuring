"""기억 시평 실험: 결합 기억 w 는 ~1/η 블록 전의 결합을 들고 있는 지연 요소다.
  가설 A (지연이 탈출을 만든다): 기억을 없애면(η=1) 반복 외삽 이득이 줄어야 한다.
  가설 B (시평이 짧아 회전한다): 시평을 늘리면(η↓) 회전이 탐색으로 바뀌어 이득이 늘어야 한다.
  carry 모드(세그먼트 넘어 w 유지)에서만 시평이 8블록을 넘는다 → 기본은 carry. reset 도 대조로.
사용: python analysis/memory_horizon.py [--n 512] [--segs 128] [--eta 1.0 -1 0.2 0.05] [--reset 0 1]"""
import argparse, json, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--segs", type=int, default=128); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--eta", type=float, nargs="+", default=[-1.0, 1.0, 0.2, 0.05]); ap.add_argument("--reset", type=int, nargs="+", default=[0])
ap.add_argument("--scale", type=float, nargs="+", default=[], help="학습 η 에 곱하는 배율 — 헤드별 다양성을 보존한 채 시평만 바꾼다 (균일 치환의 교란 통제)")
ap.add_argument("--marks", type=int, nargs="+", default=[16, 32, 64, 96, 128]); ap.add_argument("--out", default=os.path.join(ROOT, "results", "json", "memory_horizon.json"))
args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=args.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
               stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; K = 8; S = args.segs; marks = [k for k in args.marks if k <= S]; eta_learned = torch.sigmoid(inner.eta_raw).float()
def run(eta_val, reset, track_idx=None, scale=None):
    eta = (eta_learned * scale).clamp(1e-4, 0.999) if scale is not None else (eta_learned if eta_val < 0 else torch.full_like(eta_learned, eta_val))
    ex = np.zeros(S); conf = np.zeros(S); solved = torch.zeros(N, S, dtype=torch.bool, device="cuda"); traj = {}
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
                    o = torch.einsum('bhtn,bnhc->bthc', w.to(vv.dtype), vv); f = torch.einsum('bthc,hcd->btd', o, inner.w_sh)
                    h = inner.phi(h + f)
                lg = inner.w_cls(h).float()[:, :, 2:11]
            h = h.float(); P = lg.argmax(-1); ok = ((P == G[b:b + n]) | ~bl[b:b + n]).all(1); solved[b:b + n, s] = ok; ex[s] += int(ok.sum())
            fin = torch.where(bl[b:b + n], P, x - 2); conf[s] += float(((fin[:, :, None] == fin[:, None, :]) & pm[None]).sum() // 2)
            if track_idx is not None and b == 0: traj.setdefault(s, h[track_idx].clone())
    ever = solved.cumsum(1).bool().sum(0).cpu().numpy()
    return ex.astype(int), ever, conf / N, traj
res = {}
for reset in args.reset:
    for sc in args.scale:
        ex, ever, conf, _ = run(-1, reset, scale=sc)
        e = (eta_learned * sc).clamp(1e-4, .999).flatten().cpu().numpy()
        tag = f"η×{sc:g} {'reset' if reset else 'carry'}"; res[tag] = dict(exact=ex.tolist(), ever=ever.tolist(), conf=conf.round(2).tolist())
        print(f"{tag:<22s} 시평 {1/e.mean():>6.1f}블록(다양성 보존, {e.min():.2f}~{e.max():.2f}) | 완답 " + " ".join(f"{k}:{ex[k-1]}" for k in marks) + f" | 한번이라도 {ever[-1]} | 충돌 16:{conf[15]:.1f} {S}:{conf[-1]:.1f}", flush=True)
        json.dump(res, open(args.out, "w"))
    for ev in args.eta:
        tag = f"η={'학습' if ev < 0 else f'{ev:g}'} {'reset' if reset else 'carry'}"
        ex, ever, conf, traj = run(ev, reset, track_idx=None)
        hor = "8블록 이내" if reset else (f"{1/max(np.mean(eta_learned.cpu().numpy()),1e-9):.1f}블록" if ev < 0 else f"{1/ev:.0f}블록")
        res[tag] = dict(exact=ex.tolist(), ever=ever.tolist(), conf=conf.round(2).tolist())
        print(f"{tag:<22s} 시평 {hor:>9s} | 완답 " + " ".join(f"{k}:{ex[k-1]}" for k in marks) + f" | 한번이라도 {ever[-1]} | 충돌/퍼즐 16:{conf[15]:.1f} {S}:{conf[-1]:.1f}", flush=True)
        json.dump(res, open(args.out, "w"))
