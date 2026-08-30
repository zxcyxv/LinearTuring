"""세그먼트 수준 변위의 모멘텀 (stdp1, w 세그먼트 초기화, 라벨 없음):
    세그먼트 끝마다  Δ = h_s − h_{s−1},  m ← (1−δ) m + Δ,  h ← h + κ·m     (heavy-ball, 세그먼트 단위).
  세그먼트 사상의 고정점에서 Δ=0 이라 맞는 칸은 불변, 느린 단조 하강은 (1+κ/δ) 배 증폭. m 은 퍼즐 시작에만 0. --warm 뒤부터.
  (실패한 두 판: 전달장 f 의 블록 적분 → 고정점 이동으로 붕괴(완답 8); 블록 Δh 의 모멘텀 → 세그먼트 내부 과도 패턴을 증폭해 붕괴.)
사용: python analysis/integrator.py [--n 512] [--segs 64] [--delta 0.0625] [--kappa 0.0625 0.125 0.25] [--warm 0 16]"""
import argparse, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--segs", type=int, default=64); ap.add_argument("--delta", type=float, nargs="+", default=[1 / 16])
ap.add_argument("--kappa", type=float, nargs="+", default=[0.0, 0.0625, 0.125, 0.25]); ap.add_argument("--warm", type=int, nargs="+", default=[0, 16]); ap.add_argument("--marks", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128])
args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; eta_h = torch.sigmoid(inner.eta_raw).float(); K = 8; S = args.segs; marks = [k for k in args.marks if k <= S]
def run(delta, kappa, warm):
    ex = np.zeros(S); solved = torch.zeros(N, S, dtype=torch.bool, device="cuda"); conf = np.zeros(S); flips = np.zeros(S)
    for b in range(0, N, 128):
        x = inp[b:b + 128]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); mI = torch.zeros_like(h); P = None; h_seg = h.clone()
        for s in range(S):
            w = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
                for _ in range(K):
                    h = inner.boundary(h); h = h + inner.inj_gate * inj
                    a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
                    Gm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn); w = Gm if w is None else w + eta_h * (Gm - w)
                    o = torch.einsum('bhtn,bnhc->bthc', w.to(vv.dtype), vv); f = torch.einsum('bthc,hcd->btd', o, inner.w_sh).float()
                    h = inner.phi(h + f)
                lg = inner.w_cls(h).float()[:, :, 2:11]
            h = h.float()
            if kappa > 0:
                if s >= warm and s > 0: mI = (1 - delta) * mI + (h - h_seg); h = h + kappa * mI
                h_seg = h.clone()
            Pn = lg.argmax(-1); ok = ((Pn == G[b:b + n]) | ~bl[b:b + n]).all(1); solved[b:b + n, s] = ok; ex[s] += int(ok.sum())
            fin = torch.where(bl[b:b + n], Pn, x - 2); conf[s] += int(((fin[:, :, None] == fin[:, None, :]) & pm[None]).sum() // 2)
            if P is not None: flips[s] += int(((Pn != P) & bl[b:b + n]).sum())
            P = Pn
    ever = solved.cumsum(1).bool().sum(0).cpu().numpy(); keep16 = (solved & solved[:, 15:16]).sum(0).cpu().numpy()
    return dict(exact=ex.astype(int), ever=ever, keep16=keep16, conf=conf / N, flips=flips / int(bl.sum()))
for delta in args.delta:
    for warm in args.warm:
        for kappa in args.kappa:
            if kappa == 0 and warm != args.warm[0]: continue
            d = run(delta, kappa, warm); tag = f"δ={delta:.4f} κ={kappa:.4f} warm={warm} (정상 이득 1+κ/δ={1+kappa/delta:.1f})" if kappa > 0 else "기준 (적분 없음)"
            print(f"{tag}: 완답 " + " ".join(f"{k}:{d['exact'][k-1]}" for k in marks) + f" | ever {d['ever'][-1]} keep16 {d['keep16'][-1]} | 충돌/퍼즐 16:{d['conf'][15]:.2f} 64:{d['conf'][min(63,S-1)]:.2f} | 뒤집힘 64:{d['flips'][min(63,S-1)]:.3f}", flush=True)
