"""오라클 상한: 미해결 퍼즐에 정답 핀 K개를 입력으로 주고 재전개 (stdp1, w 세그먼트 초기화). 오답 칸 수 구간별로 해결률.
  핀 종류: wrong = 오답 칸 중 무작위 K개의 정답 / any = 빈칸 중 무작위 K개의 정답.  라벨은 핀 선택·채점에 씀(오라클).
사용: python analysis/pin_oracle.py [--n 512] [--K 1 3 7 15] [--segs 16]"""
import argparse, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--K", type=int, nargs="+", default=[1, 3, 7, 15]); ap.add_argument("--segs", type=int, default=16); args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
def solve(x):
    P = torch.zeros(len(x), 81, dtype=torch.long, device="cuda")
    for b in range(0, len(x), 128):
        xb = x[b:b + 128]; n = len(xb); h = inner.init_hidden.expand(n, 81, -1).clone()
        for s in range(args.segs):
            w = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(xb, xb))
                for _ in range(8):
                    h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
                lg = inner.w_cls(h).float()[:, :, 2:11]
            h = h.float()
        P[b:b + n] = lg.argmax(-1)
    return P
P0 = solve(inp); wrong0 = (P0 != G) & bl; nw = wrong0.sum(1); uns = nw > 0
buckets = [(1, 4), (5, 10), (11, 20), (21, 30), (31, 81)]
print(f"기준(핀 없음): 완답 {int((~uns).sum())}/{N}, 미해결 {int(uns.sum())}.  구간별 미해결 수: " + ", ".join(f"{lo}-{hi}: {int(((nw>=lo)&(nw<=hi)).sum())}" for lo, hi in buckets))
rng = np.random.default_rng(0)
for kind in ("wrong", "any"):
    print(f"\n핀 = {'오답 칸의 정답' if kind=='wrong' else '빈칸의 정답(무작위)'} K개 → 구간별 해결률 (해결/미해결수), 남은 오답 칸 수 중앙값")
    for K in args.K:
        x = inp.clone()
        for i in torch.where(uns)[0].tolist():
            pool = torch.where(wrong0[i] if kind == "wrong" else bl[i])[0].cpu().numpy(); k = min(K, len(pool)); sel = rng.choice(pool, k, replace=False)
            x[i, sel] = (G[i, sel] + 2).to(x.dtype)
        P = solve(x); blk = x == 1; wrongK = (P != G) & blk; solved = ~wrongK.any(1)
        row = []
        for lo, hi in buckets:
            sel = uns & (nw >= lo) & (nw <= hi); n_ = int(sel.sum())
            row.append(f"{lo}-{hi}: {int((solved & sel).sum())}/{n_} (남은 오답 {float(wrongK[sel].sum(1).float().median()) if n_ else 0:.0f})")
        print(f"  K={K:>2d}: 전체 {int((solved & uns).sum())}/{int(uns.sum())}  | " + "  ".join(row))
