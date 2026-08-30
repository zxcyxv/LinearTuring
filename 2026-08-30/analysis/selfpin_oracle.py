"""자기 예측 중 맞은 것만(오라클 필터) 이산화해서 입력으로 재투입, 틀린 칸은 빈칸으로 되돌려 재전개. 반복 R 라운드.
  상한: '내 예측 중 무엇이 맞는지' 를 완벽히 안다면 남은 부분을 풀 수 있는가. stdp1, w 세그먼트 초기화.
사용: python analysis/selfpin_oracle.py [--n 512] [--rounds 3] [--segs 16]"""
import argparse, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--segs", type=int, default=16); args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl0 = inp == 1
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
buckets = [(1, 10), (11, 20), (21, 30), (31, 81)]
x = inp.clone(); P = solve(x); wrong = (P != G) & bl0; nw0 = wrong.sum(1); uns0 = nw0 > 0
print(f"라운드 0: 완답 {int((~uns0).sum())}/{N}, 미해결 {int(uns0.sum())}. 구간별: " + ", ".join(f"{lo}-{hi}: {int(((nw0>=lo)&(nw0<=hi)).sum())}" for lo, hi in buckets))
prev_wrong = wrong.clone()
for r in range(1, args.rounds + 1):
    # 오라클 필터: 원래 빈칸 중 지금 예측이 맞은 칸 → 입력으로 고정, 틀린 칸 → 빈칸
    x = inp.clone(); corr = (P == G) & bl0; x[corr] = (G[corr] + 2).to(x.dtype)
    pinned = corr.sum(1); blank_left = (x == 1).sum(1)
    P = solve(x); wrong = (P != G) & bl0; solved = ~wrong.any(1)
    still = (wrong & prev_wrong).sum(1); newwrong = (wrong & ~prev_wrong).sum(1); fixed = (prev_wrong & ~wrong).sum(1)
    print(f"\n라운드 {r} (핀 = 직전 예측 중 맞은 빈칸, 미해결 퍼즐 핀 중앙값 {float(pinned[uns0].float().median()):.0f}, 남은 빈칸 중앙값 {float(blank_left[uns0].float().median()):.0f}):  누적 완답 {int(solved.sum())}/{N}")
    for lo, hi in buckets:
        sel = uns0 & (nw0 >= lo) & (nw0 <= hi); n_ = int(sel.sum())
        if n_ == 0: continue
        print(f"   처음 오답 {lo}-{hi}칸 ({n_}개): 해결 {int((solved & sel).sum())}/{n_} | 이번 라운드에 고쳐진 칸 중앙값 {float(fixed[sel].float().median()):.0f}, 그대로 틀린 칸 {float(still[sel].float().median()):.0f}, 새로 틀린 칸 {float(newwrong[sel].float().median()):.0f}  (남은 오답 중앙값 {float(wrong[sel].sum(1).float().median()):.0f})")
    prev_wrong = wrong.clone()
