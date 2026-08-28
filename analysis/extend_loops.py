"""정착 기준 loops 연장: 16 loop 뒤 '판 전체 정착'(모든 빈칸 argmax 가 마지막 8 loop 동안 불변) 여부로 퍼즐을 나누고,
미정착 퍼즐만 loops 를 연장했을 때 정착·정답이 늘어나는지. 라벨은 판정에만 쓴다.
사용: python extend_loops.py [--ckpt PATH --bilinear 0|1] [--loops 16 32 64 128] [--out JSON]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, rollout, logits, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--loops", type=int, nargs="+", default=[16, 32, 64, 128]); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_grad_enabled(False)
Lmax = max(args.loops)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs, loops=Lmax)
inp, lab, _ = load_test(); N = len(inp); K = m.config.blocks_per_seg
AM = np.zeros((N, Lmax, 81), np.int8)          # loop 말 argmax
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N)); n = b.stop - b.start
    def hook(loop, blk, stage, h, a, i=i, n=n):
        if stage == "post_step" and blk == K - 1:
            AM[i:i + n, loop] = (logits(m, h)[:n, :, 2:11].argmax(-1) + 1).to(torch.int8).cpu().numpy()
    rollout(m, make_batch(inp[b], lab[b]), hook=hook)
I = inp.cpu().numpy() - 1; LB = lab.cpu().numpy() - 1; blank = I == 0
res = {}
for L in args.loops:
    W = min(8, L // 2)
    frozen = (~(AM[:, L - W:L] != AM[:, L - W - 1:L - 1]).any(1) | ~blank).all(1)     # 마지막 W loop 동안 전 빈칸 불변
    ok = (AM[:, L - 1] == LB); exact = ok.all(1)
    res[L] = {"exact": int(exact.sum()), "cell": float(ok[blank].mean()), "frozen": int(frozen.sum()),
              "exact_given_frozen": float(exact[frozen].mean()) if frozen.any() else None,
              "exact_given_unfrozen": float(exact[~frozen].mean()) if (~frozen).any() else None}
# 16 에서 미정착이던 퍼즐 추적
W = 8; fr16 = (~(AM[:, 16 - W:16] != AM[:, 16 - W - 1:15]).any(1) | ~blank).all(1); ex16 = (AM[:, 15] == LB).all(1)
track = {}
for L in args.loops:
    exL = (AM[:, L - 1] == LB).all(1); Wl = min(8, L // 2)
    frL = (~(AM[:, L - Wl:L] != AM[:, L - Wl - 1:L - 1]).any(1) | ~blank).all(1)
    track[L] = {"unfrozen@16 → exact": int(exL[~fr16].sum()), "unfrozen@16 → frozen": int(frL[~fr16].sum()),
                "frozen@16 → still exact": int(exL[fr16].sum()), "exact@16 → lost": int((ex16 & ~exL).sum())}
res["track_unfrozen_at_16"] = {"n_unfrozen@16": int((~fr16).sum()), "n_frozen@16": int(fr16.sum()), "per_loops": track}
print(json.dumps(res, indent=1));
if args.out: json.dump(res, open(args.out, "w"), indent=1)
