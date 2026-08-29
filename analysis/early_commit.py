"""조기 여유 커밋 루프 — 첫 세그먼트(스텝 --early) 의 로짓 여유는 빈칸 상태가 아직 동일해 '주어진 칸에서 온 정보' 만 반영한다.
그 여유 ≥ θ 인 칸을 주어진 칸으로 핀 → 처음부터 재실행 → 반복 (--rounds). 라벨은 평가에만.
출력: 라운드별 핀 정/오, 오핀 포함 퍼즐 수, 최종 완답. 또한 라운드 0 핀을 전파깊이(cell_depth: 1=주어진 칸만으로 확정)와 대조.
사용: python early_commit.py [--theta 18 21] [--rounds 6] [--early 8] [--out JSON]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--n", type=int, default=2048); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--theta", type=float, nargs="+", default=[18, 21]); ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--early", type=int, default=8)
ap.add_argument("--out", default=None)
args = ap.parse_args(); torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=True, batch_size=args.bs); inner = m.inner; core = inner.core; K = m.config.blocks_per_seg; L = m.config.loops
inp, lab, depth = load_test(args.n); N = len(inp); I0 = inp.cpu().numpy(); G = lab.cpu().numpy() - 2; pm = peer_mask(); blank0 = I0 == 1
fc = core.kernel_fast(); AB = core.W_C()

def run(x):
    """→ (조기 argmax, 조기 여유, 최종 argmax)"""
    Pe = np.zeros((N, 81), np.int64); Me = np.zeros((N, 81), np.float32); Pf = np.zeros((N, 81), np.int64)
    for b in range(0, N, args.bs):
        xb = x[b:b + args.bs]; n = len(xb); inj = inner._injection(make_batch(xb, xb)); h = inner.init_hidden.expand(n, 81, -1).clone(); s = 0
        for _ in range(L):
            for _ in range(K):
                h = inner._boundary(h) + inner.inj_gate * inj; hp = core.phi(h, .5); f, a, *_ = core.field(hp, None, None, None, AB, fast_ctx=fc); h = core.phi(hp + f, .5); s += 1
                if s == args.early:
                    lg = logits(m, h)[:, :, 2:11].float(); sv = lg.sort(-1).values
                    Pe[b:b + n] = lg.argmax(-1).cpu().numpy(); Me[b:b + n] = (sv[..., -1] - sv[..., -2]).cpu().numpy()
        Pf[b:b + n] = logits(m, h)[:, :, 2:11].argmax(-1).cpu().numpy()
    return Pe, Me, Pf

def score(Pf, X):
    fin = Pf.copy(); giv = X != 1; fin[giv] = X[giv] - 2
    valid = ~np.stack([(fin[:, pm[t]] == fin[:, t:t + 1]).any(1) for t in range(81)], 1).any(1)
    return int((fin == G).all(1).sum()), int(valid.sum())

res = {"n": N}
Pe0, Me0, Pf0 = run(inp); res["base_exact"], _ = score(Pf0, I0); print("base exact", res["base_exact"], flush=True)
# 라운드 0 핀 vs 전파깊이
if depth is not None:
    res["depth_check"] = {}
    for th in args.theta:
        f = blank0 & (Me0 >= th); ok = Pe0 == G
        res["depth_check"][th] = {"pin_correct_depth1": int((f & ok & (depth == 1)).sum()), "pin_correct_depth_other": int((f & ok & (depth != 1)).sum()),
                                  "pin_wrong_depth1": int((f & ~ok & (depth == 1)).sum()), "pin_wrong_depth_other": int((f & ~ok & (depth != 1)).sum()),
                                  "depth1_total": int((blank0 & (depth == 1)).sum()), "depth1_pinned": int((f & (depth == 1)).sum())}
    print(json.dumps(res["depth_check"], indent=1), flush=True)
res["loops"] = {}
for th in args.theta:
    X = I0.copy(); committed = np.zeros((N, 81), bool); log = []; Pe, Me, Pf = Pe0, Me0, Pf0
    for r in range(args.rounds):
        new = (X == 1) & (Me >= th)
        if new.sum() == 0: break
        X[new] = Pe[new] + 2; committed |= new
        bad = np.array([(X[i][committed[i]] - 2 != G[i][committed[i]]).any() for i in range(N)])
        Pe, Me, Pf = run(torch.tensor(X, dtype=torch.int32, device="cuda")); ex, va = score(Pf, X)
        log.append({"round": r, "new_pins": int(new.sum()), "new_correct": int((Pe0 if r == 0 else None) is None or True) and int(((X - 2 == G) & new).sum()), "new_wrong": int(((X - 2 != G) & new).sum()),
                    "cum_pins": int(committed.sum()), "puzzles_with_wrong_pin": int(bad.sum()), "exact": ex, "valid": va, "exact_among_clean": int(((Pf.copy() * 0 + 1) * 0).sum())})
        fin = Pf.copy(); giv = X != 1; fin[giv] = X[giv] - 2; exv = (fin == G).all(1); log[-1]["exact_among_clean"] = int(exv[~bad].sum()); log[-1]["clean"] = int((~bad).sum())
        print(f"[θ={th}] " + json.dumps(log[-1]), flush=True)
    res["loops"][th] = log
print(json.dumps(res, indent=1))
if args.out: json.dump(res, open(args.out, "w"), indent=1)
