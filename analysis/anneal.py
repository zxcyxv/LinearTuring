"""진폭 담금질: 유효하지 않은 판에서 세그먼트 경계의 h 를 α 배로 줄여(경계 2차항 < 어텐션 1차항 → '재가열') --loops 더 실행, 유효 격자면 채택.
사용: python anneal.py --alpha 0.5 [--rounds 2 --loops 16] [--out JSON]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, logits, peer_mask, CKPT_DEFAULT
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--alpha", type=float, nargs="+", default=[0.5]); ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--loops", type=int, default=16)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--out", default=None)
args = ap.parse_args(); torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs); inner = m.inner; core = inner.core; K = m.config.blocks_per_seg
inp, lab, _ = load_test(); N = len(inp); I = inp.cpu().numpy(); LB = lab.cpu().numpy(); blank = I == 1; pm = peer_mask()
fc = core.kernel_fast(); AB = core.W_C(); dt = 1.0 / core.R
def run(h, x, loops):
    inj = inner._injection(make_batch(x, x))
    for _ in range(loops * K):
        h = inner._boundary(h) + inner.inj_gate * inj
        for _ in range(core.R):
            hh = core.phi(h, dt / 2); f, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc); h = core.phi(hh + dt * f, dt / 2)
    return h
def valid_board(h):
    bd = (logits(m, h)[:, :, 2:11].argmax(-1) + 2).cpu().numpy(); bd[~blank[:len(bd)] if False else (I[:len(bd)] > 1)] = 0  # placeholder
    return bd
H = torch.zeros(N, 81, core.d, device="cuda")
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N)); H[b] = run(inner.init_hidden.expand(b.stop - b.start, 81, -1).clone(), inp[b], 16)
def boards(Hs, idx):
    out = np.zeros((len(idx), 81), np.int64)
    for i in range(0, len(idx), args.bs):
        j = idx[i:i + args.bs]; out[i:i + len(j)] = (logits(m, Hs[j])[:, :, 2:11].argmax(-1) + 2).cpu().numpy()
    return out
def validity(bd, idx):
    fin = bd.copy(); giv = I[idx] > 1; fin[giv] = LB[idx][giv]
    return ~np.stack([(fin[:, pm[t]] == fin[:, t:t + 1]).any(1) for t in range(81)], 1).any(1), fin
allidx = np.arange(N); bd = boards(H, allidx); valid, fin = validity(bd, allidx); exact = (fin == LB).all(1)
res = {"base": {"exact": int(exact.sum()), "valid": int(valid.sum())}}
for alpha in args.alpha:
    final = fin.copy(); acc = valid.copy(); Hc = H.clone(); rolls = np.ones(N); log = []
    for r in range(args.rounds):
        todo = np.where(~acc)[0]
        if len(todo) == 0: break
        for i in range(0, len(todo), args.bs):
            j = torch.tensor(todo[i:i + args.bs], device="cuda"); Hc[j] = run(alpha * Hc[j], inp[j], args.loops); rolls[todo[i:i + args.bs]] += args.loops / 16
        bdt = boards(Hc, todo); v, f = validity(bdt, todo)
        for k in np.where(v)[0]: acc[todo[k]] = True; final[todo[k]] = f[k]
        ex = (final == LB).all(1); log.append({"round": r, "todo": int(len(todo)), "gained_valid": int(v.sum()), "exact": int(ex.sum()), "exact_given_accepted": float(ex[acc].mean()), "mean_rollouts": float(rolls.mean())})
        print(f"alpha {alpha} " + json.dumps(log[-1]), flush=True)
    res[f"alpha={alpha}"] = log
print(json.dumps(res, indent=1))
if args.out: json.dump(res, open(args.out, "w"), indent=1)
