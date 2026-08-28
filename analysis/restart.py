"""장 기준 부분 재시작: 유효하지 않은 판에서 '미해결' 칸 (d_t^(h0)<0 ∧ d_t^(h5)>0, 라벨 무관) 의 h_t 만 init_hidden+잡음으로 되돌리고
나머지 칸은 유지한 채 16 loop 재실행. 유효 완성 격자면 채택. 잡음 표본 --k 개, 반복 --rounds.
사용: python restart.py [--k 3 --rounds 1 --sigma 1.0] [--out JSON]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--k", type=int, default=3); ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--sigma", type=float, default=1.0)
ap.add_argument("--h0", type=int, default=0); ap.add_argument("--h1", type=int, default=5)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.set_grad_enabled(False); torch.manual_seed(args.seed)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
inner = m.inner; core = inner.core; K = m.config.blocks_per_seg; L = m.config.loops
inp, lab, _ = load_test(); N = len(inp); pm = peer_mask()
I = inp.cpu().numpy() - 1; LB = lab.cpu().numpy() - 1; blank = I == 0

def run_from(h_init, x):
    """h_init [n,81,d] 에서 L loop. 반환: 최종 h, argmax 판, d_t(헤드0/5)"""
    fc = core.kernel_fast(); AB = core.W_C(); dt = 1.0 / core.R
    inj = inner._injection(make_batch(x, x)); h = h_init.clone(); d = None
    for loop in range(L):
        for blk in range(K):
            h = inner._boundary(h) + inner.inj_gate * inj
            for _ in range(core.R):
                hh = core.phi(h, dt / 2); f, a, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc); h = core.phi(hh + dt * f, dt / 2)
    d = a.sum(-1)                                                  # [n,H,81]
    return h, (logits(m, h)[:, :, 2:11].argmax(-1) + 1).cpu().numpy(), d[:, args.h0].cpu().numpy(), d[:, args.h1].cpu().numpy()

def valid_of(board, x_np):
    fin = board.astype(np.int64); bl = x_np == 0; fin[~bl] = x_np[~bl]
    return ~np.stack([(fin[:, pm[t]] == fin[:, t:t + 1]).any(1) for t in range(81)], 1).any(1), fin

H = torch.zeros(N, 81, core.d, device="cuda"); board = np.zeros((N, 81), np.int64); D0 = np.zeros((N, 81), np.float32); D5 = np.zeros((N, 81), np.float32)
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N)); x = inp[b]
    h0 = inner.init_hidden.expand(len(x), 81, -1).clone()
    H[b], board[b], D0[b], D5[b] = run_from(h0, x)
valid, fin = valid_of(board, I); exact = (fin == LB).all(1)
res = {"k": args.k, "rounds": args.rounds, "sigma": args.sigma, "base": {"exact": int(exact.sum()), "valid": int(valid.sum())}, "rounds_log": []}
final = fin.copy(); accepted = valid.copy(); rollouts = np.ones(N)
for r in range(args.rounds):
    todo = np.where(~accepted)[0]
    if len(todo) == 0: break
    unres = (D0 < 0) & (D5 > 0) & blank                            # 장 기준 미해결 칸
    gained = 0
    for i in range(0, len(todo), args.bs):
        idx = torch.tensor(todo[i:i + args.bs], device="cuda"); n = len(idx); x = inp[idx]
        u = torch.tensor(unres[todo[i:i + args.bs]], device="cuda")
        for s in range(args.k):
            h = H[idx].clone()
            noise = inner.init_hidden.expand(n, 81, -1) + args.sigma * torch.randn(n, 81, core.d, device="cuda")
            h[u] = noise[u]
            hn, bd, d0, d5 = run_from(h, x); rollouts[todo[i:i + args.bs]] += 1
            v, f = valid_of(bd, I[todo[i:i + args.bs]])
            for j in np.where(v)[0]:
                p = todo[i + j]
                if not accepted[p]: accepted[p] = True; final[p] = f[j]; gained += 1
            # 다음 라운드용 상태 갱신 (미채택만)
            for j in range(n):
                p = todo[i + j]
                if not accepted[p]: H[p] = hn[j]; D0[p] = d0[j]; D5[p] = d5[j]
    ex = (final == LB).all(1)
    res["rounds_log"].append({"round": r, "todo": int(len(todo)), "unresolved_cells_mean": float(unres[todo].sum(1).mean()),
                              "gained_valid": gained, "exact": int(ex.sum()), "accepted": int(accepted.sum()),
                              "exact_given_accepted": float(ex[accepted].mean()), "mean_rollouts": float(rollouts.mean())})
    print(json.dumps(res["rounds_log"][-1]), flush=True)
print(json.dumps(res, indent=1))
if args.out: json.dump(res, open(args.out, "w"), indent=1)
