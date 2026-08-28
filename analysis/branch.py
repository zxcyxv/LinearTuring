"""하네스 수준 백트래킹: 모델(전파 CA) + 라벨 없는 정착 신호(블록 단위 완전 정착)로 분기 탐색.
  1. 16 loop 실행. 판 전체가 블록 단위로 정착(마지막 W 블록 동안 전 빈칸 argmax 불변)하면 채택.
  2. 미정착 판: 미정착 빈칸 중 최종 엔트로피 최대 칸 t 를 고르고 top-k 값을 각각 x[t] 에 핀(주어진 칸 토큰으로 재주입) → 재실행.
     정착한 가지가 있으면 채택(여럿이면 엔트로피 합 최소). 없으면 깊이+1 재귀 (각 가지에서 다시 분기).
라벨은 판정에만 쓴다. 사용: python branch.py [--depth 2 --topk 2 --W 64] [--out JSON]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--depth", type=int, default=2); ap.add_argument("--topk", type=int, default=2); ap.add_argument("--W", type=int, default=64)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--out", default=None)
ap.add_argument("--order", default="uncertain", choices=["uncertain","confident"], help="분기 칸: 미정착 빈칸 중 엔트로피 최대(uncertain) / 최소(confident, 결정화)")
ap.add_argument("--accept", default="frozen", choices=["frozen","valid","either"], help="채택 기준: 정착 / 제약 무위반 완성 격자(유일해 정리) / 둘 중 하나")
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
inp0, lab, _ = load_test(); N = len(inp0); K = m.config.blocks_per_seg; TS = m.config.loops * K
LB = lab.cpu().numpy() - 1; pm = peer_mask()

def run(x):
    """x: [n,81] int32 토큰 (cuda). 반환: argmax 판[n,81] (값 1..9), 칸별 정착[n,81], 칸별 엔트로피[n,81]"""
    n = len(x); AM = np.zeros((n, TS, 81), np.int8); EN = np.zeros((n, 81), np.float32)
    for i in range(0, n, args.bs):
        b = slice(i, min(i + args.bs, n)); nb = b.stop - b.start
        def hook(loop, blk, stage, h, a, i=i, nb=nb):
            if stage != "post_step": return
            t = loop * K + blk; lg = logits(m, h)[:nb, :, 2:11]
            AM[i:i + nb, t] = (lg.argmax(-1) + 1).to(torch.int8).cpu().numpy()
            if t == TS - 1:
                p = lg.softmax(-1); EN[i:i + nb] = (-(p * (p + 1e-9).log()).sum(-1)).cpu().numpy()
        rollout(m, make_batch(x[b], x[b]), hook=hook)
    settled = ~(AM[:, TS - args.W:] != AM[:, TS - args.W - 1:-1]).any(1)
    return AM[:, -1], settled, EN

def solve(x, depth):
    """x: [n,81] cuda 토큰. 반환 dict: board[n,81], accepted[n] bool, nbranch[n]"""
    n = len(x); board, settled, EN = run(x)
    blank = (x == 1).cpu().numpy(); frozen = (settled | ~blank).all(1)
    fin = board.astype(np.int64); fin[~blank] = (x.cpu().numpy() - 1)[~blank]
    valid = ~np.stack([(fin[:, pm[t]] == fin[:, t:t + 1]).any(1) for t in range(81)], 1).any(1)
    cert = {"frozen": frozen, "valid": valid, "either": frozen | valid}[args.accept]
    out_board = board.copy(); accepted = cert.copy(); nb = np.ones(n, int)
    todo = np.where(~cert)[0]
    if depth == 0 or len(todo) == 0: return out_board, accepted, nb, EN
    # 분기 칸·값
    if args.order == "uncertain": ent = np.where(blank & ~settled, EN, -1)[todo]; cell = ent.argmax(1)
    else: ent = np.where(blank & ~settled, EN, 1e9)[todo]; cell = ent.argmin(1)
    # top-k 값: 최종 로짓이 필요 → 간단히 후보집합(자기 판 기준)에서 엔트로피 대신 argmax 판과 후보로 선택
    xs, meta = [], []
    for j, p in enumerate(todo):
        t = int(cell[j]); cands = [v for v in range(1, 10) if not (board[p, pm[t]] == v).any()]
        top = [int(board[p, t])] + [v for v in cands if v != board[p, t]]
        top = (top if len(top) >= args.topk else top + [v for v in range(1, 10) if v not in top])[:args.topk]
        for v in top:
            xv = x[p].clone(); xv[t] = v + 1; xs.append(xv); meta.append((j, v))
    X = torch.stack(xs); sub_board, sub_acc, sub_nb, sub_EN = solve(X, depth - 1)
    # 가지 채택: 정착한 가지 중 엔트로피 합 최소
    best = {}
    for k, (j, v) in enumerate(meta):
        nb[todo[j]] += sub_nb[k]
        if sub_acc[k]:
            s = float(sub_EN[k].sum())
            if j not in best or s < best[j][0]: best[j] = (s, k)
    for j, (s, k) in best.items():
        p = todo[j]; out_board[p] = sub_board[k]; accepted[p] = True
    return out_board, accepted, nb, EN

board, accepted, nbranch, _ = solve(inp0, args.depth)
blank = (inp0 == 1).cpu().numpy(); ok = (board == LB); exact = ok.all(1)
base_board, base_settled, _ = run(inp0); base_exact = (base_board == LB).all(1); base_frozen = (base_settled | ~blank).all(1)
base_fin = base_board.astype(np.int64); base_fin[~blank] = (inp0.cpu().numpy() - 1)[~blank]
base_valid = ~np.stack([(base_fin[:, pm[t]] == base_fin[:, t:t + 1]).any(1) for t in range(81)], 1).any(1)
res = {"depth": args.depth, "topk": args.topk, "W": args.W, "accept": args.accept, "order": args.order,
       "baseline": {"exact": int(base_exact.sum()), "frozen": int(base_frozen.sum()), "exact_given_frozen": float(base_exact[base_frozen].mean()), "valid": int(base_valid.sum()), "exact_given_valid": float(base_exact[base_valid].mean()), "valid_given_exact": float(base_valid[base_exact].mean())},
       "branch": {"exact": int(exact.sum()), "cell_blank": float(ok[blank].mean()), "accepted": int(accepted.sum()),
                  "exact_given_accepted": float(exact[accepted].mean()), "exact_given_rejected": float(exact[~accepted].mean()) if (~accepted).any() else None,
                  "unfrozen_base_to_exact": int((exact & ~base_frozen).sum()), "lost_from_base": int((base_exact & ~exact).sum()),
                  "mean_rollouts_per_puzzle": float(nbranch.mean())}}
print(json.dumps(res, indent=1))
if args.out: json.dump(res, open(args.out, "w"), indent=1)
