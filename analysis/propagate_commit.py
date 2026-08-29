"""전파/결정의 시간 분리 — 경계(쌍선형, 결정)를 끈 '전파 전용' 전개로 강제된 칸을 읽고, 커밋(주어진 칸으로 재주입) → 반복.
  P 단계: act=0 (경계 항등) 으로 L loop 전개. 빈칸의 로짓 여유(top1−top2) ≥ τ 인 칸 = '강제됨' 으로 간주.
  커밋: 강제 칸을 입력 토큰에 고정하고 P 단계 재실행. 새 강제 칸이 없을 때까지 (--rounds 상한).
  마무리: 커밋된 입력으로 원판(경계 켬) 실행 → 유효/완답.
  대조군: 같은 루프를 원판(경계 켬)의 여유로 커밋 (--crit full).
  탐지: 원판 최종 답에 대해 '전파 전용 여유' 를 의심 점수로 써서 오답 칸 AUC (라벨은 평가에만).
사용: python propagate_commit.py [--n 2048 --tau 3 6 --rounds 6 --crit prop full] [--out JSON]"""
import argparse, json, time, numpy as np, torch
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--n", type=int, default=2048); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--tau", type=float, nargs="+", default=[3.0, 6.0]); ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--crit", nargs="+", default=["prop", "full"]); ap.add_argument("--out", default=None)
args = ap.parse_args(); torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
inp, lab, depth = load_test(args.n); N = len(inp); I0 = inp.cpu().numpy(); LB = lab.cpu().numpy() - 2; pm = peer_mask()
blank0 = I0 == 1
ZERO = lambda g: 0 * g

def run(x, act=None):
    """x [N,81] int32 cuda → 9-way 로짓 [N,81,9] (numpy)"""
    out = np.zeros((N, 81, 9), np.float32)
    for i in range(0, N, args.bs):
        b = slice(i, min(i + args.bs, N)); h = rollout(m, make_batch(x[b], x[b]), act=act)
        out[b] = logits(m, h)[:, :, 2:11].float().cpu().numpy()
    return out

def margin(lg):
    s = np.sort(lg, -1); return lg.argmax(-1), s[..., -1] - s[..., -2]

def valid_of(board, x_np):
    fin = board.copy(); bl = x_np == 1; fin[~bl] = x_np[~bl] - 2
    return ~np.stack([(fin[:, pm[t]] == fin[:, t:t + 1]).any(1) for t in range(81)], 1).any(1), fin

def auc(score, pos):
    """score 높을수록 pos. 순위 기반 AUC"""
    from scipy.stats import rankdata
    r = rankdata(score); n1 = pos.sum(); n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

t0 = time.time()
lg_full = run(inp); P_full, M_full = margin(lg_full)
lg_prop = run(inp, ZERO); P_prop, M_prop = margin(lg_prop)
print(f"[pass 2회 {time.time()-t0:.0f}s]")
valid, fin = valid_of(P_full, I0); exact = (fin == LB).all(1)
res = {"n": N, "base": {"exact": int(exact.sum()), "valid": int(valid.sum()), "cell_blank": float((P_full == LB)[blank0].mean())},
       "prop_only": {"cell_blank": float((P_prop == LB)[blank0].mean()),
                     "agree_full": float((P_prop == P_full)[blank0].mean())}}
# 전파 전용 여유 τ 별: 강제 칸 수·정밀도 (빈칸 기준)
res["prop_only"]["forced_by_tau"] = {}
for tau in [1, 2, 3, 4, 6, 8, 12]:
    f = blank0 & (M_prop >= tau); res["prop_only"]["forced_by_tau"][tau] = {"frac_blank": float(f.sum() / blank0.sum()), "precision": float((P_prop == LB)[f].mean()) if f.any() else None}
    f2 = blank0 & (M_full >= tau); res["prop_only"]["forced_by_tau"][tau]["full_frac"] = float(f2.sum() / blank0.sum()); res["prop_only"]["forced_by_tau"][tau]["full_precision"] = float((P_full == LB)[f2].mean()) if f2.any() else None
# 탐지: 원판 답 P_full 에 대한 의심 점수
wrong = (P_full != LB)[blank0]
sc_prop = -(lg_prop[np.arange(N)[:, None], np.arange(81)[None], P_full] - np.where(np.eye(9, dtype=bool)[P_full], -1e9, lg_prop).max(-1))[blank0]   # 전파 전용에서 원판 답의 여유 (음수화)
sc_full = -M_full[blank0]
sc_dis = (P_prop != P_full)[blank0].astype(float)
res["detect_auc"] = {"prop_margin_of_full_answer": auc(sc_prop, wrong), "full_own_margin": auc(sc_full, wrong), "disagree": auc(sc_dis, wrong),
                     "wrong_frac": float(wrong.mean())}
print(json.dumps({k: res[k] for k in ["base", "prop_only", "detect_auc"]}, indent=1, ensure_ascii=False), flush=True)

# 커밋 루프
res["commit"] = {}
for crit in args.crit:
    act = ZERO if crit == "prop" else None
    for tau in args.tau:
        x = inp.clone(); X = I0.copy(); log = []; committed = np.zeros((N, 81), bool)
        for r in range(args.rounds):
            lg = run(x, act); P, M = margin(lg)
            new = (X == 1) & (M >= tau)
            if new.sum() == 0: break
            X[new] = P[new] + 2; committed |= new; x = torch.tensor(X, dtype=torch.int32, device="cuda")
            prec_new = float((P == LB)[new].mean()); prec_cum = float((X[committed] - 2 == LB[committed]).mean())
            log.append({"round": r, "new": int(new.sum()), "cum": int(committed.sum()), "cum_frac_blank": float(committed.sum() / blank0.sum()),
                        "prec_new": prec_new, "prec_cum": prec_cum, "puzzles_touched": int(new.any(1).sum())})
            print(f"[{crit} τ={tau}] " + json.dumps(log[-1]), flush=True)
        lg = run(x); P, M = margin(lg); v, f = valid_of(P, I0); ex = (f == LB).all(1)
        # 커밋이 하나라도 틀린 퍼즐은 원리상 회복 불가 → 그 수도 기록
        bad = np.array([(X[i][committed[i]] - 2 != LB[i][committed[i]]).any() for i in range(N)])
        res["commit"][f"{crit}_tau{tau}"] = {"rounds": log, "final": {"exact": int(ex.sum()), "valid": int(v.sum()), "cell_blank": float((P == LB)[blank0].mean()),
                                                                        "puzzles_with_wrong_commit": int(bad.sum()), "exact_among_clean": int(ex[~bad].sum()), "clean": int((~bad).sum())}}
        print(f"[{crit} τ={tau}] final " + json.dumps(res["commit"][f"{crit}_tau{tau}"]["final"]), flush=True)
print(json.dumps(res, indent=1, ensure_ascii=False))
if args.out: json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
