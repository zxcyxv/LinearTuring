"""CA 관점 두 측정 (라벨 불필요 신호 ↔ 정답, 그리고 규칙 추출 1단계).
A. 칸별 고정점 도달 여부가 정답을 예측하는가: settle(마지막 argmax 변화 스텝) · 후반 flip 수 · 최종 ‖Δh‖ · 최종 엔트로피 의 AUC.
B. 블록 사상이 naked/hidden single 과 일치하는가: 모델의 현재 argmax 판에서 후보집합을 계산해 다음 argmax 가 그 안에 있는지,
   후보가 1개일 때 그 값을 택하는지, hidden single 이 있을 때 따르는지.
사용: python fixpoint_rules.py [--ckpt PATH --bilinear 0|1] [--n 2048] [--out JSON]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--n", type=int, default=2048); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
inp, lab, depth = load_test(args.n); N = len(inp)
L, K = m.config.loops, m.config.blocks_per_seg; TS = L * K
pm = peer_mask()
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
UNITS = [np.where(r == i)[0] for i in range(9)] + [np.where(c == i)[0] for i in range(9)] + [np.where(bx == i)[0] for i in range(9)]

AM = np.zeros((N, TS, 81), np.int8); EN = np.zeros((N, 81), np.float32); DH = np.zeros((N, 81), np.float32)
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N)); n = b.stop - b.start; prev = {}
    def hook(loop, blk, stage, h, a, i=i, n=n):
        t = loop * K + blk
        if stage == "pre": prev["h"] = h
        if stage != "post_step": return
        lg = logits(m, h)[:n, :, 2:11]
        AM[i:i + n, t] = (lg.argmax(-1) + 1).to(torch.int8).cpu().numpy()          # 값 1..9
        if t == TS - 1:
            p = lg.softmax(-1); EN[i:i + n] = (-(p * (p + 1e-9).log()).sum(-1)).cpu().numpy()
            DH[i:i + n] = ((h - prev["h"])[:n].norm(dim=-1) / (h[:n].norm(dim=-1) + 1e-9)).cpu().numpy()
    rollout(m, make_batch(inp[b], lab[b]), hook=hook)

I = inp.cpu().numpy() - 1; LB = lab.cpu().numpy() - 1                      # 값 공간 0=빈칸, 1..9
blank = I == 0; ok = AM[:, -1] == LB
chg = AM[:, 1:] != AM[:, :-1]                                              # [N,TS-1,81]
settle = np.where(chg.any(1), TS - 1 - np.argmax(chg[:, ::-1], 1), 0)       # 마지막 변화 스텝
flips_late = chg[:, TS // 2:].sum(1)                                        # 후반 절반 flip 수
def auc(score, pos):                                                        # pos=True 가 높은 score 를 갖는가
    s, y = score[blank], pos[blank]; o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    return (rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
wrong = ~ok
res = {"cell_acc_blank": float(ok[blank].mean()), "n_blank": int(blank.sum()), "n_wrong": int((wrong & blank).sum())}
res["AUC(wrong|signal)"] = {"settle_step": auc(settle, wrong), "flips_late": auc(flips_late.astype(float), wrong),
                            "final_dh": auc(DH, wrong), "final_entropy": auc(EN, wrong)}
frozen = settle < TS // 2                                                   # 후반 절반 동안 argmax 불변
res["wrong_cells_frozen_frac"] = float(frozen[wrong & blank].mean()); res["right_cells_frozen_frac"] = float(frozen[ok & blank].mean())
res["settle_median_right/wrong"] = [float(np.median(settle[ok & blank])), float(np.median(settle[wrong & blank]))]
if depth is not None:
    srch = depth == -1
    res["search_cells"] = {"acc": float(ok[srch].mean()), "wrong_frozen_frac": float(frozen[wrong & srch].mean()),
                           "AUC_settle": auc(np.where(srch, settle, 0), wrong & srch) if False else None}
# 퍼즐 단위: 완답 vs 미완답 퍼즐의 '판 전체 정착' 비율
puz_ok = ok.all(1); puz_frozen = (frozen | ~blank).all(1)
res["puzzle"] = {"exact": int(puz_ok.sum()), "frozen_given_exact": float(puz_frozen[puz_ok].mean()), "frozen_given_wrong": float(puz_frozen[~puz_ok].mean())}

# ── B. 규칙 추출 1단계: 모델의 현재 판 → 후보집합 → 다음 argmax ──
def cands_from(board):                                                     # board [N,81] 값 0..9 → 후보 [N,81,9] bool (빈칸 기준 아님: 모든 칸)
    C = np.ones((len(board), 81, 9), bool)
    for t in range(81):
        pv = board[:, pm[t]]                                                # [N,20]
        for v in range(1, 10): C[:, t, v - 1] &= ~(pv == v).any(1)
    return C
in_cand = []; naked_follow = []; naked_n = 0; hidden_follow = []; hidden_n = 0; given_viol = []
for t in [0, 1, 3, 7, 15, 31, 63, 95, 126]:
    board = AM[:, t].astype(np.int64); board[~blank] = I[~blank]            # 주어진 칸은 고정
    C = cands_from(board); nxt = AM[:, t + 1]
    nb = blank
    inc = C[np.arange(N)[:, None], np.arange(81)[None], nxt - 1]            # 다음 argmax 가 후보 안인가
    in_cand.append(float(inc[nb].mean()))
    one = (C.sum(-1) == 1) & nb
    if one.any(): naked_follow.append(float((nxt[one] - 1 == C[one].argmax(-1)).mean())); naked_n += int(one.sum())
    # hidden single: 유닛 안에서 값 v 의 후보 칸이 하나뿐
    hf = [];
    for u in UNITS:
        Cu = C[:, u]                                                        # [N,9cells,9vals]
        Cu = Cu & nb[:, u][:, :, None]
        cnt = Cu.sum(1)                                                     # [N,9vals]
        for v in range(9):
            hit = np.where(cnt[:, v] == 1)[0]
            if len(hit) == 0: continue
            cell = u[Cu[hit, :, v].argmax(1)]
            hf.append(nxt[hit, cell] - 1 == v)
    if hf: hf = np.concatenate(hf); hidden_follow.append(float(hf.mean())); hidden_n += len(hf)
res["rule"] = {"steps": [0, 1, 3, 7, 15, 31, 63, 95, 126], "next_argmax_in_candidates": in_cand,
               "naked_single_followed": naked_follow, "hidden_single_followed": hidden_follow,
               "naked_cases_total": naked_n, "hidden_cases_total": hidden_n}
# 최종 판의 제약 위반율 (라벨 불필요)
fin = AM[:, -1].astype(np.int64); fin[~blank] = I[~blank]
viol = np.zeros((N, 81), bool)
for t in range(81): viol[:, t] = (fin[:, pm[t]] == fin[:, t:t + 1]).any(1)
res["final_violation_rate_blank"] = float(viol[blank].mean()); res["AUC(wrong|violation)"] = auc(viol.astype(float), wrong)
res["wrong_cells_no_violation_frac"] = float((~viol)[wrong & blank].mean())
print(json.dumps(res, indent=1, ensure_ascii=False))
if args.out: json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
