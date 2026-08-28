"""모델의 장 {a_tn, d_t, ‖h_t‖} 에 '제약 위반' 이 직접 찍혀 있는가 — 학습된 프로브 없이 스칼라 그대로 읽는다.
  간선: 피어 쌍의 a_tn^(m) 을 (같은 argmax / 다른 argmax) × (둘 다 정답 / 오답 포함) 으로 분할
  칸:   d_t^(m), ‖h_t‖, ‖W_m h_t‖ 를 (위반 칸 / 무위반 오답 칸 / 정답 칸) 으로 분할 + 오답 AUC
  시간: 최종 충돌 쌍 간선의 블록별 궤적 (충돌이 생긴 시점 정렬)
사용: python field_violation.py [--n 1024] [--out JSON]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--n", type=int, default=1024); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
core = m.inner.core; H = core.H
inp, lab, _ = load_test(args.n); N = len(inp); K = m.config.blocks_per_seg; TS = m.config.loops * K
pm = peer_mask()
A_fin = np.zeros((N, H, 81, 81), np.float16); D = np.zeros((N, TS, H, 81), np.float16)
HN = np.zeros((N, TS, 81), np.float16); WN = np.zeros((N, H, 81), np.float16)
AM = np.zeros((N, TS, 81), np.int8); Apeer_t = np.zeros((N, TS, H, 81, 81), np.float16) if False else None
# 시간 궤적용: 블록마다 피어 간선만 저장 (81×20)
peer_idx = np.stack([np.where(pm[t])[0] for t in range(81)])            # [81,20]
AP = np.zeros((N, TS, H, 81, 20), np.float16)
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N)); n = b.stop - b.start
    def hook(loop, blk, stage, h, a, i=i, n=n):
        if stage != "post_step": return
        t = loop * K + blk; a = a[:n].float(); h = h[:n].float()
        D[i:i + n, t] = a.sum(-1).cpu().numpy(); HN[i:i + n, t] = h.norm(dim=-1).cpu().numpy()
        AM[i:i + n, t] = (logits(m, h)[:, :, 2:11].argmax(-1) + 1).to(torch.int8).cpu().numpy()
        AP[i:i + n, t] = a[:, :, torch.arange(81)[:, None], torch.tensor(peer_idx)].cpu().numpy()
        if t == TS - 1:
            A_fin[i:i + n] = a.cpu().numpy()
            WN[i:i + n] = torch.einsum('btd,hcd->bhtc', h, core.w_sh.float()).norm(dim=-1).cpu().numpy()
    rollout(m, make_batch(inp[b], lab[b]), hook=hook)
I = inp.cpu().numpy() - 1; LB = lab.cpu().numpy() - 1; blank = I == 0
fin = AM[:, -1].astype(np.int64); fin[~blank] = I[~blank]; ok = fin == LB
viol = np.stack([(fin[:, pm[t]] == fin[:, t:t + 1]).any(1) for t in range(81)], 1)
same = fin[:, :, None] == fin[:, None, :]                                  # [N,81,81]
P = np.broadcast_to(pm, same.shape)
res = {"n": N, "cell_acc_blank": float(ok[blank].mean()), "viol_cells": int((viol & blank).sum()), "wrong_cells": int((~ok & blank).sum())}
def auc(s, y):
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))
# ── 간선 ──
bothok = ok[:, :, None] & ok[:, None, :]
edges = {}
for nm, msk in (("peer_same_val", P & same), ("peer_diff_val", P & ~same), ("peer_same_bothok(=불가능)", P & same & bothok),
                ("peer_diff_bothok", P & ~same & bothok), ("peer_same_anywrong", P & same & ~bothok), ("peer_diff_anywrong", P & ~same & ~bothok),
                ("nonpeer_same_val", ~P & same & ~np.eye(81, dtype=bool)), ("nonpeer_diff_val", ~P & ~same)):
    edges[nm] = {"count": int(msk.sum()), "mean_a_per_head": [float(A_fin[:, h][msk].astype(np.float32).mean()) for h in range(H)]}
res["edges_final"] = edges
# 간선 단위 AUC: 피어 간선에서 '같은 값' 을 a 로 판별 (헤드별, 부호 방향 = 더 음수)
pe = P & ~np.eye(81, dtype=bool)
res["AUC_edge_sameval_by_-a"] = [auc(-A_fin[:, h][pe].astype(np.float32), same[pe]) for h in range(H)]
# ── 칸 ──
cls = {"viol": viol & blank, "clean_wrong": ~viol & ~ok & blank, "right": ok & blank}
cells = {}
for nm, msk in cls.items():
    cells[nm] = {"count": int(msk.sum()), "d_t_per_head": [float(D[:, -1, h][msk].astype(np.float32).mean()) for h in range(H)],
                 "h_norm": float(HN[:, -1][msk].astype(np.float32).mean()), "Wh_norm_per_head": [float(WN[:, h][msk].astype(np.float32).mean()) for h in range(H)]}
res["cells_final"] = cells
wr = (~ok)[blank]; vi = viol[blank]
res["AUC_cell"] = {"wrong|d_t_head": [auc(D[:, -1, h][blank].astype(np.float32), wr) for h in range(H)],
                   "wrong|h_norm": auc(HN[:, -1][blank].astype(np.float32), wr),
                   "viol|d_t_head": [auc(D[:, -1, h][blank].astype(np.float32), vi) for h in range(H)],
                   "viol|h_norm": auc(HN[:, -1][blank].astype(np.float32), vi),
                   "wrong|Wh_norm_head": [auc(WN[:, h][blank].astype(np.float32), wr) for h in range(H)]}
# ── 시간: 최종 충돌 쌍의 간선 궤적 vs 최종 비충돌 피어 쌍 (같은 칸들 기준) ──
conf = P & same & ~np.eye(81, dtype=bool)
traj = {"steps": [0, 3, 7, 15, 31, 63, 95, 127]}
for nm, msk in (("conflict_pair", conf), ("nonconflict_peer_pair", P & ~same)):
    # AP 는 [N,TS,H,81,20] 피어 순서; msk 를 그 순서로 변환
    mk = np.stack([msk[:, t, peer_idx[t]] for t in range(81)], 1)         # [N,81,20]
    traj[nm] = {f"head{h}": [float(AP[:, k, h][mk].astype(np.float32).mean()) for k in traj["steps"]] for h in range(H)}
res["edge_traj"] = traj
# 충돌 쌍이 '생긴 뒤' 뒤집히는가: 마지막 32 블록 동안 같은 값인 피어 쌍 중, 그 전에 같은 값이 된 시점부터 뒤집힘까지
print(json.dumps(res, indent=1, ensure_ascii=False))
if args.out: json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
