"""학습된 sheaf 위의 디리클레 에너지 E_m^±(k) = Σ_{a≷0} |a_tn| ‖W_m h_t − W_m h_n‖² (피어/비피어 분리) — 라벨 없는 양.
측정만: ① 후반 블록에서 E 변화의 부호 일관성(단조성) — 완답 판 vs 비완답 판 ② 최종 E·|ΔE| 의 오답 AUC (퍼즐·칸 단위)
③ 제약 위반 0 인 오답 판(자기일관 오답) vs 완답 판을 E 가 가르는가.
사용: python sheaf_energy.py [--n 2048] [--out JSON]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--n", type=int, default=2048); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
core = m.inner.core; H = core.H
inp, lab, _ = load_test(args.n); N = len(inp); K = m.config.blocks_per_seg; TS = m.config.loops * K
PM = torch.tensor(peer_mask(), device="cuda")
Wsh = core.w_sh                                        # [H,dh,d]
# 퍼즐×블록×헤드×{+peer,+non,−peer,−non} 에너지, 칸 단위 최종 에너지
E = np.zeros((N, TS, H, 4), np.float32); Ecell = np.zeros((N, 81), np.float32); AM = np.zeros((N, 81), np.int8)
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N)); n = b.stop - b.start
    def hook(loop, blk, stage, h, a, i=i, n=n):
        if stage != "post_step": return
        t = loop * K + blk
        v = torch.einsum('btd,hcd->bhtc', h[:n].float(), Wsh.float())          # [n,H,81,dh]
        sq = (v * v).sum(-1)                                                     # [n,H,81]
        D = sq[..., :, None] + sq[..., None, :] - 2 * torch.einsum('bhtc,bhnc->bhtn', v, v)   # ‖v_t−v_n‖² [n,H,81,81]
        A = a[:n].float(); ap_, an_ = A.clamp(min=0), (-A).clamp(min=0)
        for j, (w, msk) in enumerate([(ap_, PM), (ap_, ~PM), (an_, PM), (an_, ~PM)]):
            E[i:i + n, t, :, j] = (w * D * msk).sum((-1, -2)).cpu().numpy()
        if t == TS - 1:
            Ecell[i:i + n] = (A * D).sum((1, 3)).cpu().numpy()                    # 부호 포함 칸별 합 (헤드 합)
            AM[i:i + n] = (logits(m, h)[:n, :, 2:11].argmax(-1) + 1).to(torch.int8).cpu().numpy()
    rollout(m, make_batch(inp[b], lab[b]), hook=hook)
I = inp.cpu().numpy() - 1; LB = lab.cpu().numpy() - 1; blank = I == 0
ok = AM == LB; exact = ok.all(1)
pm = peer_mask(); fin = AM.astype(np.int64); fin[~blank] = I[~blank]
viol = np.stack([(fin[:, pm[t]] == fin[:, t:t + 1]).any(1) for t in range(81)], 1)
selfcons_wrong = ~exact & ~viol[blank.any(1)][:0].any() if False else (~exact) & ~(viol & blank).any(1)
Etot = E[..., 0] + E[..., 1] - E[..., 2] - E[..., 3]                    # 부호 포함 총 에너지 [N,TS,H]
Es = Etot.sum(-1)                                                       # 헤드 합 [N,TS]
def auc(s, y):
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))
res = {"n": N, "exact": int(exact.sum()), "selfconsistent_wrong": int(selfcons_wrong.sum())}
# ① 단조성: 후반 64 블록의 ΔE 부호 일관도 = |Σ sign(ΔE)| / 64
dE = np.diff(Es[:, TS - 65:], axis=1); mono = np.abs(np.sign(dE).sum(1)) / dE.shape[1]
res["monotonicity_last64"] = {"exact_mean": float(mono[exact].mean()), "wrong_mean": float(mono[~exact].mean()),
                              "exact_frac_dE_neg": float((dE[exact] < 0).mean()), "wrong_frac_dE_neg": float((dE[~exact] < 0).mean())}
# 궤적 요약 (완답/비완답 평균), 성분별
for nm, msk in (("exact", exact), ("wrong", ~exact)):
    res[f"E_traj_{nm}"] = {"steps": [0, 7, 15, 31, 63, 95, 127],
                           "total": [float(Es[msk, k].mean()) for k in [0, 7, 15, 31, 63, 95, 127]],
                           "+peer": [float(E[msk, k, :, 0].sum(-1).mean()) for k in [0, 7, 15, 31, 63, 95, 127]],
                           "+non": [float(E[msk, k, :, 1].sum(-1).mean()) for k in [0, 7, 15, 31, 63, 95, 127]],
                           "-peer": [float(E[msk, k, :, 2].sum(-1).mean()) for k in [0, 7, 15, 31, 63, 95, 127]],
                           "-non": [float(E[msk, k, :, 3].sum(-1).mean()) for k in [0, 7, 15, 31, 63, 95, 127]]}
# ② AUC: 퍼즐 단위 (오답 판 = 양성), 칸 단위
res["AUC_puzzle_wrong"] = {"final_E": auc(Es[:, -1], ~exact), "final_|dE|": auc(np.abs(dE[:, -8:]).mean(1), ~exact),
                           "final_E-peer": auc(E[:, -1, :, 2].sum(-1), ~exact), "final_E+non": auc(E[:, -1, :, 1].sum(-1), ~exact)}
res["AUC_cell_wrong"] = {"final_Ecell": auc(Ecell[blank], (~ok)[blank]), "final_|Ecell|": auc(np.abs(Ecell)[blank], (~ok)[blank])}
res["AUC_per_head_puzzle"] = {h: auc(Etot[:, -1, h], ~exact) for h in range(H)}
# ③ 자기일관 오답 vs 완답
if selfcons_wrong.any():
    sel = exact | selfcons_wrong; y = selfcons_wrong[sel]
    res["AUC_selfconsistent_wrong_vs_exact"] = {"final_E": auc(Es[sel, -1], y), "final_|dE|": auc(np.abs(dE[sel, -8:]).mean(1), y),
                                                "per_head": {h: auc(Etot[sel, -1, h], y) for h in range(H)}}
print(json.dumps(res, indent=1))
if args.out: json.dump(res, open(args.out, "w"), indent=1)
