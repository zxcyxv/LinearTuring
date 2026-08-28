"""한 칸 기제: 정답 격자에서 --k 칸만 비우고 실행. 빈칸 t 에 대해
  (1) 정답률  (2) 헤드별 |a_tn| 질량: 피어(행/열/박스) vs 비피어, 피어 간 변동계수
  (3) 출처별 로짓 분해 C_tn = w_cls·(a_tn W_mᵀW_m h_n) (마지막 블록, 헤드 합) — 피어의 숫자 v_n 자리 vs 다른 자리
  (4) 최종 상태 h_t 를 저장 (이식 실험용)
사용: python one_cell.py [--k 1] [--n 2048] [--out NPZ]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--k", type=int, default=1); ap.add_argument("--n", type=int, default=2048); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_grad_enabled(False); rng = np.random.default_rng(args.seed)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
core = m.inner.core; H = core.H; K = m.config.blocks_per_seg; TS = m.config.loops * K
_, lab, _ = load_test(args.n); N = len(lab); LB = lab.cpu().numpy()          # 토큰 2..10
pm = peer_mask(); r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
# 입력: 정답에서 k 칸 비움
X = LB.copy(); T = np.zeros((N, args.k), int)
for i in range(N):
    T[i] = rng.choice(81, args.k, replace=False); X[i, T[i]] = 1
x = torch.tensor(X, dtype=torch.int32, device="cuda"); t0 = torch.tensor(T[:, 0], device="cuda")
Wsh = core.w_sh.float(); Wc = core.w_cls.weight.float()[2:11]                 # [9,d]
A_t = np.zeros((N, H, 81), np.float32); C = np.zeros((N, 81, 9), np.float32); Hs = np.zeros((N, core.d), np.float32)
pred = np.zeros(N, np.int64); AM = np.zeros((N, TS), np.int8)                 # 첫 빈칸의 argmax 궤적
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N)); n = b.stop - b.start; tt = t0[b]
    def hook(loop, blk, stage, h, a, i=i, n=n, tt=tt):
        if stage != "post_step": return
        s = loop * K + blk; h = h[:n].float(); a = a[:n].float(); ar = torch.arange(n, device="cuda")
        AM[i:i + n, s] = (logits(m, h)[ar, tt, 2:11].argmax(-1) + 1).to(torch.int8).cpu().numpy()
        if s == TS - 1:
            at = a[ar, :, tt]                                                # [n,H,81]  t 가 받는 간선
            A_t[i:i + n] = at.cpu().numpy(); Hs[i:i + n] = h[ar, tt].cpu().numpy()
            v = torch.einsum('bnd,hcd->bhnc', h, Wsh)                        # [n,H,81,dh]
            msg = torch.einsum('bhn,bhnc,hcd->bnd', at, v, Wsh)              # 출처 n 별 메시지 (헤드 합) [n,81,d]
            C[i:i + n] = torch.einsum('bnd,vd->bnv', msg, Wc).cpu().numpy()  # 판독 [n,81,9]
            pred[i:i + n] = (logits(m, h)[ar, tt, 2:11].argmax(-1) + 2).cpu().numpy()
    rollout(m, make_batch(x[b], x[b]), hook=hook)
ok = pred == LB[np.arange(N), T[:, 0]]
res = {"k": args.k, "n": N, "acc_first_blank": float(ok.mean()),
       "commit_step_median": float(np.median([int(np.argmax(np.cumprod((AM[i, ::-1] == AM[i, -1])))) for i in range(N)]))}
# (2) 질량 분포
peer = pm[T[:, 0]]                                                            # [N,81]
rowm = (r[None] == r[T[:, 0]][:, None]) & ~np.eye(81, dtype=bool)[T[:, 0]]; colm = (c[None] == c[T[:, 0]][:, None]) & ~np.eye(81, dtype=bool)[T[:, 0]]
boxm = (bx[None] == bx[T[:, 0]][:, None]) & ~np.eye(81, dtype=bool)[T[:, 0]]
aa = np.abs(A_t); tot = aa.sum(-1) + 1e-9
res["mass"] = {"peer_frac_per_head": [float((aa[:, h] * peer).sum(-1).mean() / tot[:, h].mean()) for h in range(H)],
               "row/col/box_mean_|a|_per_head": [[float(aa[:, h][msk].mean()) for msk in (rowm, colm, boxm)] for h in range(H)],
               "nonpeer_mean_|a|_per_head": [float(aa[:, h][~peer & ~np.eye(81, dtype=bool)[T[:, 0]]].mean()) for h in range(H)],
               "peer_sign_mean_per_head": [float(A_t[:, h][peer].mean()) for h in range(H)],
               "peer_CV_per_head": [float((aa[:, h] * peer).std(-1).mean() / ((aa[:, h] * peer).sum(-1) / 20).mean()) for h in range(H)]}
# (3) 출처별 로짓 분해: 피어 n 의 숫자 v_n 자리 vs 다른 8자리, 그리고 정답 숫자 자리
vn = LB - 2                                                                   # [N,81] 0..8
ans = (LB[np.arange(N), T[:, 0]] - 2)
Cp = C[peer]                                                                  # [N*20, 9]
vp = vn[peer]
own = Cp[np.arange(len(Cp)), vp]; others = (Cp.sum(-1) - own) / 8
ans_p = np.repeat(ans, 20); at_ans = Cp[np.arange(len(Cp)), ans_p]
Cn = C[~peer & ~np.eye(81, dtype=bool)[T[:, 0]]]
res["source_logits"] = {"peer_at_own_digit": float(own.mean()), "peer_at_other_digits": float(others.mean()), "peer_at_answer_digit": float(at_ans.mean()),
                        "peer_own_minus_others_frac_negative": float((own < others).mean()),
                        "nonpeer_abs_mean": float(np.abs(Cn).mean()), "peer_abs_mean": float(np.abs(Cp).mean()),
                        "total_peer_msg_at_answer_vs_max_other": [float(C[np.arange(N)[:, None], np.where(peer)[1].reshape(N, 20)].sum(1)[np.arange(N), ans].mean()),
                                                                  float(np.delete(C[np.arange(N)[:, None], np.where(peer)[1].reshape(N, 20)].sum(1), 0, 0).max(-1).mean() if False else 0)]}
# 정답 자리 vs 최대 오답 자리 (피어 메시지 합)
S = np.stack([C[i, peer[i]].sum(0) for i in range(N)])                        # [N,9]
sa = S[np.arange(N), ans]; So = S.copy(); So[np.arange(N), ans] = -1e9
res["source_logits"]["peer_sum_at_answer_mean"] = float(sa.mean()); res["source_logits"]["peer_sum_max_other_mean"] = float(So.max(-1).mean())
res["source_logits"]["peer_sum_answer_is_max_frac"] = float((sa > So.max(-1)).mean())
print(json.dumps(res, indent=1))
if args.out: np.savez(args.out, h_t=Hs, t=T[:, 0], pred=pred, ok=ok, A_t=A_t, C=C, res=json.dumps(res))
