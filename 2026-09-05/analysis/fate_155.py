"""#155 숫자 치환 256개: 성패가 언제, 어느 칸에서, 임베딩의 무엇으로 결정되는가."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; K = 256; SEGS = 64
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=K, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
rng = np.random.default_rng(7); dms = [np.concatenate([[0], rng.permutation(9) + 1]) for _ in range(K)]; dms[0] = np.arange(10)
Xa = np.stack([np.where(X0 > 0, dm[X0], 0) for dm in dms]); Ya = np.stack([dm[Y0] for dm in dms])
inv = np.stack([np.argsort(dm) for dm in dms])                                   # 역치환
x = torch.from_numpy(Xa.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Ya.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(K, dtype=torch.int32, device="cuda"))
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
preds = []; orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    preds.append((I.w_cls(I.phi(I.boundary(L, hout))).argmax(-1) - 1).cpu().numpy()); return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
t0 = time.time()
for si in range(SEGS):
    carry, _ = m(carry, batch)
    if (si + 1) % 16 == 0: print(f"  seg {si+1}/{SEGS} {time.time()-t0:.0f}s  풀림 {int((preds[-1] == Ya).all(-1).sum())}/{K}", flush=True)
P = np.stack(preds)                                                              # [blk, K, 81] (치환된 숫자)
Pc = np.stack([inv[k][np.clip(P[:, k], 0, 9)] for k in range(K)], 1)             # 원래 숫자로 되돌림 [blk,K,81]
solved = (P[-1] == Ya).all(-1); first = np.array([next((b for b in range(len(P)) if (P[b, k] == Ya[k]).all()), -1) for k in range(K)])
print(f"\n풀림 {solved.sum()}/{K} ({100*solved.mean():.0f}%)  풀린 것의 첫 완답 블록 중앙값 {np.median(first[solved]):.0f} (세그 {np.median(first[solved])/16+1:.1f})")
wrong = (Pc != Y0[None, None]).sum(-1)                                            # [blk,K]
viol = np.array([[((P[b, k][:, None] == P[b, k][None]) & peer).sum() // 2 for k in range(K)] for b in range(0, len(P))])
def auc(score, label):
    pos = score[label]; neg = score[~label]
    return (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
print("\n[언제 결정되나] 블록 b 의 상태로 최종 성패를 예측하는 AUC (0.5 = 정보 없음, 1 = 완전 결정)")
print(f"{'block':>5} {'seg':>4} | {'틀린칸수 AUC':>11} {'위반수 AUC':>10} | {'풀린쪽 평균 틀림/위반':>20} {'실패쪽 평균 틀림/위반':>20}")
for b in [0, 1, 2, 3, 4, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127]:
    print(f"{b:5d} {b//16+1:4d} | {auc(-wrong[b], solved):11.2f} {auc(-viol[b], solved):10.2f} | {wrong[b][solved].mean():8.1f} / {viol[b][solved].mean():5.1f}      {wrong[b][~solved].mean():8.1f} / {viol[b][~solved].mean():5.1f}")
print("\n[어느 칸이 가르나] 블록 b 에서 '그 칸이 정답값인가' 가 성패와 가장 강하게 연관된 칸 (원래 숫자 기준)")
blank = X0 == 0
for b in [3, 7, 15]:
    corr = (Pc[b] == Y0[None])                                                    # [K,81]
    stats = []
    for t in np.where(blank)[0]:
        p_s = corr[solved, t].mean(); p_f = corr[~solved, t].mean(); stats.append((p_s - p_f, t, p_s, p_f))
    stats.sort(reverse=True)
    print(f"  block {b:2d}: " + "  ".join(f"r{t//9}c{t%9}(정답{Y0[t]}: 풀림{100*ps:.0f}% vs 실패{100*pf:.0f}%)" for d, t, ps, pf in stats[:5]))
    print(f"           반대 방향(맞으면 오히려 실패): " + "  ".join(f"r{t//9}c{t%9}(풀림{100*ps:.0f}% vs 실패{100*pf:.0f}%)" for d, t, ps, pf in stats[-3:]))
# 가짜 해와의 거리: 실패한 변형들이 같은 가짜 해로 가나
Ff = Pc[-1][~solved]; ref = Pc[-1][0] if not solved[0] else Ff[0]
dist = (Ff != ref[None]).sum(-1); print(f"\n[실패 변형의 끝 격자] 원본 가짜 해와 다른 칸 수: 중앙값 {np.median(dist):.0f}, 최소 {dist.min()}, 0인 것 {int((dist==0).sum())}개 / {len(dist)}")
np.savez_compressed(os.path.join(ROOT, "2026-09-05/results/json/fate_155.npz"), Pc=Pc[:16].astype(np.int8), solved=solved, first=first, dms=np.stack(dms), wrong=wrong, viol=viol)
