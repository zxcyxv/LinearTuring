"""256 숫자 치환의 성패가 임베딩 기하의 어떤 특징과 상관되는가. fate_155.npz 의 solved 와 dms 사용."""
import os, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155
d = np.load(os.path.join(ROOT, "2026-09-05/results/json/fate_155.npz")); solved = d["solved"]; dms = d["dms"]
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
E = sd["inner.embed.weight"].float().numpy()                     # [11, 832]  토큰 = 숫자+1
En = E / np.linalg.norm(E, axis=1, keepdims=True); G = En @ En.T   # 코사인
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
print("숫자 임베딩 코사인 (토큰 2..10 = 숫자 1..9), 대각 제외 평균 {:.3f}, 최소 {:.3f}, 최대 {:.3f}".format(*(lambda g: (g[~np.eye(9, dtype=bool)].mean(), g[~np.eye(9, dtype=bool)].min(), g[~np.eye(9, dtype=bool)].max()))(G[2:, 2:])))
print("빈칸 토큰(1)과 숫자들의 코사인:", np.round(G[1, 2:], 2))
def auc(score, label):
    pos = score[label]; neg = score[~label]; return (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
giv = np.where(X0 > 0)[0]
feats = {}
for k, dm in enumerate(dms):
    Xk = np.where(X0 > 0, dm[X0], 0); Yk = dm[Y0]
    tok = lambda v: v + 1
    # (1) 주어진 동료 쌍(서로 다른 숫자)의 임베딩 코사인 평균 — 동료 주어진 칸들이 임베딩상 얼마나 가까운가
    pp = [(i, j) for i in giv for j in giv if i < j and peer[i, j]]
    feats.setdefault("동료 주어진 쌍 코사인 평균", []).append(np.mean([G[tok(Xk[i]), tok(Xk[j])] for i, j in pp]))
    # (2) 정답 격자 전체에서 동료 쌍 코사인 평균 (정답 배치의 '분리도')
    pa = np.where(peer)
    feats.setdefault("정답 격자 동료 쌍 코사인 평균", []).append(G[tok(Yk[pa[0]]), tok(Yk[pa[1]])].mean())
    # (3) 주어진 숫자 빈도 가중 노름: 많이 주어진 숫자의 임베딩 노름 평균
    cnt = np.bincount(Xk[giv], minlength=10)[1:]; nrm = np.linalg.norm(E[2:], axis=1)
    feats.setdefault("주어진 숫자 빈도가중 노름", []).append((cnt * nrm).sum() / cnt.sum())
    # (4) 각 빈칸에 대해 '동료 주어진 칸 임베딩 합' 이 정답 임베딩과 이루는 코사인 (첫 블록 추측의 정답 정렬도)
    sc = []
    for t in np.where(X0 == 0)[0]:
        nb = [j for j in giv if peer[t, j]]
        if nb: v = -En[[tok(Xk[j]) for j in nb]].sum(0); sc.append(v @ En[tok(Yk[t])] / (np.linalg.norm(v) + 1e-9))
    feats.setdefault("빈칸: −Σ동료임베딩 과 정답 임베딩 코사인", []).append(np.mean(sc))
print(f"\n256 치환, 풀림 {solved.sum()}개.  특징별 AUC (0.5 = 무관):")
for name, v in feats.items():
    v = np.array(v); a = auc(v, solved); print(f"  {name:34s} AUC {a:.2f}  (풀림 평균 {v[solved].mean():+.4f}, 실패 평균 {v[~solved].mean():+.4f})")
