"""FAIL run 32세그: 위반 쌍의 위치가 세그먼트마다 어떻게 옮겨 다니는가 (같은 자리 순환인지, 새 자리인지)."""
import os, importlib.util, numpy as np, torch
from collections import Counter
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; S = 32
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=S + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
x = torch.from_numpy(X0[None].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y0[None].astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(1, dtype=torch.int32, device="cuda"))
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
preds = []; orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    preds.append((I.w_cls(I.phi(I.boundary(L, hout))).argmax(-1)[0] - 1).cpu().numpy()); return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(S): carry, _ = m(carry, batch)
P = np.stack(preds); nb = len(P)
nm = lambda t: f"r{t//9}c{t%9}"
def pairs(p): return [(t, n) for t in range(81) for n in range(t + 1, 81) if peer[t, n] and p[t] == p[n]]
print("세그 끝 블록의 위반 쌍 (세그 8부터):")
for sg in range(7, S):
    V = pairs(P[16 * sg + 15]); print(f"  seg {sg+1:2d}: {len(V):2d}개  " + "  ".join(f"{nm(t)}-{nm(n)}({int(P[16*sg+15][t])})" for t, n in V))
# 블록 단위 통계 (세그 8 이후)
cnt = Counter(); cells = Counter(); nblk = 0
for b in range(16 * 7, nb):
    V = pairs(P[b]); nblk += 1
    for t, n in V: cnt[(t, n)] += 1; cells[t] += 1; cells[n] += 1
print(f"\n세그 8~32 ({nblk} 블록): 서로 다른 위반 쌍 {len(cnt)}종, 관여한 칸 {len(cells)}개")
print("가장 자주 위반한 쌍 (블록 수 / 전체):", [(f"{nm(t)}-{nm(n)}", k) for (t, n), k in cnt.most_common(8)])
print("가장 자주 관여한 칸:", [(nm(t), k) for t, k in cells.most_common(10)])
# 각 블록의 위반 칸 집합이 직전 블록과 얼마나 겹치나 (Jaccard) 와, 어떤 칸이 값을 바꾸나
J = []; changed = Counter()
for b in range(16 * 7 + 1, nb):
    A_ = {t for p in pairs(P[b - 1]) for t in p}; B_ = {t for p in pairs(P[b]) for t in p}
    J.append(len(A_ & B_) / max(len(A_ | B_), 1))
    for t in np.where(P[b] != P[b - 1])[0]: changed[t] += 1
print(f"블록 간 위반 칸 집합 Jaccard 평균 {np.mean(J):.2f}  (1 = 그대로, 0 = 완전히 다른 자리)")
print("값이 자주 바뀌는 칸 (블록 수):", [(nm(t), k) for t, k in changed.most_common(10)])
print("바뀌는 칸 수 총합:", len(changed), " — 그중 끝 틀린 칸:", sum(1 for t in changed if P[-1][t] != Y0[t]))
