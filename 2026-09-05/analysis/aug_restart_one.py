"""퍼즐 하나를 대칭 변형(숫자 치환 × 전치 × 행/열 밴드 순열)으로 K개 궤적을 만들어 돌린다.
하나라도 풀리면 '출발점 운', 전부 실패면 '구조적 한계'. 답은 역변환해 원 퍼즐 정답과 비교."""
import os, sys, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False)
PID = int(sys.argv[1]) if len(sys.argv) > 1 else 155; K = int(sys.argv[2]) if len(sys.argv) > 2 else 32; SEGS = int(sys.argv[3]) if len(sys.argv) > 3 else 256
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=K, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
X0 = z["test_inputs"][PID].astype(int); Y0 = z["test_labels"][PID].astype(int)
rng = np.random.default_rng(0)
def aug(g, dm, tr, rp, cp):
    g = g.reshape(9, 9); g = np.where(g > 0, dm[g], 0)
    if tr: g = g.T
    return g[rp][:, cp].reshape(81)
def band_perm():  # 밴드 순열 × 밴드 안 행 순열
    b = rng.permutation(3); return np.concatenate([b[i] * 3 + rng.permutation(3) for i in range(3)])
params = [(np.concatenate([[0], rng.permutation(9) + 1]), bool(rng.integers(2)), band_perm(), band_perm()) for _ in range(K)]
params[0] = (np.arange(10), False, np.arange(9), np.arange(9))            # 0번 = 원본
Xa = np.stack([aug(X0, *p) for p in params]); Ya = np.stack([aug(Y0, *p) for p in params])
x = torch.from_numpy(Xa.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Ya.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(K, dtype=torch.int32, device="cuda"))
with torch.device("cuda"): carry = m.initial_carry(batch)
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
first = [None] * K; t0 = time.time(); hist = []
for si in range(SEGS):
    carry, o = m(carry, batch); p = o["logits"].argmax(-1).cpu().numpy() - 1
    ok = (p == Ya).all(-1)
    for k in range(K):
        if ok[k] and first[k] is None: first[k] = si + 1
    hist.append(p)
    if (si + 1) % 32 == 0: print(f"  seg {si+1}/{SEGS}  {time.time()-t0:.0f}s  풀린 변형 {int(sum(f is not None for f in first))}/{K}", flush=True)
P = np.stack(hist); final = P[-1]
wrong = (final != Ya).sum(-1); viol = np.array([((final[k][:, None] == final[k][None]) & peer).sum() // 2 for k in range(K)])
osc = (np.diff(P[-32:], axis=0) != 0).any(0).sum(-1)
print(f"\n퍼즐 #{PID}: 변형 {K}개 중 풀림 {int(sum(f is not None for f in first))}개")
print(f"{'변형':>4} {'전치':>4} {'풀림seg':>7} {'끝 틀림':>7} {'위반쌍':>6} {'진동칸':>6}")
for k in range(K):
    print(f"{k:4d} {'T' if params[k][1] else '-':>4} {str(first[k]) if first[k] else '-':>7} {int(wrong[k]):7d} {int(viol[k]):6d} {int(osc[k]):6d}")
print(f"\n끝 틀린 칸 분포: 중앙값 {int(np.median(wrong))}, 최소 {int(wrong.min())}, 최대 {int(wrong.max())}   위반쌍 중앙값 {int(np.median(viol))}   진동 없는 변형 {int((osc==0).sum())}개")
