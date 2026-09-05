"""원본 #155 에서 숫자 두 개만 맞바꾼 36가지 전치 + 원본. 어느 전치가 성패를 뒤집나."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; SEGS = 64
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
pairs = [(a, b) for a in range(1, 10) for b in range(a + 1, 10)]; K = len(pairs) + 1
cfg = dict(ck["cfg"]); cfg.update(batch_size=K, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
dms = [np.arange(10)]
for a, b in pairs: dm = np.arange(10); dm[a], dm[b] = b, a; dms.append(dm)
Xa = np.stack([np.where(X0 > 0, dm[X0], 0) for dm in dms]); Ya = np.stack([dm[Y0] for dm in dms])
x = torch.from_numpy(Xa.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Ya.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(K, dtype=torch.int32, device="cuda"))
with torch.device("cuda"): carry = m.initial_carry(batch)
first = np.full(K, -1); t0 = time.time()
for si in range(SEGS):
    carry, o = m(carry, batch); ok = ((o["logits"].argmax(-1).cpu().numpy() - 1) == Ya).all(-1); first[(first < 0) & ok] = si + 1
    if (si + 1) % 16 == 0: print(f"  seg {si+1}/{SEGS} {time.time()-t0:.0f}s 풀림 {(first>0).sum()}/{K}", flush=True)
cnt = np.bincount(X0[X0 > 0], minlength=10)
print(f"\n원본: {'풀림 seg '+str(first[0]) if first[0]>0 else '미해결'}")
print("전치 (a↔b): 풀림 세그 (– = 미해결).  괄호 = 주어진 칸에서 a, b 의 빈도")
grid = {}
for (a, b), f in zip(pairs, first[1:]): grid[(a, b)] = f
print("     " + "".join(f"{b:5d}" for b in range(2, 10)))
for a in range(1, 9):
    print(f"  {a}: " + "".join(f"{(str(grid[(a,b)]) if grid[(a,b)]>0 else '–'):>5}" if b > a else "     " for b in range(2, 10)) + f"   (빈도 {cnt[a]})")
print(f"  9 빈도 {cnt[9]}")
print(f"\n36개 전치 중 풀림 {(first[1:]>0).sum()}개.  각 숫자가 관여한 전치의 풀림 수:", {dd: int(sum(1 for (a, b), f in grid.items() if f > 0 and dd in (a, b))) for dd in range(1, 10)})
