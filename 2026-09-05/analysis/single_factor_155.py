"""단일 요인 통제: 치환에서 한 항목(d→e)만 고정하고 나머지를 무작위. 조건별 64개, 64세그."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; SEGS = 64; R = 64
COND = [("4→6", 4, 6), ("4→5", 4, 5), ("9→2", 9, 2), ("1→6", 1, 6), ("무조건(대조)", 0, 0)]; K = R * len(COND)
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=K, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
rng = np.random.default_rng(11); dms = []
for name, d_, e_ in COND:
    for _ in range(R):
        while True:
            dm = np.concatenate([[0], rng.permutation(9) + 1])
            if d_ == 0 or dm[d_] == e_: break
        dms.append(dm)
Xa = np.stack([np.where(X0 > 0, dm[X0], 0) for dm in dms]); Ya = np.stack([dm[Y0] for dm in dms])
x = torch.from_numpy(Xa.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Ya.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(K, dtype=torch.int32, device="cuda"))
with torch.device("cuda"): carry = m.initial_carry(batch)
first = np.full(K, -1); t0 = time.time()
for si in range(SEGS):
    carry, o = m(carry, batch); ok = ((o["logits"].argmax(-1).cpu().numpy() - 1) == Ya).all(-1); first[(first < 0) & ok] = si + 1
    if (si + 1) % 16 == 0: print(f"  seg {si+1}/{SEGS} {time.time()-t0:.0f}s", flush=True)
print(f"\n{'조건':>12} | {'풀림/64':>8} {'%':>5} | 16세그 안 풀림")
for i, (name, _, _) in enumerate(COND):
    f = first[i * R:(i + 1) * R]; print(f"{name:>12} | {(f>0).sum():5d}/64 {100*(f>0).mean():5.0f} | {((f>0)&(f<=16)).sum():3d}")
