"""변형 요인 분해: 숫자 치환만 / 공간(전치·밴드순열)만 / 둘 다. 여러 퍼즐에 걸어 풀림률과 변형 간 편차를 본다."""
import os, sys, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False)
K, SEGS = 32, 128
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=K, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); XT = z["test_inputs"].reshape(-1, 81).astype(int); YT = z["test_labels"].reshape(-1, 81).astype(int)
rng = np.random.default_rng(1)
def aug(g, dm, tr, rp, cp):
    g = g.reshape(9, 9); g = np.where(g > 0, dm[g], 0)
    if tr: g = g.T
    return g[rp][:, cp].reshape(81)
def band_perm():
    b = rng.permutation(3); return np.concatenate([b[i] * 3 + rng.permutation(3) for i in range(3)])
def params(mode):
    out = []
    for k in range(K):
        dm = np.concatenate([[0], rng.permutation(9) + 1]) if mode in ("digit", "both") else np.arange(10)
        tr = bool(rng.integers(2)) if mode in ("space", "both") else False
        rp, cp = (band_perm(), band_perm()) if mode in ("space", "both") else (np.arange(9), np.arange(9))
        out.append((dm, tr, rp, cp))
    out[0] = (np.arange(10), False, np.arange(9), np.arange(9)); return out
def run(pid, mode):
    ps = params(mode); Xa = np.stack([aug(XT[pid], *p) for p in ps]); Ya = np.stack([aug(YT[pid], *p) for p in ps])
    x = torch.from_numpy(Xa.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Ya.astype(np.int32) + 1).cuda().long()
    batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(K, dtype=torch.int32, device="cuda"))
    with torch.device("cuda"): carry = m.initial_carry(batch)
    first = np.full(K, -1)
    for si in range(SEGS):
        carry, o = m(carry, batch); ok = ((o["logits"].argmax(-1).cpu().numpy() - 1) == Ya).all(-1)
        first[(first < 0) & ok] = si + 1
    return first
for pid, label in [(155, "고정점형"), (2, "진동형"), (0, "해결형"), (3, "?"), (5, "?")]:
    print(f"\n=== 퍼즐 #{pid} ({label})   원본 결과: ", end="", flush=True)
    res = {mode: run(pid, mode) for mode in ("digit", "space", "both")}
    f0 = res["digit"][0]; print(f"{'풀림 seg '+str(f0) if f0>0 else '미해결'}")
    for mode, name in [("digit", "숫자 치환만"), ("space", "공간만(전치·밴드순열)"), ("both", "둘 다")]:
        f = res[mode]; n = int((f > 0).sum()); within16 = int(((f > 0) & (f <= 16)).sum())
        print(f"  {name:22s} 풀림 {n:2d}/{K}  (16세그 안 {within16:2d})   풀린 것들의 seg 중앙값 {int(np.median(f[f>0])) if n else '-'}", flush=True)
