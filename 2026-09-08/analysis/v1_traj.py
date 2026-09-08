"""v1(9/1@310527) base 를 512퍼즐 x 128세그 돌리고 세그먼트별 예측 전체를 기록."""
import os, sys, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False)
NB, SEG = 512, 128
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "2026-09-06/analysis/train_0901.py"))
tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
sys.path.insert(0, os.path.join(ROOT, "checkpoints"))
from ckpt_npz import load as npz_load
cfg = dict(tk.CFG); cfg.update(batch_size=NB, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1)
m = tk.LT(cfg).cuda().eval()
sd, meta = npz_load(os.path.join(ROOT, "checkpoints/0901_R1B8_min_faith_step310527.npz"), which="ema")
PER = {"wc_raw","psi","theta","alpha_raw","w_sh","eta_raw","lam_raw","gain_raw","beta","b_gate_up","b_down"}
def remap(k):
    k = k.replace("_orig_mod.", ""); k = k[len("model."):] if k.startswith("model.") else k
    tail = k[len("inner."):]
    if tail == "inj_gate": return "inner.embed_scale"
    return f"inner.layers.0.{tail}" if tail.split(".")[0] in PER else k
m.load_state_dict({remap(k): v for k, v in sd.items()}, strict=True)

z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
X = z["test_inputs"].reshape(-1, 81).astype(np.int32)[:NB]; Y = z["test_labels"].reshape(-1, 81).astype(np.int32)[:NB]
x = torch.from_numpy(X + 1).cuda(); y = torch.from_numpy(Y + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(NB, dtype=torch.int32, device="cuda"))
t0 = time.time()
with torch.device("cuda"): carry = m.initial_carry(batch)
P = np.zeros((SEG, NB, 81), dtype=np.int8)
for si in range(SEG):
    carry, out = m(carry, batch)
    P[si] = out["logits"].argmax(-1).cpu().numpy().astype(np.int8)
np.savez_compressed("/tmp/claude-0/-workspace-LinearTuring/8092b54b-030b-4f2c-b7a7-8107892a6681/scratchpad/v1_traj.npz",
                    P=P, X=X, Y=Y)
ok = (P == (Y + 1)[None]).all(-1)                       # [SEG,NB]
ex = ok.sum(1)
print(f"{time.time()-t0:.0f}s   exact by seg: " + " ".join(f"s{k+1}:{ex[k]}" for k in [0,3,7,15,31,63,95,127]))
first = np.where(ok.any(0), ok.argmax(0) + 1, -1)       # 처음 풀린 세그 (안 풀리면 -1)
stable = ok[-1]                                          # seg128 에서 풀려 있나
import collections
print("처음 풀린 세그 분포:", collections.Counter(np.clip(first, -1, 200) // 16 * 16).most_common())
print(f"seg128 해결 {stable.sum()}/{NB},  한 번도 못 푼 퍼즐 {(first==-1).sum()}")
