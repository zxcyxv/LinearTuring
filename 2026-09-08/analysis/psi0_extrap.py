"""v1(9/1 R1B8_min_faith@310527) 에서 ψ 만 0 으로 덮어쓰고 세그먼트 외삽."""
import os, sys, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"
torch.set_grad_enabled(False)
NB = SEG = None
NB, SEG = 512, 128

s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "2026-09-06/analysis/train_0901.py"))
tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
sys.path.insert(0, os.path.join(ROOT, "checkpoints"))
from ckpt_npz import load as npz_load

cfg = {k: v for k, v in tk.CFG.items()}
cfg.update(batch_size=NB, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1)
m = tk.LT(cfg).cuda().eval()

sd, meta = npz_load(os.path.join(ROOT, "checkpoints/0901_R1B8_min_faith_step310527.npz"), which="ema")
PER_LAYER = {"wc_raw","psi","theta","alpha_raw","w_sh","eta_raw","lam_raw","gain_raw","beta","b_gate_up","b_down"}
def remap(k):
    k = k.replace("_orig_mod.", "")
    k = k[len("model."):] if k.startswith("model.") else k
    assert k.startswith("inner."), k
    tail = k[len("inner."):]
    if tail == "inj_gate": return "inner.embed_scale"
    return f"inner.layers.0.{tail}" if tail.split(".")[0] in PER_LAYER else k
sd = {remap(k): v for k, v in sd.items()}
missing, unexpected = m.load_state_dict(sd, strict=False)
print(f"step={meta['step']}  missing={list(missing)}  unexpected={list(unexpected)}", flush=True)

# ---- ψ = 0
L = m.inner.layers[0]
print(f"ψ 덮어쓰기 전: mean cos ψ = {torch.cos(L.psi).mean():+.4f}", flush=True)
L.psi.data.zero_()
print(f"ψ 덮어쓰기 후: mean cos ψ = {torch.cos(L.psi).mean():+.4f}", flush=True)

# ---- 데이터
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
X = z["test_inputs"].reshape(-1, 81).astype(np.int32)[:NB]
Y = z["test_labels"].reshape(-1, 81).astype(np.int32)[:NB]
x = torch.from_numpy(X + 1).cuda(); y = torch.from_numpy(Y + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(NB, dtype=torch.int32, device="cuda"))
blank = (x == 1)                                   # 빈칸(비단서)

t0 = time.time()
with torch.device("cuda"): carry = m.initial_carry(batch)
prev = None
print(f"\n{'seg':>5} {'acc':>8} {'exact':>6} {'exact%':>8} {'churn':>8}")
for si in range(SEG):
    carry, out = m(carry, batch)
    pred = out["logits"].argmax(-1)
    acc = (pred == y).float().mean().item()
    ex  = (pred == y).all(-1).sum().item()
    ch  = 0.0 if prev is None else ((pred != prev) & blank).float().sum().item() / blank.sum().item()
    prev = pred.clone()
    if si + 1 in (1,2,4,8,12,16,20,24,32,48,64,96,128):
        print(f"{si+1:5d} {acc:8.4f} {ex:6d} {100*ex/NB:8.2f} {ch:8.5f}" + ("  <-train" if si == 15 else ""), flush=True)
print(f"\n{time.time()-t0:.0f}s")
