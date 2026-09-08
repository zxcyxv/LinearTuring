"""v1(9/1@310527): seg1~16 은 학습된 ψ 로, seg17 부터 ψ=0 으로 바꿔서 계속."""
import os, sys, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False)
NB, SEG, SWITCH = 512, 128, 16

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
L = m.inner.layers[0]

z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
X = z["test_inputs"].reshape(-1, 81).astype(np.int32)[:NB]; Y = z["test_labels"].reshape(-1, 81).astype(np.int32)[:NB]
x = torch.from_numpy(X + 1).cuda(); y = torch.from_numpy(Y + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(NB, dtype=torch.int32, device="cuda"))
blank = (x == 1)

t0 = time.time()
with torch.device("cuda"): carry = m.initial_carry(batch)
prev = None; solved16 = None
print(f"switch at seg{SWITCH}  (seg1~{SWITCH} 학습된 ψ, 이후 ψ=0)")
print(f"{'seg':>5} {'acc':>8} {'exact':>6} {'churn':>8}  {'seg16해결 중 유지':>16}")
for si in range(SEG):
    if si == SWITCH:
        L.psi.data.zero_()
        print(f"      ---- ψ = 0 으로 전환 ----")
    carry, out = m(carry, batch)
    pred = out["logits"].argmax(-1)
    ok = (pred == y).all(-1)
    acc, ex = (pred == y).float().mean().item(), ok.sum().item()
    ch = 0.0 if prev is None else ((pred != prev) & blank).float().sum().item() / blank.sum().item()
    prev = pred.clone()
    if si + 1 == SWITCH: solved16 = ok.clone()
    keep = "-" if solved16 is None else f"{(ok & solved16).sum().item()}/{solved16.sum().item()}"
    if si + 1 in (1,2,4,8,12,14,15,16,17,18,19,20,22,24,28,32,48,64,96,128):
        print(f"{si+1:5d} {acc:8.4f} {ex:6d} {ch:8.5f}  {keep:>16}")
print(f"\n{time.time()-t0:.0f}s")
