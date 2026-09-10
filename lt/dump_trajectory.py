"""체크포인트(.pt/.npz) → 512 퍼즐 × seg128 예측 P 를 npz 로 저장 (state_an / conv_an 입력)."""
import os, sys, time, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "lt"))
from ckpt_npz import load_lt, load_data
OUT = os.environ.get("LT_OUT", os.path.join(ROOT, "runs", "analysis")); os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
NB, SEG = int(os.environ.get("LT_NB", "512")), int(os.environ.get("LT_SEG", "128"))
P = os.environ.get("LT_CKPT") or sys.argv[1]
m, cfg, step = load_lt(P, batch_size=NB, loops=SEG + 1)
X, Y, batch = load_data(ROOT, NB)
with torch.device("cuda"): carry = m.initial_carry(batch)
PP = np.zeros((SEG, NB, 81), np.int8)
for si in range(SEG):
    carry, out = m(carry, batch); PP[si] = out["logits"].argmax(-1).cpu().numpy().astype(np.int8)
out = os.environ.get("LT_SAVE") or os.path.join(OUT, f"traj_{os.path.basename(P).rsplit('.',1)[0]}.npz")
np.savez_compressed(out, P=PP, X=X, Y=Y); print(f"# {os.path.basename(P)} step={step} → {out}")
