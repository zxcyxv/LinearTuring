import os, sys, time, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "lt"))
from ckpt_npz import load_lt, load_data
OUT = os.environ.get("LT_OUT", os.path.join(ROOT, "runs", "analysis")); os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
NB = int(os.environ.get("LT_NB", "512")); SEG = int(os.environ.get("LT_SEG", "128"))   # 기본 128 = 기존 기록과 비교 가능
P = os.environ.get("LT_CKPT") or sys.argv[1]
m, cfg, step = load_lt(P, batch_size=NB, loops=SEG + 1)
print(f"# ckpt {os.path.basename(P)}  step={step}  weights=ema  gauge={'legacy' if cfg.get('legacy_gauge') else 'sqrt_d'} order={cfg.get('block_order')} trace={cfg.get('use_trace', 'trace_rho_init' in cfg)}")
X, Y, batch = load_data(ROOT, NB); y = batch["labels"]; blank = (batch["inputs"] == 1)
t0 = time.time()
with torch.device("cuda"): carry = m.initial_carry(batch)
prev = None; rows = []
for si in range(SEG):
    carry, out = m(carry, batch); p = out["logits"].argmax(-1)
    acc = (p == y).float().mean().item(); ex = int((p == y).all(-1).sum())
    ch = 0.0 if prev is None else ((p != prev) & blank).float().sum().item() / blank.sum().item()
    prev = p.clone(); rows.append((si + 1, acc, ex, ch))
print(f"{'seg':>5} {'acc':>8} {'exact':>6} {'exact%':>8} {'churn':>8}")
for si, acc, ex, ch in rows:
    if si in (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256) or si == SEG:
        print(f"{si:5d} {acc:8.4f} {ex:6d} {100*ex/NB:8.2f} {ch:8.5f}" + ("  <-train" if si == 16 else ""))
e16 = rows[15][2]
def summarize(rs, tag):
    eb = max(r[2] for r in rs); sb = [r[0] for r in rs if r[2] == eb][0]
    cap = "  <-상한에 걸림" if sb == rs[-1][0] else ""
    print(f"seg16 {e16}/{NB} ({100*e16/NB:.2f}%)   best {eb}/{NB} ({100*eb/NB:.2f}%) @seg{sb}   +{eb-e16}{cap}   [{tag}]")
    return 100 * ((NB - e16) - (NB - eb)) / (NB - e16)
print()
r128 = summarize(rows[:128], "seg128 기준"); print(f"오차 감소율 = {r128:.1f}%   ({time.time()-t0:.0f}s)")
if SEG > 128:
    rS = summarize(rows, f"seg{SEG} 기준"); print(f"오차 감소율(seg{SEG}) = {rS:.1f}%")
