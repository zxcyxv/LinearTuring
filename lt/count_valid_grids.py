"""체크포인트별 세그먼트마다 무모순 격자(행·열·박스 전부 순열) 수. 분류: 정답 / 무모순·단서변경 / 무모순·단서유지·오답 / 모순.  LT_SAVE 로 P 저장."""
import os, sys, time, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "lt"))
from ckpt_npz import load_lt, load_data
OUT = os.environ.get("LT_OUT", os.path.join(ROOT, "runs", "analysis")); os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
from state_transitions import valid
NB, SEG = int(os.environ.get("LT_NB", "512")), int(os.environ.get("LT_SEG", "128"))
P = os.environ.get("LT_CKPT") or sys.argv[1]
m, cfg, step = load_lt(P, batch_size=NB, loops=SEG + 1)
X, Y, batch = load_data(ROOT, NB); clue = (X != 0)
t0 = time.time()
with torch.device("cuda"): carry = m.initial_carry(batch)
rows = []; PP = np.zeros((SEG, NB, 81), np.int8)
for si in range(SEG):
    carry, out = m(carry, batch); p = out["logits"].argmax(-1).cpu().numpy(); PP[si] = p.astype(np.int8)
    D = p - 1; v = valid(D); correct = (D == Y).all(1)
    clue_kept = np.array([(D[b][clue[b]] == X[b][clue[b]]).all() for b in range(NB)])
    rows.append((si + 1, int(correct.sum()), int((v & ~correct & ~clue_kept).sum()), int((v & ~correct & clue_kept).sum()), int((~v).sum())))
print(f"# {os.path.basename(P)} step={step} EMA  ({time.time()-t0:.0f}s)")
print(f"{'seg':>4} {'정답':>5} {'무모순·단서변경':>9} {'무모순·단서유지·오답':>11} {'모순':>5} | {'무모순 합':>7}")
for r in rows:
    if r[0] in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
        print(f"{r[0]:4d} {r[1]:5d} {r[2]:9d} {r[3]:11d} {r[4]:5d} | {r[1]+r[2]+r[3]:7d}")
bestv = max(rows, key=lambda r: r[1] + r[2] + r[3]); bestc = max(rows, key=lambda r: r[1])
print(f"무모순 최대 {bestv[1]+bestv[2]+bestv[3]} @seg{bestv[0]}   정답 최대 {bestc[1]} @seg{bestc[0]}")
if os.environ.get("LT_SAVE"): np.savez_compressed(os.environ["LT_SAVE"], P=PP, X=X, Y=Y)
