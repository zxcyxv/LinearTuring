"""독립 eval — 체크포인트를 로드해 held-out 테스트 2048 에서 셀/전체일치 + 전파깊이별 분해."""
import sys, glob, os
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
import numpy as np, torch
from models.lt.lt import LT
torch.set_grad_enabled(False)

CKD = "/workspace/LinearTuring/refs/URM/checkpoints/lt-sudoku-d832-sw"
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
ck_file = sorted(glob.glob(f"{CKD}/*.pt") + glob.glob(f"{CKD}/step_*"), key=os.path.getmtime)[-1] if (glob.glob(f"{CKD}/*.pt") or glob.glob(f"{CKD}/step_*")) else None
assert ck_file, "체크포인트 없음"
print("체크포인트:", ck_file)
st = torch.load(ck_file, map_location='cuda', weights_only=False)
sd = st["model_state_dict"] if "model_state_dict" in st else st
sd = { (k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith(("model.", "_orig_mod.")) else k): v
       for k, v in sd.items() }

BS = 128
cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=8, loops=16, causal=False, boundary_mlp=True, ckpt=False)
with torch.device("cuda"):
    m = LT(cfg)
missing, unexpected = m.load_state_dict(sd, strict=False)
print("state_dict:", "OK" if not missing else f"missing {len(missing)}", f"unexpected {len(unexpected)}" if unexpected else "")
m.eval()

inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")).to(torch.int32)
depth = np.load(f"{D}/cell_depth.npy")
N = len(inp)
preds = torch.zeros(N, 81, dtype=torch.int32)
for i in range(0, N, BS):
    b = slice(i, min(i+BS, N))
    n = b.stop - b.start
    batch = dict(inputs=torch.zeros(BS,81,dtype=torch.int32), labels=torch.zeros(BS,81,dtype=torch.int32),
                 puzzle_identifiers=torch.zeros(BS, dtype=torch.int32))
    batch["inputs"][:n] = inp[b]; batch["labels"][:n] = lab[b]
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.device("cuda"):
        carry = m.initial_carry(batch)
    for _ in range(16):
        carry, out = m(carry, batch)
    preds[b] = out["logits"][:n].argmax(-1).to(torch.int32).cpu()

lab_n, inp_n = lab.numpy(), inp.numpy()
ok = (preds.numpy() == lab_n)
cell, exact = ok.mean(), ok.all(1).mean()
print(f"\n══ held-out 2048  셀 {cell:.4f}   전체일치 {exact:.4f} ══")
giv = inp_n > 1
print(f"  주어진 칸 (31.2%):   {ok[giv].mean():.4f}")
for lo, hi, nm in ((1,3,"전파 1–2"), (3,6,"전파 3–5"), (6,20,"전파 6+")):
    msk = (depth >= lo) & (depth < hi)
    print(f"  {nm:10s} ({msk.mean()*100:4.1f}%): {ok[msk].mean():.4f}")
srch = depth == -1
print(f"  탐색 필요  ({srch.mean()*100:4.1f}%): {ok[srch].mean():.4f}   [찍기 기준 ≈ 0.25~0.33]")
