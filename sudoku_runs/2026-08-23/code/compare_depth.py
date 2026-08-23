"""R1B8 vs URM-L4C6 — 같은 스텝(5859) 체크포인트의 오류를 전파 깊이 클래스별로 분해."""
import sys, numpy as np, torch
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
torch.set_grad_enabled(False)
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")).to(torch.int32)
depth = np.load(f"{D}/cell_depth.npy")
N, BS = len(inp), 128

def predict(model, extra_pad=0):
    preds = torch.zeros(N, 81, dtype=torch.int32)
    for i in range(0, N, BS):
        b = slice(i, min(i+BS, N)); n = b.stop - b.start
        batch = dict(inputs=torch.zeros(BS,81,dtype=torch.int32),
                     labels=torch.zeros(BS,81,dtype=torch.int32),
                     puzzle_identifiers=torch.zeros(BS, dtype=torch.int32))
        batch["inputs"][:n] = inp[b]; batch["labels"][:n] = lab[b]
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.device("cuda"):
            carry = model.initial_carry(batch)
        for _ in range(16):
            carry, out = model(carry, batch)
        lg = out["logits"]
        preds[b] = lg[:n].argmax(-1).to(torch.int32).cpu()
    return preds.numpy()

def load_sd(path):
    st = torch.load(path, map_location='cuda', weights_only=False)
    sd = st["model_state_dict"]
    return {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}

# R1B8
from models.lt.lt import LT
cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, seg_steps=0,
           loops=16, grid=9, ckpt=False, boundary_mlp=True, forward_dtype="float32",
           causal=False, blocks_per_seg=8, block_inj=True)
with torch.device("cuda"):
    lt = LT(cfg)
ms, us = lt.load_state_dict(load_sd("/workspace/LinearTuring/refs/URM/checkpoints/lt1k_R1B8/step_5859.pt"), strict=False)
assert not [m for m in ms if "pos_" not in m], ms
lt.eval(); p_lt = predict(lt)
del lt; torch.cuda.empty_cache()

# URM L4C6
from models.urm.urm import URM
ucfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
            puzzle_emb_ndim=512, hidden_size=512, num_heads=8, expansion=4,
            num_layers=4, loops=16, H_cycles=2, L_cycles=6,
            pos_encodings="rope", forward_dtype="bfloat16", causal=False)
with torch.device("cuda"):
    um = URM(ucfg)
ms, us = um.load_state_dict(load_sd("/workspace/LinearTuring/refs/URM/checkpoints/urm1k_L4C6/step_5859.pt"), strict=False)
assert not ms, ms
um.eval(); p_um = predict(um)

lab_n, inp_n = lab.numpy(), inp.numpy()
giv = inp_n > 1
classes = [("주어짐", giv), ("전파 1-2", (depth>=1)&(depth<3)), ("전파 3-5", (depth>=3)&(depth<6)),
           ("전파 6+", depth>=6), ("탐색 필요", depth==-1)]
print(f"{'클래스':12s} {'비중':>6s} | {'R1B8':>7s} | {'L4C6':>7s}")
for nm, msk in classes:
    print(f"{nm:12s} {msk.mean()*100:5.1f}% | {(p_lt==lab_n)[msk].mean():.4f} | {(p_um==lab_n)[msk].mean():.4f}")
for nm, p in (("R1B8", p_lt), ("L4C6", p_um)):
    ok = (p == lab_n); frac = ok.mean(1)
    bins = [(1.0,1.01,"=1.0 (완답)"), (0.95,1.0,"[.95,1)"), (0.8,0.95,"[.8,.95)"), (0.0,0.8,"<.8")]
    dist = "  ".join(f"{lb} {( (frac>=lo)&(frac<hi) ).mean()*100:.1f}%" for lo,hi,lb in bins)
    print(f"{nm}: 셀 {ok.mean():.4f}  완답 {int(ok.all(1).sum())}/2048 | 퍼즐분포: {dist}")
