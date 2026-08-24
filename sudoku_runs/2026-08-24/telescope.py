"""망원경 절제: 공유가중치 블록 16개가 각자 일을 하는가, 아니면 접혀서 항등이 되는가.
   학습 불필요 — R1B16 체크포인트에 블록 건너뛰기/절단을 가해 성능 낙폭을 측정."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
torch.set_grad_enabled(False)

D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")).to(torch.int32)
N = 512
inp, lab = inp[:N].cuda(), lab[:N].cuda()

cfg = dict(batch_size=N, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, seg_steps=0,
           loops=16, grid=9, ckpt=False, boundary_mlp=True, forward_dtype="float32",
           causal=False, blocks_per_seg=16, block_inj=True)
with torch.device("cuda"):
    lt = LT(cfg)
st = torch.load("/workspace/LinearTuring/sudoku_runs/2026-08-23/checkpoints/R1B16_step9765.pt",
                map_location="cuda", weights_only=False)
sd = st["model_state_dict"] if "model_state_dict" in st else st
sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}
missing, unexpected = lt.load_state_dict(sd, strict=False)
print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
lt.eval(); inner = lt.inner; m = inner.core

batch = dict(inputs=inp, labels=lab,
             puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device="cuda"))
fc = m.kernel_fast(); AB = m.W_C(); dt = 1.0
injv = inner._injection(batch)

def run(skip=None, keep=16, trace=False):
    """skip: 매 세그에서 건너뛸 블록 인덱스. keep: 세그당 실행할 블록 수."""
    h = inner.init_hidden.expand(N, 81, 832).clone()
    disp = np.zeros(16)
    for loop in range(16):
        for blk in range(keep):
            if skip is not None and blk == skip:
                continue
            h0 = h
            g_, u_ = inner.b_gate_up(h).chunk(2, dim=-1)
            h = inner.b_carry * h + inner.b_down(F.silu(g_) * u_)
            h = h + inner.inj_gate * injv
            h = m.phi(h, dt/2)
            f, a, *_ = m.field(h, None, None, None, AB, fast_ctx=fc)
            h = h + dt*f; h = m.phi(h, dt/2)
            if trace and loop == 15:
                disp[blk] = ((h-h0).norm(dim=-1)/h0.norm(dim=-1)).mean().item()
    pred = m.w_cls(h).argmax(-1)
    blank = inp == 1
    cell = ((pred == lab) & blank).sum().item() / blank.sum().item()
    exact = ((pred == lab) | ~blank).all(-1).sum().item()
    return cell, exact, disp

base_c, base_e, disp = run(trace=True)
print(f"\n기준(16블록 전체):  셀 {base_c:.4f}  완답 {base_e}/{N}\n")

print("── ① 블록 1개 제거 (매 세그, 15블록 실행) ──")
print(f"{'제거블록':>8} {'셀':>8} {'Δ셀':>8} {'완답':>6} {'변위‖Δh‖/‖h‖':>14}")
for k in range(16):
    c, e, _ = run(skip=k)
    print(f"{k:>8} {c:>8.4f} {c-base_c:>+8.4f} {e:>6} {disp[k]:>14.4f}")

print("\n── ② 앞에서부터 k블록만 실행 (깊이 절단) ──")
print(f"{'keep':>5} {'셀':>8} {'완답':>6}")
for k in [1,2,4,6,8,10,12,14,16]:
    c, e, _ = run(keep=k)
    print(f"{k:>5} {c:>8.4f} {e:>6}")
