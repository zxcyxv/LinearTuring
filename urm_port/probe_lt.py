"""구조 프로브: ① 세그먼트별 정확도 (깊이 사용 여부) ② 경계 노름비 (carry 익사 여부)."""
import sys, glob, os
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
import numpy as np, torch
from models.lt.lt import LT
torch.set_grad_enabled(False)

CKD = "/workspace/LinearTuring/refs/URM/checkpoints/lt-sudoku-d832-sw"
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
ck = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(f"{CKD}/step_*.pt"), key=os.path.getmtime)[-1]
st = torch.load(ck, map_location='cuda', weights_only=False)
sd = { (k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith(("model.", "_orig_mod.")) else k): v
       for k, v in st["model_state_dict"].items() }
BS = 128
cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=8, loops=16, causal=False, boundary_mlp=True, ckpt=False)
with torch.device("cuda"): m = LT(cfg)
m.load_state_dict(sd, strict=False); m.eval()

inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")[:512]).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")[:512]).to(torch.int32)
giv = (inp > 1)
segacc, ratios = [], []
for i in range(0, 512, BS):
    b = slice(i, i + BS)
    batch = {k: v.cuda() for k, v in dict(inputs=inp[b], labels=lab[b],
             puzzle_identifiers=torch.zeros(BS, dtype=torch.int32)).items()}
    with torch.device("cuda"): carry = m.initial_carry(batch)
    accs, rats = [], []
    for s in range(16):
        # 경계 노름비: W_O·carry vs 주입 (inner.forward 직전 상태 재현)
        hin = torch.where(carry.halted.view(-1,1,1), m.inner.init_hidden, carry.current_hidden)
        if getattr(m.inner.config, "boundary_mlp", False):
            import torch.nn.functional as F
            gate, up = m.inner.b_gate_up(hin).chunk(2, dim=-1)
            woh = m.inner.b_carry * hin + m.inner.b_down(F.silu(gate) * up)
        else:
            woh = hin @ m.inner.core.w_bo.t()
        g = float(getattr(m.inner, "inj_gate", torch.tensor(1.0)))
        inj = g * m.inner._injection(batch)
        rats.append(float(woh.norm() / inj.norm()))
        carry, out = m(carry, batch)
        pr = out["logits"].argmax(-1).to(torch.int32).cpu()
        ng = ~giv[b]
        accs.append(float((pr[ng.cpu() if ng.device.type=='cuda' else ng] == lab[b][ng]).float().mean()))
    segacc.append(accs); ratios.append(rats)
segacc = np.array(segacc).mean(0); ratios = np.array(ratios).mean(0)
print(f"프로브 ({os.path.basename(ck)}, 512 퍼즐, 빈칸만):  학습된 β = {float(getattr(m.inner, 'inj_gate', torch.tensor(float('nan')))):.3f}")
print("  세그먼트별 정확도:", " ".join(f"{a:.3f}" for a in segacc))
print(f"  → 정점 세그먼트 {int(segacc.argmax())+1}/16,  세그1→16 이득 {segacc[-1]-segacc[0]:+.4f}")
print("  경계 노름비 ‖W_O h‖/‖inj‖:", " ".join(f"{r:.2f}" for r in ratios[:8]), "...")
print(f"  → carry {'익사 (주입 지배)' if ratios[1:].mean() < 0.5 else '생존'} (첫 경계 제외 평균 {ratios[1:].mean():.2f})")
