"""head5 '파랑→하양' 사건의 논리적 정체: 미해결 피어 수집 가설 검증."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from models.lt.lt import LT
torch.set_grad_enabled(False)
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")).to(torch.int32)
solved = np.load("/workspace/LinearTuring/sudoku_runs/2026-08-23/interp_solved.npz")["solved_idx"]
S = solved[:128]; BS = 128
cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, seg_steps=0,
           loops=16, grid=9, ckpt=False, boundary_mlp=True, forward_dtype="float32",
           causal=False, blocks_per_seg=16, block_inj=True)
with torch.device("cuda"): lt = LT(cfg)
st = torch.load("/workspace/LinearTuring/refs/URM/checkpoints/lt1k_R1B16/step_19530.pt", map_location="cuda", weights_only=False)
lt.load_state_dict({k.replace("model.","",1) if k.startswith("model.") else k: v for k,v in st["model_state_dict"].items()}, strict=False)
lt.eval(); inner = lt.inner; m = inner.core
batch = dict(inputs=inp[S].cuda(), labels=lab[S].cuda(), puzzle_identifiers=torch.zeros(BS, dtype=torch.int32, device="cuda"))
fc = m.kernel_fast(); AB = m.W_C(); dt = 1.0
injv = inner._injection(batch); h = inner.init_hidden.expand(BS, 81, 832).clone()
A5 = np.zeros((16, BS, 81, 81), np.float16); A0 = np.zeros((16, BS, 81, 81), np.float16)
am = np.zeros((BS, 16, 81), np.int8)
for loop in range(16):
    for blk in range(16):
        g_, u_ = inner.b_gate_up(h).chunk(2, dim=-1)
        h = inner.b_carry * h + inner.b_down(F.silu(g_) * u_)
        h = h + inner.inj_gate * injv
        h = m.phi(h, dt/2)
        f, a, *_ = m.field(h, None, None, None, AB, fast_ctx=fc)
        h = h + dt*f; h = m.phi(h, dt/2)
    A5[loop] = a[:, 5].cpu().numpy(); A0[loop] = a[:, 0].cpu().numpy()
    am[:, loop] = m.w_cls(h).argmax(-1).to(torch.int8).cpu().numpy()
lab_n, inp_n = lab[S].numpy(), inp[S].numpy()
commit = 16 - np.cumprod((am == lab_n[:,None,:])[:, ::-1, :], 1).astype(bool).sum(1)  # [B,81] (loop 단위)

# ── S1 사건 국소 판독 ──
p = 0; focus = 0*9+5
d75 = np.abs(A5[7, p, focus]) - np.abs(A5[15, p, focus])
top = np.argsort(-d75)[:4]
print("S1 focus r0c5: 확정 loop =", commit[p, focus], " 정답 =", int(lab_n[p, focus]-1))
print("loop7→15 에서 head5 유입이 가장 죽은 칸들:")
for n in top:
    print(f"  r{n//9}c{n%9}: given={inp_n[p,n]>1} 정답={int(lab_n[p,n]-1)} 확정loop={commit[p,n]} "
          f"a5: L0 {A5[0,p,focus,n]:+.3f} L7 {A5[7,p,focus,n]:+.3f} L15 {A5[15,p,focus,n]:+.3f}")
# 시계열 그림
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
for n in top[:3]:
    ax.plot(range(16), A5[:, p, focus, n], marker='o', ms=3, label=f"from r{n//9}c{n%9} (commit L{commit[p,n]})")
ax.plot(range(16), [A5[l, p, focus, focus] for l in range(16)], 'k--', lw=1, label="self")
ax.axvline(commit[p, focus], color='r', ls=':', label=f"focus commits (L{commit[p,focus]})")
ax.axhline(0, color='gray', lw=.5); ax.set_xlabel("loop"); ax.set_ylabel("a_head5[focus <- n]")
ax.legend(fontsize=8); ax.set_title("S1: head-5 excitatory inflow to focus r0c5 over loops")
fig.tight_layout(); fig.savefig("/workspace/LinearTuring/sudoku_runs/2026-08-23/figs/head5_timecourse_S1.png", dpi=130)
print("→ head5_timecourse_S1.png")

# ── 모집단 검증: 미확정 vs 확정 피어에서 오는 유입 (빈칸 수신자만) ──
pm = np.zeros((81,81), bool)
for t in range(81):
    r,c = t//9, t%9
    for u in range(81):
        if u==t: continue
        if u//9==r or u%9==c or (u//9//3==r//3 and u%9//3==c//3): pm[t,u]=True
rows = []
for loop in [0,3,7,11,15]:
    unres = commit[:, None, :] > loop        # 송신자 n 이 loop 시점 미확정 [B,1,81]
    blank_rx = (inp_n == 1)[:, :, None]      # 수신자 t 는 빈칸
    msk_u = blank_rx & pm[None] & unres
    msk_r = blank_rx & pm[None] & ~unres
    a5u = np.abs(A5[loop])[msk_u].mean(); a5r = np.abs(A5[loop])[msk_r].mean()
    s5u = A5[loop][msk_u].mean(); s5r = A5[loop][msk_r].mean()
    a0u = np.abs(A0[loop])[msk_u].mean(); a0r = np.abs(A0[loop])[msk_r].mean()
    rows.append((loop, a5u, a5r, s5u, s5r, a0u, a0r))
print("\n피어→빈칸 유입 (모집단 128): |a| 및 부호평균, 송신자 미확정 vs 확정")
print(f"{'loop':>4} | h5 |a| 미확정/확정 | h5 부호 미확정/확정 | h0 |a| 미확정/확정")
for r_ in rows:
    print(f"{r_[0]:>4} |  {r_[1]:.4f} / {r_[2]:.4f}  |  {r_[3]:+.4f} / {r_[4]:+.4f}  |  {r_[5]:.4f} / {r_[6]:.4f}")
