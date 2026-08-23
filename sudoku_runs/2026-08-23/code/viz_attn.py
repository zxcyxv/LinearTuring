"""완답 샘플 4개: 루프별 (중간 답안 보드 + 초점 칸으로 들어오는 어텐션) 진화 + 8헤드 전체 행렬."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.lt.lt import LT
torch.set_grad_enabled(False)

D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
OUT = "/workspace/LinearTuring/sudoku_runs/2026-08-23/figs"
import os; os.makedirs(OUT, exist_ok=True)
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")).to(torch.int32)
depth = np.load(f"{D}/cell_depth.npy")
solved = np.load("/workspace/LinearTuring/sudoku_runs/2026-08-23/interp_solved.npz")["solved_idx"]
S = solved[:4]
BS = 4

cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, seg_steps=0,
           loops=16, grid=9, ckpt=False, boundary_mlp=True, forward_dtype="float32",
           causal=False, blocks_per_seg=16, block_inj=True)
with torch.device("cuda"):
    lt = LT(cfg)
st = torch.load("/workspace/LinearTuring/refs/URM/checkpoints/lt1k_R1B16/step_19530.pt",
                map_location="cuda", weights_only=False)
sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in st["model_state_dict"].items()}
lt.load_state_dict(sd, strict=False); lt.eval()
inner = lt.inner; m = inner.core

batch = dict(inputs=inp[S].cuda(), labels=lab[S].cuda(),
             puzzle_identifiers=torch.zeros(BS, dtype=torch.int32, device="cuda"))
LOOPS_SHOW = [0, 1, 3, 7, 15]
fc = m.kernel_fast(); AB = m.W_C(); dt = 1.0
injv = inner._injection(batch)
h = inner.init_hidden.expand(BS, 81, 832).clone()
attn_frames = {}     # loop -> a [B,H,81,81] (loop 마지막 스텝)
am_loops = np.zeros((BS, 16, 81), np.int8)
for loop in range(16):
    for blk in range(16):
        g_, u_ = inner.b_gate_up(h).chunk(2, dim=-1)
        h = inner.b_carry * h + inner.b_down(F.silu(g_) * u_)
        h = h + inner.inj_gate * injv
        h = m.phi(h, dt / 2)
        f, a, *_ = m.field(h, None, None, None, AB, fast_ctx=fc)
        h = h + dt * f
        h = m.phi(h, dt / 2)
    if loop in LOOPS_SHOW:
        attn_frames[loop] = a.float().cpu().numpy()
    am_loops[:, loop] = m.w_cls(h).argmax(-1).to(torch.int8).cpu().numpy()

inp_n, lab_n, dep = inp[S].numpy(), lab[S].numpy(), depth[S]
# 초점 칸: 퍼즐별 가장 늦게 굳는 칸 (loop 단위 재계산)
correct = (am_loops == lab_n[:, None, :])
commit_loop = 16 - np.cumprod(correct[:, ::-1, :], 1).astype(bool).sum(1)
focus = [int(np.argmax(commit_loop[p] * (inp_n[p] == 1))) for p in range(BS)]  # 빈칸 중 최후 확정

def draw_board(ax, ann, colors, title=""):
    ax.imshow(colors, vmin=0, vmax=1)
    for i in range(10):
        lw = 2.2 if i % 3 == 0 else 0.4
        ax.axhline(i - .5, color='k', lw=lw); ax.axvline(i - .5, color='k', lw=lw)
    for t in range(81):
        if ann[t]: ax.text(t % 9, t // 9, ann[t], ha='center', va='center', fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
    ax.set_xlim(-.5, 8.5); ax.set_ylim(8.5, -.5)

H_SHOW = [0, 5]   # 억제 대표 / 흥분 대표
for p in range(BS):
    fig, axes = plt.subplots(1 + len(H_SHOW), len(LOOPS_SHOW), figsize=(3.0*len(LOOPS_SHOW), 3.1*(1+len(H_SHOW))))
    fr, fcol = focus[p] // 9, focus[p] % 9
    for j, lp in enumerate(LOOPS_SHOW):
        # 1행: 중간 답안 보드
        ann, colors = [], np.ones((9, 9, 3))
        for t in range(81):
            given = inp_n[p, t] > 1
            cur = am_loops[p, lp, t]
            ok_now = cur == lab_n[p, t]
            committed = commit_loop[p, t] <= lp
            ann.append(str(int(cur - 1)) if cur >= 2 else "")
            r, c = t // 9, t % 9
            if given: colors[r, c] = (.85, .85, .85)
            elif committed: colors[r, c] = (.72, .92, .72)     # 확정(이후 불변)
            elif ok_now: colors[r, c] = (1.0, 1.0, .75)        # 지금은 정답이나 아직 흔들림
            else: colors[r, c] = (1.0, .72, .72)               # 오답
        ax = axes[0, j]; draw_board(ax, ann, colors, f"loop {lp} (board)")
        ax.add_patch(plt.Rectangle((fcol-.5, fr-.5), 1, 1, fill=False, edgecolor='blue', lw=2.5))
        # 2-3행: 초점 칸으로 들어오는 어텐션 (헤드 0 / 5)
        for k, hh in enumerate(H_SHOW):
            av = attn_frames[lp][p, hh, focus[p], :].reshape(9, 9)
            vmax = np.abs(av).max() + 1e-9
            ax = axes[1 + k, j]
            ax.imshow(av, cmap='RdBu', vmin=-vmax, vmax=vmax)   # 빨강=음수(억제), 파랑=양수
            for i in range(10):
                lw = 2.2 if i % 3 == 0 else 0.4
                ax.axhline(i-.5, color='k', lw=lw); ax.axvline(i-.5, color='k', lw=lw)
            for t in range(81):
                if inp_n[p, t] > 1:
                    ax.text(t % 9, t // 9, str(int(inp_n[p, t] - 1)), ha='center', va='center',
                            fontsize=6, color='gray')
            ax.add_patch(plt.Rectangle((fcol-.5, fr-.5), 1, 1, fill=False, edgecolor='blue', lw=2.5))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"head {hh} → focus  loop {lp}", fontsize=9)
    dcls = "search" if dep[p, focus[p]] == -1 else f"prop{dep[p, focus[p]]}"
    fig.suptitle(f"sample {S[p]} - focus r{fr}c{fcol} [{dcls}, commits@loop {commit_loop[p, focus[p]]}]  |  "
                 f"board: gray=given green=locked yellow=correct-unstable red=wrong  |  attn to focus: RED=inhibit BLUE=excite", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/evo_S{S[p]}.png", dpi=130); plt.close(fig)
    print(f"evo_S{S[p]}.png  (focus r{fr}c{fcol}, {dcls}, commit@loop{commit_loop[p,focus[p]]})")

# 8헤드 전체 81×81 (샘플 0, loop 0 vs 15)
fig, axes = plt.subplots(2, 8, figsize=(22, 6))
for k, lp in enumerate([0, 15]):
    for hh in range(8):
        av = attn_frames[lp][0, hh]
        vmax = np.abs(av).max()
        axes[k, hh].imshow(av, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[k, hh].set_xticks([]); axes[k, hh].set_yticks([])
        axes[k, hh].set_title(f"head {hh}  loop {lp}", fontsize=9)
fig.suptitle(f"sample {S[0]} - 8 heads, 81x81 signed attention (RED=inhibit BLUE=excite; row-major: row-peers=diagonal blocks, col-peers=stride-9 stripes)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/heads_S{S[0]}.png", dpi=120); plt.close(fig)
print(f"heads_S{S[0]}.png")
