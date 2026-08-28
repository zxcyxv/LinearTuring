"""완답 샘플의 루프별 (중간 답안 보드 + 초점 칸으로 들어오는 부호 어텐션) 진화 그림 + 8헤드 81×81 행렬.
빨강 = 음수(억제), 파랑 = 양수(흥분). 초점 칸 = 빈칸 중 가장 늦게 확정된 칸.
사용: python viz_attn.py [--ckpt PATH --bilinear 0|1] [--samples 4] [--heads 0 5] [--out DIR]
원 그림: results/figs/evo_S*.png, heads_S1.png (SwiGLU B16 판)."""
import argparse, os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_lt, load_test, make_batch, rollout, logits, CKPT_DEFAULT, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--samples", type=int, default=4); ap.add_argument("--heads", type=int, nargs="+", default=[0, 5])
ap.add_argument("--loops", type=int, nargs="+", default=[0, 1, 3, 7, 15])
ap.add_argument("--out", default=f"{ROOT}/results/figs")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.samples)
inp, lab, depth = load_test(); L, K = m.config.loops, m.config.blocks_per_seg

# 완답 샘플 고르기 (앞에서부터)
S = []
mm = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=128)
for i in range(0, len(inp), 128):
    b = slice(i, i + 128); h = rollout(mm, make_batch(inp[b], lab[b]))
    S += (i + torch.where((logits(mm, h).argmax(-1).to(torch.int32) == lab[b]).all(1))[0].cpu()).tolist()
    if len(S) >= args.samples: break
S = S[:args.samples]; B = len(S); idx = torch.tensor(S, device="cuda")
frames, am = {}, np.zeros((B, L, 81), np.int8)
def hook(loop, blk, stage, h, a):
    if stage == "post_step" and blk == K - 1:
        if loop in args.loops: frames[loop] = a.float().cpu().numpy()
        am[:, loop] = logits(m, h).argmax(-1).to(torch.int8).cpu().numpy()
rollout(m, make_batch(inp[idx], lab[idx]), hook=hook)
inp_n, lab_n, dep = inp[idx].cpu().numpy(), lab[idx].cpu().numpy(), depth[S]
commit_loop = L - np.cumprod((am == lab_n[:, None, :])[:, ::-1, :], 1).astype(bool).sum(1)
focus = [int(np.argmax(commit_loop[p] * (inp_n[p] == 1))) for p in range(B)]

def grid(ax):
    for i in range(10):
        lw = 2.2 if i % 3 == 0 else 0.4; ax.axhline(i - .5, color='k', lw=lw); ax.axvline(i - .5, color='k', lw=lw)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-.5, 8.5); ax.set_ylim(8.5, -.5)

for p in range(B):
    fig, axes = plt.subplots(1 + len(args.heads), len(args.loops), figsize=(3.0 * len(args.loops), 3.1 * (1 + len(args.heads))))
    fr, fc = focus[p] // 9, focus[p] % 9
    for j, lp in enumerate(args.loops):
        colors = np.ones((9, 9, 3)); ax = axes[0, j]
        for t in range(81):
            r, c = t // 9, t % 9; cur = am[p, lp, t]
            if inp_n[p, t] > 1: colors[r, c] = (.85, .85, .85)
            elif commit_loop[p, t] <= lp: colors[r, c] = (.72, .92, .72)
            elif cur == lab_n[p, t]: colors[r, c] = (1, 1, .75)
            else: colors[r, c] = (1, .72, .72)
            if cur >= 2: ax.text(c, r, str(int(cur - 1)), ha='center', va='center', fontsize=7.5)
        ax.imshow(colors, vmin=0, vmax=1); grid(ax); ax.set_title(f"loop {lp} (board)", fontsize=9)
        ax.add_patch(plt.Rectangle((fc - .5, fr - .5), 1, 1, fill=False, edgecolor='blue', lw=2.5))
        for k, hh in enumerate(args.heads):
            av = frames[lp][p, hh, focus[p], :].reshape(9, 9); vmax = np.abs(av).max() + 1e-9; ax = axes[1 + k, j]
            ax.imshow(av, cmap='RdBu', vmin=-vmax, vmax=vmax); grid(ax)
            for t in range(81):
                if inp_n[p, t] > 1: ax.text(t % 9, t // 9, str(int(inp_n[p, t] - 1)), ha='center', va='center', fontsize=6, color='gray')
            ax.add_patch(plt.Rectangle((fc - .5, fr - .5), 1, 1, fill=False, edgecolor='blue', lw=2.5))
            ax.set_title(f"head {hh} → focus  loop {lp}", fontsize=9)
    dcls = "search" if dep[p, focus[p]] == -1 else f"prop{dep[p, focus[p]]}"
    fig.suptitle(f"sample {S[p]} - focus r{fr}c{fc} [{dcls}, commits@loop {commit_loop[p, focus[p]]}]  |  "
                 f"board: gray=given green=locked yellow=correct-unstable red=wrong  |  attn to focus: RED=inhibit BLUE=excite", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"{args.out}/evo_S{S[p]}.png", dpi=130); plt.close(fig)
    print(f"evo_S{S[p]}.png  (focus r{fr}c{fc}, {dcls}, commit@loop{commit_loop[p, focus[p]]})")
H = m.config.num_heads
fig, axes = plt.subplots(2, H, figsize=(2.75 * H, 6))
for k, lp in enumerate([args.loops[0], args.loops[-1]]):
    for hh in range(H):
        av = frames[lp][0, hh]; vmax = np.abs(av).max()
        axes[k, hh].imshow(av, cmap='RdBu', vmin=-vmax, vmax=vmax); axes[k, hh].set_xticks([]); axes[k, hh].set_yticks([])
        axes[k, hh].set_title(f"head {hh}  loop {lp}", fontsize=9)
fig.suptitle(f"sample {S[0]} - {H} heads, 81x81 signed attention (RED=inhibit BLUE=excite)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{args.out}/heads_S{S[0]}.png", dpi=120); print(f"heads_S{S[0]}.png")
