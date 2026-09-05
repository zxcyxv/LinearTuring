"""attn_viz_155.npz 재시각화 — 명확한 라벨, 컬러바, 변화량 중심."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/workspace/LinearTuring"; FIG = os.path.join(ROOT, "2026-09-05/results/figs")
d = np.load(os.path.join(ROOT, "2026-09-05/results/json/attn_viz_155.npz")); A = d["A"].astype(np.float32); LAY = d["layer"]; PR = d["pred"]; X = d["X"]; Y = d["Y"]
nb = len(LAY); bps = 16; S = nb // bps; RUN = ["ORIGINAL (fails, 37 wrong at end)", "RELABELED digits (solved at seg 14)"]; RS = ["ORIGINAL", "RELABELED"]
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
wrong = (PR != Y[None]); nwrong = wrong.sum(-1)
solve_blk = int(np.argmax(nwrong[:, 1] == 0)) if (nwrong[:, 1] == 0).any() else None
Ah = A.mean(2)                                                           # head mean [blk,2,81,81]
def L1blocks(): return [i for i in range(nb) if LAY[i] == 1]
def last_L1_in_seg(sg): return max(i for i in L1blocks() if i < sg * bps)
# ---------- Fig A: time series (clear)
fig, ax = plt.subplots(4, 1, figsize=(15, 11), sharex=True); t = np.arange(nb); col = ["#d62728", "#1f77b4"]
for k in range(2): ax[0].plot(t, nwrong[:, k], color=col[k], lw=2, label=RUN[k])
churn = np.concatenate([np.zeros((1, 2)), (PR[1:] != PR[:-1]).mean(-1)])
for k in range(2): ax[1].plot(t, churn[:, k], color=col[k], lw=1.5, label=RUN[k])
L1 = np.array(L1blocks()); a_peer = np.array([[Ah[b, k][peer].mean() for k in range(2)] for b in L1]); a_non = np.array([[Ah[b, k][~peer & ~np.eye(81, dtype=bool)].mean() for k in range(2)] for b in L1])
for k in range(2): ax[2].plot(L1, a_peer[:, k], color=col[k], lw=2, label=RUN[k] + " — peer pairs"); ax[2].plot(L1, a_non[:, k], color=col[k], lw=1, ls=":", label=RUN[k] + " — non-peer pairs")
rel = np.array([np.linalg.norm(Ah[b, 1] - Ah[b, 0]) / np.linalg.norm(Ah[b, 0]) for b in L1]); ax[3].plot(L1, rel, color="k", lw=2)
for a_ in ax:
    for sg in range(S): a_.axvspan(sg * bps, (sg + 1) * bps, color="k" if sg % 2 else "w", alpha=0.04)
    if solve_blk: a_.axvline(solve_blk, color="green", lw=2, ls="--")
ax[0].text(solve_blk + 2, 30, f"RELABELED solved\n(block {solve_blk}, seg {solve_blk//bps+1})", color="green", fontsize=11)
ax[0].set_ylabel("wrong cells (of 81)", fontsize=11); ax[0].legend(fontsize=11); ax[0].set_title("puzzle #155 — same puzzle, only digit names changed (2→3,3→8,4→2,5→4,6→7,7→5,8→6)", fontsize=13)
ax[1].set_ylabel("churn per block\n(frac cells changed)", fontsize=11); ax[1].legend(fontsize=9)
ax[2].set_ylabel("layer-1 a_eff mean\n(head-avg)", fontsize=11); ax[2].legend(fontsize=9, ncol=2); ax[2].axhline(0, color="gray", lw=.5)
ax[3].set_ylabel("|A_relabeled − A_original|\n/ |A_original|  (layer 1)", fontsize=11); ax[3].set_xlabel("block index (16 blocks per segment; shaded bands = segments)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "v2_A_timeseries.png"), dpi=110); plt.close(fig)
# ---------- Fig B: change over time within each run (relative to seg1 end), layer 1
segs = [1, 2, 4, 8, 12, 14, 16]; idx = [last_L1_in_seg(s) for s in segs]; ref = idx[0]
fig, ax = plt.subplots(3, len(idx), figsize=(2.9 * len(idx) + 1.5, 9.6))
absmax = np.percentile(np.abs(Ah[idx]), 99.5); dmax = max(np.abs(Ah[i, k] - Ah[ref, k]).max() for i in idx for k in range(2)) * 0.6
for j, b in enumerate(idx):
    im0 = ax[0, j].imshow(Ah[b, 0], cmap="RdBu_r", vmin=-absmax, vmax=absmax)
    im1 = ax[1, j].imshow(Ah[b, 1] - Ah[ref, 1], cmap="PuOr", vmin=-dmax, vmax=dmax)
    im2 = ax[2, j].imshow(Ah[b, 1] - Ah[b, 0], cmap="PuOr", vmin=-dmax, vmax=dmax)
    for k in range(3): ax[k, j].set_xticks([]); ax[k, j].set_yticks([]); ax[k, j].set_title(f"seg {segs[j]}" + (" ← SOLVED" if solve_blk and b >= solve_blk else ""), fontsize=11, color=("green" if solve_blk and b >= solve_blk else "k"))
ax[0, 0].set_ylabel("ORIGINAL run\nabsolute a_eff\n(blue<0 repel, red>0)", fontsize=10); ax[1, 0].set_ylabel("RELABELED run\nchange since seg 1\n(a(t) − a(seg1))", fontsize=10); ax[2, 0].set_ylabel("RELABELED − ORIGINAL\nat same time", fontsize=10)
fig.colorbar(im0, ax=ax[0, :].tolist(), fraction=0.015, pad=0.01); fig.colorbar(im1, ax=ax[1, :].tolist(), fraction=0.015, pad=0.01); fig.colorbar(im2, ax=ax[2, :].tolist(), fraction=0.015, pad=0.01)
fig.suptitle("puzzle #155 — layer-1 coupling matrix (81×81, cells in row-major order; diagonal 9×9 blocks = same row, stripes at offset 9k = same column)", fontsize=12)
fig.savefig(os.path.join(FIG, "v2_B_L1_change.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
# ---------- Fig C: per-head at seg 14 (just before solve) and seg 16 (after), both runs
b14, b16 = last_L1_in_seg(14), last_L1_in_seg(16); H = A.shape[2]
fig, ax = plt.subplots(4, H, figsize=(2.3 * H + 1, 10)); vm = np.percentile(np.abs(A[[b14, b16]]), 99.5)
for h in range(H):
    for row, (b, k, name) in enumerate([(b14, 0, "ORIGINAL seg14"), (b14, 1, "RELABELED seg14"), (b16, 0, "ORIGINAL seg16"), (b16, 1, "RELABELED seg16 (solved)")]):
        im = ax[row, h].imshow(A[b, k, h], cmap="RdBu_r", vmin=-vm, vmax=vm); ax[row, h].set_xticks([]); ax[row, h].set_yticks([])
        if h == 0: ax[row, h].set_ylabel(name, fontsize=10)
        if row == 0: ax[row, h].set_title(f"head {h}", fontsize=10)
fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.01, pad=0.01); fig.suptitle("puzzle #155 — layer-1 coupling per head (blue<0 repel, red>0)", fontsize=12)
fig.savefig(os.path.join(FIG, "v2_C_L1_heads.png"), dpi=100, bbox_inches="tight"); plt.close(fig)
# ---------- Fig D: per-cell received coupling, clear labels
fig, ax = plt.subplots(4, len(idx), figsize=(2.6 * len(idx) + 1, 11))
recv_abs = np.abs(Ah[idx]).sum(-1); recv_net = Ah[idx].sum(-1); vabs = (recv_abs.min(), recv_abs.max()); vnet = np.abs(recv_net).max()
for j, b in enumerate(idx):
    for k in range(2):
        for row, (img, cm, vmin, vmax) in enumerate([(recv_abs[j, k].reshape(9, 9), "viridis", *vabs), (recv_net[j, k].reshape(9, 9), "RdBu_r", -vnet, vnet)]):
            a_ = ax[2 * k + row, j]; im = a_.imshow(img, cmap=cm, vmin=vmin, vmax=vmax); a_.set_xticks([]); a_.set_yticks([])
            for (ii, jj) in zip(*np.where(wrong[b, k].reshape(9, 9))): a_.add_patch(plt.Rectangle((jj - .5, ii - .5), 1, 1, fill=False, ec="red", lw=1.6))
            for (ii, jj) in zip(*np.where(X[k].reshape(9, 9) > 0)): a_.plot(jj, ii, "k.", ms=3)
            if j == 0: a_.set_ylabel(f"{RS[k]}\n" + ("Σ|a| received" if row == 0 else "Σa received (signed)"), fontsize=10)
            if k == 0 and row == 0: a_.set_title(f"seg {segs[j]}" + (" ← SOLVED" if solve_blk and b >= solve_blk else ""), fontsize=11, color=("green" if solve_blk and b >= solve_blk else "k"))
fig.suptitle("puzzle #155 — layer-1 coupling received per cell (9×9). red box = currently wrong cell, dot = given cell", fontsize=12)
fig.savefig(os.path.join(FIG, "v2_D_cellmap.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
# ---------- 수치 표
print("=== 레이어1 헤드별 동료 쌍 평균 a_eff (seg1 끝 / seg14 끝 / seg16 끝)")
print(f"{'head':>4} | {'ORIG s1':>8} {'ORIG s14':>8} {'ORIG s16':>8} | {'RELB s1':>8} {'RELB s14':>8} {'RELB s16':>8}")
for h in range(H):
    v = [A[b, k, h][peer].mean() for k in range(2) for b in (idx[0], b14, b16)]
    print(f"{h:4d} | {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f} | {v[3]:8.4f} {v[4]:8.4f} {v[5]:8.4f}")
print("\n=== 두 런의 레이어1 결합 차이 |Δ|/|A| (헤드평균)")
for s_ in segs: b = last_L1_in_seg(s_); print(f"  seg {s_:2d}: {np.linalg.norm(Ah[b,1]-Ah[b,0])/np.linalg.norm(Ah[b,0]):.4f}")
print("\n=== 원본 런: 틀린 칸 vs 맞은 칸의 수신 결합 (레이어1, 빈칸만)")
blank = X[0] == 0
for s_ in (1, 8, 16):
    b = last_L1_in_seg(s_); ra = np.abs(Ah[b, 0]).sum(1); rn = Ah[b, 0].sum(1); w = wrong[b, 0] & blank; ok = ~wrong[b, 0] & blank
    print(f"  seg {s_:2d}: Σ|a| 틀린칸 {ra[w].mean():.3f} / 맞은칸 {ra[ok].mean():.3f}    Σa 틀린칸 {rn[w].mean():+.3f} / 맞은칸 {rn[ok].mean():+.3f}   (틀린 {w.sum()} 맞은 {ok.sum()})")
print("\n=== 치환 런: 해결 전후 (seg14 끝 → seg16 끝)")
for lab, bb in (("seg14", b14), ("seg16", b16)):
    print(f"  {lab}: 동료 평균 {Ah[bb,1][peer].mean():+.4f}  비동료 평균 {Ah[bb,1][~peer & ~np.eye(81,dtype=bool)].mean():+.4f}  Σa 수신 평균 {Ah[bb,1].sum(1).mean():+.3f}  음수비율(동료) {(A[bb,1][:,peer]<0).mean():.3f}")
print("\n=== 두 런의 예측이 갈라지는 시점: 첫 블록에서 예측이 다른 칸 수 (치환 역변환 후 비교)")
inv = np.zeros(10, dtype=int); inv[d["dm"]] = np.arange(10)
for b in (0, 1, 2, 4, 8, 15, 31, 63):
    p0 = PR[b, 0]; p1 = inv[np.clip(PR[b, 1], 0, 9)]; print(f"  block {b:2d} (seg {b//bps+1}): 다른 칸 {int((p0 != p1).sum())}  원본 틀림 {nwrong[b,0]}  치환 틀림 {nwrong[b,1]}")
print("\n저장:", ", ".join(f"v2_{x}.png" for x in ("A_timeseries", "B_L1_change", "C_L1_heads", "D_cellmap")))
