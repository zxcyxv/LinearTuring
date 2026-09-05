"""퍼즐 #155: 원본(실패) vs 숫자만 치환한 변형(성공) — 풀릴 때까지 블록별 유효 결합 a_eff 를 기록하고 시각화."""
import os, importlib.util, time, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/workspace/LinearTuring"; FIG = os.path.join(ROOT, "2026-09-05/results/figs"); torch.set_grad_enabled(False)
PID = 155
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
def make(bs, loops):
    cfg = dict(ck["cfg"]); cfg.update(batch_size=bs, seq_len=81, num_puzzle_identifiers=1, loops=loops + 1)
    m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); return m
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
# ---- 1. aug_factor.py 와 같은 난수열로 숫자 치환 32개 재현 → 가장 빨리 풀리는 치환 선택
rng = np.random.default_rng(1); dms = [np.concatenate([[0], rng.permutation(9) + 1]) for _ in range(32)]; dms[0] = np.arange(10)
K = 32; m = make(K, 64)
Xa = np.stack([np.where(X0 > 0, dm[X0], 0) for dm in dms]); Ya = np.stack([dm[Y0] for dm in dms])
x = torch.from_numpy(Xa.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Ya.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(K, dtype=torch.int32, device="cuda"))
with torch.device("cuda"): carry = m.initial_carry(batch)
first = np.full(K, -1)
for si in range(64):
    carry, o = m(carry, batch); ok = ((o["logits"].argmax(-1).cpu().numpy() - 1) == Ya).all(-1); first[(first < 0) & ok] = si + 1
best = int(np.argmin(np.where(first > 0, first, 999))); dm = dms[best]
print(f"숫자 치환 32개 중 풀린 것 {(first>0).sum()}개, 가장 빠른 변형 #{best} seg {first[best]}  치환: " + " ".join(f"{d}→{dm[d]}" for d in range(1, 10)), flush=True)
S = int(first[best]) + 2; del m; torch.cuda.empty_cache()
# ---- 2. 두 런을 S 세그 돌리며 블록별 기록
m = make(2, S); I = m.inner
Xb = np.stack([X0, np.where(X0 > 0, dm[X0], 0)]); Yb = np.stack([Y0, dm[Y0]])
x = torch.from_numpy(Xb.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Yb.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(2, dtype=torch.int32, device="cuda"))
rec = dict(a=[], layer=[], pred=[]); orig = I.step
def hooked(L, h, AB, kc, w=None, fresh=None, kcb=None, vprev=None, apply_phi=True):
    li = 0 if L is I.layers[0] else 1
    a = I.attn_xy(I.addr(h, AB), kc)
    hout, w_new = orig(L, h, AB, kc, w, fresh, kcb, vprev, apply_phi)
    lam = torch.sigmoid(L.lam_raw) if I.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(I.config.stdp_lam_fixed))
    a_eff = (1 - lam) * a + lam * w_new
    rec["a"].append(a_eff.float().cpu().numpy()); rec["layer"].append(li)
    h_end = I.phi(I.boundary(L, hout)) if not apply_phi else hout             # post 순서: 블록 끝 상태 재현
    rec["pred"].append((I.w_cls(h_end).argmax(-1).cpu().numpy() - 1))
    return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(S): carry, o = m(carry, batch)
A = np.stack(rec["a"]); LAY = np.array(rec["layer"]); PR = np.stack(rec["pred"])       # A [blk, 2, H, 81, 81], PR [blk, 2, 81]
nb = len(LAY); bps = nb // S
wrong = (PR != Yb[None]).sum(-1)                                                       # [blk, 2]
churn = np.concatenate([np.zeros((1, 2)), (PR[1:] != PR[:-1]).mean(-1)])
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
Ah = A.mean(2)                                                                          # 헤드 평균 [blk,2,81,81]
a_peer = np.array([[Ah[b, k][peer].mean() for k in range(2)] for b in range(nb)]); a_non = np.array([[Ah[b, k][~peer & ~np.eye(81, dtype=bool)].mean() for k in range(2)] for b in range(nb)])
neg_peer = np.array([[(A[b, k][:, peer] < 0).mean() for k in range(2)] for b in range(nb)])
print(f"기록: {nb} 블록 ({S} 세그 × {bps}), 원본 끝 틀린 칸 {wrong[-1,0]}, 변형 끝 틀린 칸 {wrong[-1,1]}", flush=True)
np.savez_compressed(os.path.join(ROOT, "2026-09-05/results/json/attn_viz_155.npz"), A=A.astype(np.float16), layer=LAY, pred=PR, X=Xb, Y=Yb, dm=dm)
lab = ["FAIL run (original digits)", f"SOLVED run (digits renamed, solved at seg {first[best]})"]; col = ["tab:red", "tab:blue"]
# ---- Fig 1: 시계열
fig, ax = plt.subplots(5, 1, figsize=(14, 13), sharex=True); t = np.arange(nb)
for k in range(2):
    ax[0].plot(t, wrong[:, k], color=col[k], label=lab[k]); ax[1].plot(t, churn[:, k], color=col[k])
    for li, ls in ((0, "-"), (1, "--")):
        sel = LAY == li
        ax[2].plot(t[sel], a_peer[sel, k], color=col[k], ls=ls, label=f"{lab[k]} L{li}")
        ax[3].plot(t[sel], a_non[sel, k], color=col[k], ls=ls)
        ax[4].plot(t[sel], neg_peer[sel, k], color=col[k], ls=ls)
for a_ in ax:
    for sgb in range(0, nb, bps): a_.axvline(sgb, color="k", alpha=0.08)
ax[0].set_ylabel("number of WRONG cells (of 81)"); ax[0].legend(fontsize=11); ax[1].set_ylabel("churn = fraction of cells whose\nanswer changed since previous block"); ax[2].set_ylabel("coupling between PEER cells\n(same row/col/box), mean\nsolid = layer 0, dashed = layer 1"); ax[2].legend(fontsize=8)
ax[3].set_ylabel("coupling between NON-peer cells, mean"); ax[4].set_ylabel("fraction of peer couplings\nthat are negative (= pushing apart)"); ax[4].set_xlabel(f"block index  ({bps} blocks = 1 segment; grey vertical lines = segment boundaries)")
fig.suptitle(f"Puzzle #{PID}, same puzzle twice.  FAIL run = original digits.  SOLVED run = digits renamed (2->3, 3->8, 4->2, 5->4, 6->7, 7->5, 8->6)", fontsize=12); fig.tight_layout(); fig.savefig(os.path.join(FIG, "attn155_timeseries.png"), dpi=110); plt.close(fig)
# ---- Fig 2: 결합 히트맵 (L1, 헤드 평균) 시간 변화 + 차이
times = sorted(set([1, bps, 2 * bps, 4 * bps, nb] ) & set(range(1, nb + 1)))
if first[best] > 4: times = sorted(set(times + [int(first[best]) * bps]))
for li in (0, 1):
    idx = [max(i for i in range(nb) if LAY[i] == li and i <= tt) for tt in times]
    fig, ax = plt.subplots(3, len(idx), figsize=(3.2 * len(idx), 9.5))
    vmax = np.percentile(np.abs(Ah[idx]), 99)
    for j, b in enumerate(idx):
        for k in range(2):
            ax[k, j].imshow(Ah[b, k], cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[k, j].set_title(("FAIL run" if k == 0 else "SOLVED run") + f"\nsegment {b//bps+1}, block {b%bps+1}", fontsize=9)
        d = Ah[b, 1] - Ah[b, 0]; ax[2, j].imshow(d, cmap="PuOr", vmin=-vmax, vmax=vmax); ax[2, j].set_title("SOLVED minus FAIL\n(same segment, block)", fontsize=9)
        for a_ in ax[:, j]: a_.set_xticks([]); a_.set_yticks([])
    fig.colorbar(ax[0, -1].images[0], ax=ax[:2, :].ravel().tolist(), fraction=0.012, pad=0.01, label="a_eff:  blue < 0 = push apart,  red > 0 = pull together"); fig.colorbar(ax[2, -1].images[0], ax=ax[2, :].ravel().tolist(), fraction=0.012, pad=0.01, label="difference")
    fig.suptitle(f"Puzzle #{PID}, layer {li}: coupling matrix a_eff[t, n], averaged over heads.  Axes = 81 cells in row-major order (9x9 diagonal blocks = same row; stripes every 9 = same column)", fontsize=11); fig.savefig(os.path.join(FIG, f"attn155_heatmap_L{li}.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
# ---- Fig 3: 칸별 수신 결합 9×9 지도 (L1), 틀린 칸 표시
li = 1; idx = [max(i for i in range(nb) if LAY[i] == li and i <= tt) for tt in times]
fig, ax = plt.subplots(2 * 2, len(idx), figsize=(2.6 * len(idx), 10.5))
for j, b in enumerate(idx):
    for k in range(2):
        recv_abs = np.abs(Ah[b, k]).sum(1).reshape(9, 9); recv_net = Ah[b, k].sum(1).reshape(9, 9)
        wr = (PR[b, k] != Yb[k]).reshape(9, 9)
        for row, (img, ttl, cm) in enumerate([(recv_abs, "total coupling strength received, sum|a|", "viridis"), (recv_net, "net coupling received, sum a\n(blue = net push, red = net pull)", "RdBu_r")]):
            a_ = ax[2 * k + row, j]; a_.imshow(img, cmap=cm, **({} if row == 0 else dict(vmin=-np.abs(img).max(), vmax=np.abs(img).max())))
            for (ii, jj) in zip(*np.where(wr)): a_.add_patch(plt.Rectangle((jj - .5, ii - .5), 1, 1, fill=False, ec="red", lw=1.5))
            for (ii, jj) in zip(*np.where(Xb[k].reshape(9, 9) > 0)): a_.plot(jj, ii, "k.", ms=3)
            a_.set_title(("FAIL run" if k == 0 else "SOLVED run") + f", segment {b//bps+1}\n{ttl}", fontsize=7.5); a_.set_xticks([]); a_.set_yticks([])
fig.colorbar(ax[0, -1].images[0], ax=[ax[0, -1], ax[2, -1]], fraction=0.05, pad=0.02, label="sum|a|"); fig.colorbar(ax[1, -1].images[0], ax=[ax[1, -1], ax[3, -1]], fraction=0.05, pad=0.02, label="sum a")
fig.suptitle(f"Puzzle #{PID}, layer 1: coupling each cell receives, drawn on the 9x9 board.   RED BOX = cell currently answered WRONG.   BLACK DOT = given clue.", fontsize=11); fig.savefig(os.path.join(FIG, "attn155_cellmap_L1.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
print("저장:", ", ".join(os.path.join(FIG, f) for f in ("attn155_timeseries.png", "attn155_heatmap_L0.png", "attn155_heatmap_L1.png", "attn155_cellmap_L1.png")))
# 요약 수치
for k in range(2):
    print(f"{lab[k]}: seg1 끝 틀린 칸 {wrong[bps-1,k]}  seg2 끝 {wrong[2*bps-1,k]}  마지막 {wrong[-1,k]}   동료 a_eff 평균 L1 첫블록 {a_peer[1,k]:+.4f} → 마지막 {a_peer[nb-1,k]:+.4f}   음수비율 {neg_peer[1,k]:.2f} → {neg_peer[nb-1,k]:.2f}")
