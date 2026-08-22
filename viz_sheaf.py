"""채택 레시피(sheaf + Λfull + 경계 W_O contract)의 장 계산 시각화.
Rule 110 k=8, τ=4 → 재귀 32 스텝, 경계 재부호화 r=8,16,24."""
import sys; sys.path.insert(0, '/workspace/LinearTuring')
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import interp, modes as M, fields as FD
from ca_task import step

torch.set_grad_enabled(False)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.titlesize": 10,
                     "axes.unicode_minus": False, "axes.edgecolor": "#cccccc"})

m, cfg = M.load_ca('b110k8_shCL_t4_long')
steps = m.R * cfg['tau']                       # 32
xb, yb = M.ca_batch(cfg, 64)
F, traj, ctx = FD.record(m, m.embed_patches(xb), steps)
d = torch.stack([F['a'][r].sum(-1).sum(1) for r in range(steps)])       # [32,B,T]
rolls = [xb]; cur = xb
for _ in range(cfg['k']): cur = step(cur, cfg['rule']); rolls.append(cur)
truth = torch.stack(rolls).float()                                       # [9,B,T]
corr = lambda u, v: float((lambda a, b: (a*b).sum()/(a.norm()*b.norm()+1e-9))(
    (u.flatten().float()-u.float().mean()), (v.flatten().float()-v.float().mean())))
ci = [corr(d[r], xb) for r in range(steps)]
co = [corr(d[r], yb) for r in range(steps)]
# 중간 정답과의 상관도: 장이 중간 단계 k'를 순서대로 지나가는가
cmid = {kk: [corr(d[r], rolls[kk]) for r in range(steps)] for kk in (2, 4, 6)}

fig = plt.figure(figsize=(13.5, 6.2))
gs = GridSpec(2, 3, width_ratios=[1.0, 1.6, 1.25], wspace=0.28, hspace=0.45)
for row, b in enumerate((3, 7)):
    axT = fig.add_subplot(gs[row, 0])
    axT.imshow(truth[:, b].cpu(), cmap="Greys", aspect="auto", interpolation="nearest")
    axT.set_yticks(range(cfg['k']+1)); axT.set_yticklabels(["입력"]+[f"k={i}" for i in range(1, cfg['k']+1)], fontsize=7)
    axT.set_xticks([]); axT.set_title(f"정답 계산 (Rule 110 k=8, 샘플 {b})", loc="left")
    for sp in axT.spines.values(): sp.set_visible(False)
    axF = fig.add_subplot(gs[row, 1])
    img = d[:, b].cpu().numpy()
    img = (img - img.mean(1, keepdims=True)) / (img.std(1, keepdims=True) + 1e-9)
    v = np.abs(img).max()
    axF.imshow(img, cmap="RdBu", vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    for rb in (8, 16, 24):
        axF.axhline(rb - 0.5, color="#333333", lw=1.2, ls="--", alpha=0.8)
    axF.set_yticks([0, 8, 16, 24, 31]); axF.set_yticklabels(["r=0", "8 ⟵W_O", "16 ⟵W_O", "24 ⟵W_O", "31"], fontsize=7)
    axF.set_xticks([]); axF.set_title("카운트 장 d_t(r) — 점선 = 경계 재부호화", loc="left")
    for sp in axF.spines.values(): sp.set_visible(False)
axC = fig.add_subplot(gs[:, 2])
rr = list(range(steps))
axC.axhline(0, color="#bbbbbb", lw=0.8)
for rb in (8, 16, 24): axC.axvline(rb, color="#333333", lw=0.9, ls="--", alpha=0.5)
axC.plot(rr, ci, color="#c0392b", lw=2)
axC.plot(rr, co, color="#2166ac", lw=2)
greys = {2: "#c8c8c8", 4: "#999999", 6: "#666666"}
for kk, cv in cmid.items():
    axC.plot(rr, cv, color=greys[kk], lw=1.2)
    axC.annotate(f"k={kk}", (rr[np.argmax(cv)], max(cv)), textcoords="offset points",
                 xytext=(0, 5), ha="center", fontsize=7.5, color=greys[kk])
axC.annotate("corr(d, 입력)", (rr[-1], ci[-1]), textcoords="offset points", xytext=(-4, -13),
             ha="right", color="#c0392b", fontsize=9)
axC.annotate("corr(d, 정답 k=8)", (rr[-1], co[-1]), textcoords="offset points", xytext=(-4, 8),
             ha="right", color="#2166ac", fontsize=9)
axC.set_xlabel("재귀 스텝 r (32 = R·τ)"); axC.set_ylabel("상관")
axC.set_title("장이 중간 단계들을 순서대로 통과하는가", loc="left")
axC.grid(alpha=0.25, lw=0.4); axC.set_ylim(-1, 1)
for sp in ("top", "right"): axC.spines[sp].set_visible(False)
fig.suptitle("채택 레시피의 장 계산 — sheaf + Λfull + 경계 W_O(contract), Rule 110 k=8 (전체일치 0.95 포화판)",
             fontsize=12, y=0.99)
fig.savefig("figs/field_computation_sheaf_sat.png", bbox_inches="tight")
print("saved figs/field_computation_sheaf_sat.png")
print("경계 직전/직후 corr(d,정답):", [(rb, round(co[rb-1],3), round(co[rb],3)) for rb in (8,16,24)])
