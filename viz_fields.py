"""2일차 시각화 — 구조가 보이는 세 그림.
  1. CA: 카운트 장 d_t(r) 의 시공간도 — 입력 표현이 정답 표현으로 회전하는 것이 눈에 보임
  2. MNIST noov: 부호 그래프의 균형 흐름 — 좌절 상태 → 골격 → 분화
  3. 분산관계 심볼 곡면 — 학습이 띠를 어떻게 조각하는가
색 규약(전 그림 일관): 파랑 = 양(인력/성장), 빨강 = 음(척력/감쇠), 중점 = 중립 회색. 진리값 = 무채색.
"""
import sys; sys.path.insert(0, '/workspace/LinearTuring')
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import interp, modes as M, fields as FD, dispersion as D
from ca_task import step

DEV = interp.DEV; torch.set_grad_enabled(False)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.titlesize": 10,
                     "axes.unicode_minus": False, "axes.edgecolor": "#cccccc"})
CM_DIV = "RdBu"      # 낮음=빨강(음) ↔ 높음=파랑(양)

# ══════════ 그림 1: CA — 계산이 장에서 보인다 ══════════
m, cfg = M.load_ca('ca110_k4_full')
xb, yb = M.ca_batch(cfg, 64)
F, traj, ctx = FD.record(m, m.embed_patches(xb), m.R)
d = torch.stack([F['a'][r].sum(-1).sum(1) for r in range(m.R)])          # [R,B,T]
rolls = [xb]; cur = xb
for _ in range(cfg['k']): cur = step(cur, cfg['rule']); rolls.append(cur)
truth = torch.stack(rolls).float()                                        # [k+1,B,T]
corr = lambda u, v: float((lambda a, b: (a*b).sum()/(a.norm()*b.norm()+1e-9))(
    (u.flatten().float()-u.float().mean()), (v.flatten().float()-v.float().mean())))
ci = [corr(d[r], xb) for r in range(m.R)]
co = [corr(d[r], yb) for r in range(m.R)]

fig = plt.figure(figsize=(13.5, 5.6))
gs = GridSpec(2, 3, width_ratios=[1.0, 1.6, 1.15], wspace=0.28, hspace=0.5)
for row, b in enumerate((3, 7)):
    axT = fig.add_subplot(gs[row, 0])
    axT.imshow(truth[:, b].cpu(), cmap="Greys", aspect="auto", interpolation="nearest")
    axT.set_yticks(range(cfg['k']+1)); axT.set_yticklabels(["입력"]+[f"k={i}" for i in range(1, cfg['k']+1)])
    axT.set_xticks([]); axT.set_title(f"정답 계산 (Rule 110, 샘플 {b})", loc="left")
    for sp in axT.spines.values(): sp.set_visible(False)
    axF = fig.add_subplot(gs[row, 1])
    img = d[:, b].cpu().numpy()
    img = (img - img.mean(1, keepdims=True)) / (img.std(1, keepdims=True) + 1e-9)
    v = np.abs(img).max()
    axF.imshow(img, cmap=CM_DIV, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    axF.set_yticks(range(m.R)); axF.set_yticklabels([f"r={r}" for r in range(m.R)])
    axF.set_xticks([]); axF.set_title("모델의 카운트 장  d_t(r)  (행별 표준화)", loc="left")
    for sp in axF.spines.values(): sp.set_visible(False)
axC = fig.add_subplot(gs[:, 2])
rr = list(range(m.R))
axC.axhline(0, color="#bbbbbb", lw=0.8)
axC.plot(rr, ci, color="#c0392b", lw=2, marker="o", ms=4)
axC.plot(rr, co, color="#2166ac", lw=2, marker="o", ms=4)
axC.annotate("corr(d, 입력)", (rr[-1], ci[-1]), textcoords="offset points", xytext=(-6, -14),
             ha="right", color="#c0392b", fontsize=9)
axC.annotate("corr(d, 정답)", (rr[-1], co[-1]), textcoords="offset points", xytext=(-6, 8),
             ha="right", color="#2166ac", fontsize=9)
axC.set_xlabel("재귀 스텝 r"); axC.set_ylabel("상관")
axC.set_title("장이 입력 표현 → 정답 표현으로 회전", loc="left")
axC.grid(alpha=0.25, lw=0.4); axC.set_ylim(-1, 1)
for sp in ("top", "right"): axC.spines[sp].set_visible(False)
fig.suptitle("계산이 장(場)에서 보인다 — Rule 110 k=4 를 8 스텝 재귀가 수행하는 동안의 카운트 장",
             fontsize=12, y=1.00)
fig.savefig("figs/field_computation.png", bbox_inches="tight")
print("saved figs/field_computation.png")

# ══════════ 그림 2: MNIST noov — 부호 그래프의 균형 흐름 ══════════
mn, cfgn = interp.load('noov')
x2, y2 = interp.testset(); b = 0
xb2 = x2[:4].to(DEV)
F2, traj2, ctx2 = FD.record(mn, mn.embed_patches(xb2), mn.R)
G = mn.grid
u = np.repeat(np.arange(G), G); w = np.tile(np.arange(G), G)
rs = [0, 2, 4, 6, 7]
fig, axes = plt.subplots(1, len(rs)+1, figsize=(3.0*(len(rs)+1), 3.4))
axes[0].imshow(x2[b, 0], cmap="Greys"); axes[0].set_title(f"입력 (라벨 {int(y2[b])})", loc="left")
axes[0].set_xticks([]); axes[0].set_yticks([])
prev = None
for ax, r in zip(axes[1:], rs):
    A = F2['a'][r].mean(1)[b]
    As = 0.5*(A+A.T)
    Dg = As.abs().sum(-1); Di = (Dg+1e-9).pow(-0.5)
    L = torch.eye(mn.T, device=DEV) - Di[:, None]*As*Di[None, :]
    ev, evec = torch.linalg.eigh(L)
    frus = float(ev[0])
    # 최소 고유벡터가 상수에 가까우면(부호 순도 >0.85) 다음 것으로 — '이분'을 보여주는 벡터 선택
    part = None
    for j in range(3):
        cand = evec[:, j].cpu().numpy()
        if abs(np.sign(cand).sum()) / len(cand) < 0.85: part = cand; break
    if part is None: part = evec[:, 0].cpu().numpy()
    if prev is not None and np.dot(part, prev) < 0: part = -part
    prev = part
    Asn = As.cpu().numpy()
    idx = np.dstack(np.triu_indices(mn.T, 1))[0]
    vals = Asn[idx[:, 0], idx[:, 1]]
    top = np.argsort(-np.abs(vals))[:70]
    for i in top:
        t_, n_ = idx[i]; v_ = vals[i]
        ax.plot([w[t_], w[n_]], [-u[t_], -u[n_]],
                color=("#2166ac" if v_ > 0 else "#c0392b"),
                lw=0.4 + 2.2*abs(v_)/np.abs(vals).max(), alpha=0.55, zorder=1)
    s = np.abs(part); s = s/s.max()
    ax.scatter(w, -u, c=part, cmap=CM_DIV, vmin=-np.abs(part).max(), vmax=np.abs(part).max(),
               s=60 + 340*s, zorder=2, edgecolors="white", linewidths=0.6)
    ax.set_title(f"r={r}   좌절도 {frus:.2f}", loc="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values(): sp.set_visible(False)
fig.suptitle("부호 그래프의 균형 흐름 (noov) — 좌절 상태 → 공유 골격(r≈6) → 분화 · "
             "간선: 파랑=인력, 빨강=척력 · 노드색 = 이분 분할", fontsize=12, y=1.02)
fig.savefig("figs/balance_flow.png", bbox_inches="tight")
print("saved figs/balance_flow.png")

# ══════════ 그림 3: 분산관계 심볼 곡면 ══════════
hdir, rad, _ = D.get_hdir(mn, cfgn)
wgt = D.zhat_weights(mn, hdir/hdir.norm()*rad)
K = 161; ks = torch.linspace(-np.pi, np.pi, K, device=DEV)
kx = ks.view(1, 1, K, 1); ky = ks.view(1, 1, 1, K)
reT, imT = D.symbol_inf(mn, wgt, kx, ky)
mu = D.untrained(cfgn)
wgtU = D.zhat_weights(mu, hdir/hdir.norm()*rad)
reU, _ = D.symbol_inf(mu, wgtU, kx, ky)
ReT, ReU, ImT = reT.sum(0).cpu(), reU.sum(0).cpu(), imT.sum(0).abs().cpu()
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
ext = [-np.pi, np.pi, -np.pi, np.pi]
for ax, Z, ttl, div in ((axes[0], ReT, "Re Â(k) — 학습", True),
                        (axes[1], ReU, "Re Â(k) — 미학습", True),
                        (axes[2], ImT, "|Im Â(k)| — 학습 (진동수)", False)):
    Zn = Z.numpy()
    if div:
        v = np.percentile(np.abs(Zn), 99)          # 극단 피크 포화 — 띠 구조가 보이게
        im_ = ax.imshow(np.clip(Zn, -v, v), cmap=CM_DIV, vmin=-v, vmax=v, extent=ext, origin="lower")
        ax.contour(Zn, levels=[0], colors="#555555", linewidths=0.8, extent=ext)
    else:
        v = np.percentile(Zn, 99)
        im_ = ax.imshow(np.clip(Zn, 0, v), cmap="Blues", vmax=v, extent=ext, origin="lower")
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(0.905*np.cos(th), 0.905*np.sin(th), ls="--", lw=1.0, color="#333333", alpha=0.7)
    if ttl.startswith("Re Â(k) — 학습"):
        ax.annotate("실측 |k*|=0.91", (0.64, 0.75), fontsize=8, color="#333333")
    ax.set_title(ttl, loc="left"); ax.set_xlabel("k_u"); ax.set_ylabel("k_w")
    fig.colorbar(im_, ax=ax, shrink=0.85)
fig.suptitle("무한격자 어텐션 심볼 (noov, 99% 포화 스케일) — 성장 섬(파랑)의 위치는 θ, 부호는 cos ψ, 진동수는 sin ψ · 점선 = 실측 최대성장 파수",
             fontsize=11.5, y=1.02)
fig.savefig("figs/dispersion_symbol.png", bbox_inches="tight")
print("saved figs/dispersion_symbol.png")
