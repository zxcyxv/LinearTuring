"""끌개 판정: 유한시간 리아푸노프 지수(Benettin) + 장시간 자기상관.
randR 처럼 dt 에 무관한 모델에서는 이것이 연속 동역학계에 대한 진술이 된다."""
import sys, json, os; sys.path.insert(0,'.')
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
import interp
plt.rcParams.update({"figure.dpi":130,"font.size":8,"axes.titlesize":8,"axes.unicode_minus":False})

R_INT, TAU_BURN, TAU_SPAN, RENORM = 64, 4.0, 12.0, 0.5
EPS = 1e-5

@torch.no_grad()
def integrate(m, h, decay, Theta, nsteps, dt, a_fix=None):
    for _ in range(nsteps):
        h = m.phi(h, dt/2); f,*_ = m.field(h, decay, Theta, a_fix); h = h + dt*f; h = m.phi(h, dt/2)
    return h

@torch.no_grad()
def run(tag, nb=64):
    m, cfg = interp.load(tag); x, _ = interp.testset(noisy=cfg["noisy"])
    xb = x[:nb].to(interp.DEV)
    decay, Theta = m.kernel(); dt = 1.0/R_INT
    h0 = m.embed_patches(xb)
    a_fix = m.attn(h0, decay, Theta)[0] if getattr(m, 'freeze_A', False) else None
    h = integrate(m, h0, decay, Theta, int(TAU_BURN*R_INT), dt, a_fix)
    # --- Benettin: 주기적 재정규화로 FTLE 추정 ---
    scale = h.norm() * EPS
    d = torch.randn_like(h); d = d / d.norm() * scale
    h2 = h + d
    nchunk = int(TAU_SPAN / RENORM); nstep = int(RENORM * R_INT); logs = []
    hh, h2h = h.clone(), h2.clone()
    traj = [hh.clone()]
    for _ in range(nchunk):
        hh  = integrate(m, hh,  decay, Theta, nstep, dt, a_fix)
        h2h = integrate(m, h2h, decay, Theta, nstep, dt, a_fix)
        sep = (h2h - hh).norm()
        logs.append(torch.log(sep/scale).item())
        h2h = hh + (h2h - hh) / sep * scale        # 재정규화
        traj.append(hh.clone())
    lam = float(np.sum(logs) / (nchunk * RENORM))
    # --- 장시간 자기상관 ---
    H = torch.stack(traj)                          # [nchunk+1, B, T, d] (RENORM 간격 샘플)
    X = H.reshape(H.shape[0], -1, m.d); Xc = X - X.mean(0, keepdim=True)
    Xn = Xc / (Xc.norm(dim=-1, keepdim=True)+1e-9)
    L = Xn.shape[0]; C = [float((Xn[:L-k]*Xn[k:]).sum(-1).mean()) for k in range(L//2)]
    return {"tag": tag, "ftle": lam, "log_growth": logs, "autocorr": C,
            "renorm_dt": RENORM, "h_norm": float(hh.norm(dim=-1).mean())}

tags = [t for t in sys.argv[1:]] or ["randR"]
res = [run(t) for t in tags]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
for r in res:
    tt = np.arange(1, len(r["log_growth"])+1) * RENORM
    ax[0].plot(tt, np.cumsum(r["log_growth"])/tt, 'o-', ms=3, lw=1.1,
               label=f"{r['tag']}  λ≈{r['ftle']:.2f}")
    ax[1].plot(np.arange(len(r["autocorr"]))*RENORM, r["autocorr"], 'o-', ms=3, lw=1.1, label=r['tag'])
ax[0].axhline(0, ls='--', c='k', lw=.7); ax[0].set_xlabel("적분 시간 τ")
ax[0].set_ylabel("누적 FTLE 추정 λ(τ)"); ax[0].legend(fontsize=6)
ax[0].set_title("리아푸노프 지수: λ>0 이면 카오스")
ax[1].axhline(0, ls='--', c='k', lw=.7); ax[1].set_xlabel("lag τ"); ax[1].set_ylabel("cos<h(τ),h(τ+lag)>")
ax[1].legend(fontsize=6); ax[1].set_title("장시간 자기상관 (dt=1/64)")
fig.suptitle("끌개의 정체 — 연속 흐름에서의 판정", y=1.03); fig.tight_layout()
fig.savefig("figs/attractor.png", bbox_inches="tight")
json.dump(res, open("runs/attractor.json","w"))
for r in res: json.dump(r, open(f"runs/{r['tag']}_attractor.json","w"))
for r in res:
    print(f"[{r['tag']}] FTLE λ = {r['ftle']:+.3f} /τ  → " +
          ("카오스(민감 의존)" if r['ftle']>0.05 else "비카오스(주기/준주기)" if r['ftle']>-0.05 else "수축"),
          f" | 자기상관 lag=1 {r['autocorr'][2]:+.2f}, lag=3 {r['autocorr'][6]:+.2f}")
