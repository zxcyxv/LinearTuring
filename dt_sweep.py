"""dt(이산화)와 τ(적분 시간)를 분리해서 쓸어본다.
  진동이 벡터장의 성질이면  → dt 를 줄여도 같은 τ 에서 계속 진동한다
  진동이 큰 스텝의 산물이면  → dt 를 줄이면 사라지고 고정점으로 이완한다"""
import sys, json; sys.path.insert(0,'.')
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
import interp
plt.rcParams.update({"figure.dpi":130,"font.size":8,"axes.titlesize":8,"axes.unicode_minus":False})

tag = sys.argv[1] if len(sys.argv)>1 else "full"
m, cfg = interp.load(tag); x, y = interp.testset(noisy=cfg["noisy"])
xb, yb = x[:256].to(interp.DEV), y[:256].to(interp.DEV)
TAU = 8.0
out = {}
fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.0))
for Rint in [8, 16, 32, 64, 128]:
    steps = int(TAU * Rint)
    with torch.no_grad():
        _, tr = m(xb, R=Rint, steps=steps, record=True, record_fields=["h"])
    hs = torch.stack(tr["h"])
    hn   = hs.norm(dim=-1).mean(dim=(1,2)).cpu().numpy()
    incr = (hs[1:]-hs[:-1]).norm(dim=-1).mean(dim=(1,2)).cpu().numpy()
    rel  = incr / hn[:-1] * Rint            # ‖dh/dτ‖/‖h‖ : dt 로 정규화한 속도
    tt   = np.arange(len(hn)) / Rint
    ax[0].plot(tt, hn, lw=1.1, label=f"dt=1/{Rint}")
    ax[1].semilogy(tt[:-1], np.maximum(rel,1e-9), lw=1.1, label=f"dt=1/{Rint}")
    # 후반부 τ∈[6,8] 에서의 상대 속도 = 고정점 판정
    late = rel[int(6*Rint):]
    out[Rint] = {"h_end": float(hn[-1]), "speed_late": float(late.mean()),
                 "speed_cv": float(late.std()/(late.mean()+1e-12)),
                 "acc_tau1": float((m.w_cls(hs[Rint]).mean(1).argmax(1)==yb).float().mean()),
                 "acc_end": float((m.w_cls(hs[-1]).mean(1).argmax(1)==yb).float().mean())}
    print(f"dt=1/{Rint:<4d} τ=1 acc {out[Rint]['acc_tau1']:.3f} | τ=8 acc {out[Rint]['acc_end']:.3f} "
          f"| ‖h‖(8)={out[Rint]['h_end']:.2f} | 후반 상대속도 {out[Rint]['speed_late']:.3f}", flush=True)
ax[0].axvline(1.0, ls=":", c="k"); ax[0].set_xlabel("적분 시간 τ"); ax[0].set_ylabel("mean ‖h‖")
ax[0].legend(fontsize=6); ax[0].set_title("상태 노름 (τ 축 정렬)")
ax[1].axvline(1.0, ls=":", c="k"); ax[1].set_xlabel("적분 시간 τ"); ax[1].set_ylabel("‖dh/dτ‖ / ‖h‖")
ax[1].legend(fontsize=6); ax[1].set_title("상대 속도: 0으로 죽으면 고정점")
r_ = sorted(out); ax[2].semilogy(r_, [out[k]["speed_late"] for k in r_], 'o-')
ax[2].set_xscale("log"); ax[2].set_xlabel("1/dt"); ax[2].set_ylabel("τ∈[6,8] 평균 상대속도")
ax[2].set_title("dt→0 에서 진동이 사라지나")
fig.suptitle(f"[{tag}] dt 와 τ 의 분리: 진동은 벡터장인가 이산화인가", y=1.03)
fig.tight_layout(); fig.savefig(f"figs/{tag}_38_dtsweep.png", bbox_inches="tight")
json.dump(out, open(f"runs/{tag}_dtsweep.json","w"))
