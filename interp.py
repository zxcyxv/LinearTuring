"""
Model 1 해석가능성 실험 — 사양 §3 전체.
  3.1 상태장 영상 (증분/막전위 d_t/위상장)
  3.2 부호형 수용장 + 채널별 Gabor + (α,θu,θw) 산점도
  3.3 위상평면 (PCA 투영 벡터장 + 널클라인)
  3.4 R 외삽 (학습 R=8 → 추론 R=64) + 흡수구 반경 대조
  3.5 입력 대비 c 에 대한 분기 다이어그램
  3.6 Anytime 정확도 곡선
  3.7 ψ 히스토그램의 학습 중 변화 (대칭 함정 감시, §2.4)
"""
import argparse, json, math, os
import numpy as np
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import koreanize_matplotlib  # 한글 라벨
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model1 import Model1

FIG = "/workspace/mnist_model1/figs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.titlesize": 8, "axes.unicode_minus": False})

def load(tag):
    ck = torch.load(f"/workspace/mnist_model1/runs/{tag}.pt", map_location=DEV, weights_only=False)
    a = ck["args"]
    m = Model1(d=a["d"], H=a["H"], R=a["R"], gamma_init=a["gamma"], alpha_init=a["alpha"], eps=a["eps"],
               use_ov=not a["no_ov"], lam_mode=a["lam_mode"], orth_wc=not a["no_orth"],
               use_bias_v=not a["no_bias_v"], psi_zero=a.get("psi_zero", False),
               learn_gamma=not a.get("fix_gamma", False), freeze_A=a.get("freeze_A", False)).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, a

def testset(noisy=False, n=2048):
    tf = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if noisy: tf.append(transforms.Lambda(lambda t: t + 0.6 * torch.randn_like(t)))
    ds = datasets.MNIST("/workspace/data", False, download=True, transform=transforms.Compose(tf))
    dl = DataLoader(ds, 256, shuffle=False, num_workers=2)
    xs, ys = [], []
    for x, y in dl:
        xs.append(x); ys.append(y)
        if sum(t.shape[0] for t in xs) >= n: break
    return torch.cat(xs)[:n], torch.cat(ys)[:n]

def g(v, G=7):  # 벡터 → 7×7 격자
    return v.reshape(G, G)

# ---------------------------------------------------------------- 3.1
@torch.no_grad()
def fig_state_field(m, x, y, tag, idx=0, head=0, chan=0):
    _, tr = m(x[idx:idx+1].to(DEV), R=m.R, record=True)
    R = m.R
    h0 = tr["h"][0]
    rows, cols = 3, R + 1
    fig, ax = plt.subplots(rows, cols, figsize=(1.05 * cols, 3.4))
    for r in range(R + 1):
        inc = (tr["h"][r] - h0)[0].norm(dim=-1).cpu().numpy()
        ax[0, r].imshow(g(inc), cmap="magma"); ax[0, r].set_title(f"r={r}")
        if r < R:
            d = tr["d"][r][0].sum(0).cpu().numpy()           # Σ_m d_t^(m) : 막전위 유사물
            v = np.abs(d).max() + 1e-9
            ax[1, r].imshow(g(d), cmap="coolwarm", vmin=-v, vmax=v)
            ph = tr["zphase"][r][0, :, head, chan].cpu().numpy()
            amp = tr["znorm"][r][0, :, head].cpu().numpy(); amp = amp / (amp.max() + 1e-9)
            hsv = np.stack([g((ph % (2*np.pi)) / (2*np.pi)), np.ones((7,7)), g(amp)], -1)
            ax[2, r].imshow(hsv_to_rgb(hsv))
        else:
            ax[1, r].axis("off"); ax[2, r].axis("off")
    for a_ in ax.ravel(): a_.set_xticks([]); a_.set_yticks([])
    ax[0,0].set_ylabel("‖h(r)-h(0)‖", fontsize=7)
    ax[1,0].set_ylabel("d_t (막전위)", fontsize=7)
    ax[2,0].set_ylabel(f"arg z (h{head}c{chan})", fontsize=7)
    fig.suptitle(f"[{tag}] 3.1 상태장 진화 (label={y[idx].item()})", y=0.99)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_31_statefield.png"); plt.close(fig)

# ---------------------------------------------------------------- 3.2
@torch.no_grad()
def attn_per_channel(m, h):
    """a_{tn,j}^{(m)} : [B,H,p,T,T] — 채널 분해된 부호형 수용장"""
    A, B_ = m.W_C()
    x = torch.einsum('btd,hjd->bthj', h, A); yy = torch.einsum('btd,hjd->bthj', h, B_)
    nrm = (x.pow(2)+yy.pow(2)).sum(-1, keepdim=True).sqrt()
    x = x/(nrm+m.eps); yy = yy/(nrm+m.eps)
    decay, Th = m.kernel(); Dc, Ds = decay*torch.cos(Th), decay*torch.sin(Th)
    return (torch.einsum('bthj,bnhj,hjtn->bhjtn', x, x, Dc)
            + torch.einsum('bthj,bnhj,hjtn->bhjtn', yy, yy, Dc)
            + torch.einsum('bthj,bnhj,hjtn->bhjtn', x, yy, Ds)
            - torch.einsum('bthj,bnhj,hjtn->bhjtn', yy, x, Ds))

@torch.no_grad()
def fig_receptive(m, x, y, tag, idx=0, r_at=0, qt=24):
    _, tr = m(x[idx:idx+1].to(DEV), R=m.R, record=True)
    h = tr["h"][r_at]
    a = tr["a"][r_at][0]                      # [H,T,T]
    aj = attn_per_channel(m, h)[0]            # [H,p,T,T]
    H, p = m.H, m.p
    fig = plt.figure(figsize=(1.05*(H+1), 1.15*(1+min(p,4))))
    gs = fig.add_gridspec(1+min(p,4), H+1)
    ax = fig.add_subplot(gs[0,0]); ax.imshow(x[idx,0].cpu(), cmap="gray"); ax.set_title("input", fontsize=7)
    ax.plot([ (qt%7)*4+1.5 ],[ (qt//7)*4+1.5 ], 'r+', ms=8); ax.set_xticks([]); ax.set_yticks([])
    for hh in range(H):
        A = a[hh, qt].cpu().numpy(); v = np.abs(A).max()+1e-9
        ax = fig.add_subplot(gs[0, hh+1]); ax.imshow(g(A), cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"head {hh}  Σ_j", fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        for j in range(min(p,4)):
            Aj = aj[hh, j, qt].cpu().numpy(); vj = np.abs(Aj).max()+1e-9
            ax = fig.add_subplot(gs[1+j, hh+1]); ax.imshow(g(Aj), cmap="RdBu_r", vmin=-vj, vmax=vj)
            ax.set_xticks([]); ax.set_yticks([])
            if hh == 0: ax.set_ylabel(f"ch {j}", fontsize=6)
    fig.suptitle(f"[{tag}] 3.2 부호형 수용장 a_tn (query t={qt}, r={r_at})", y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_32_receptive.png"); plt.close(fig)

    # 필터뱅크 커버리지: (α_j, θ_j^u, θ_j^w)
    al = m.alpha.detach().cpu().numpy().ravel()
    th = m.theta.detach().cpu().numpy().reshape(-1,2)
    ps = ((m.psi.detach().cpu().numpy().ravel()+np.pi)%(2*np.pi))-np.pi
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.7))
    sc = ax[0].scatter(th[:,0], th[:,1], c=1.0/al, s=28, cmap="viridis")
    ax[0].set_xlabel("θ_u"); ax[0].set_ylabel("θ_w"); ax[0].set_title("2D 파수 벡터 (색=수용장 폭 1/α)")
    ax[0].axhline(0,lw=.4,c='k'); ax[0].axvline(0,lw=.4,c='k'); plt.colorbar(sc, ax=ax[0])
    ax[1].scatter(np.linalg.norm(th,axis=1), 1.0/al, c=np.abs(ps), cmap="twilight", s=28)
    ax[1].set_xlabel("‖θ‖ (선호 주파수)"); ax[1].set_ylabel("1/α (수용장 폭)"); ax[1].set_title("주파수–대역폭 커버리지")
    ax[2].hist(ps, bins=24, color="tab:purple"); ax[2].set_xlabel("ψ"); ax[2].set_title("ψ 분포 (0 = 대칭 함정)")
    fig.suptitle(f"[{tag}] 3.2 학습된 Gabor 필터뱅크 파라미터", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_32_filterbank.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 3.3
@torch.no_grad()
def fig_phaseplane(m, x, tag, nb=256, R=None):
    R = R or m.R
    _, tr = m(x[:nb].to(DEV), R=R, record=True)
    Hs = torch.stack(tr["h"][:-1])              # [R,B,T,d] (f 와 짝을 맞춤)
    Fs = torch.stack(tr["f"])                   # [R,B,T,d]
    X = Hs.reshape(-1, m.d); Xc = X - X.mean(0)
    U, S, V = torch.pca_lowrank(Xc, q=4)
    P = V[:, :2]                                # [d,2]
    pts = (Xc @ P).cpu().numpy()
    vec = (Fs.reshape(-1, m.d) @ P).cpu().numpy()
    # 격자 binning 으로 평균 벡터장 추정
    n = 26
    xi = np.linspace(np.percentile(pts[:,0],0.5), np.percentile(pts[:,0],99.5), n)
    yi = np.linspace(np.percentile(pts[:,1],0.5), np.percentile(pts[:,1],99.5), n)
    ix = np.clip(np.digitize(pts[:,0], xi)-1, 0, n-1); iy = np.clip(np.digitize(pts[:,1], yi)-1, 0, n-1)
    U1 = np.full((n,n), np.nan); U2 = np.full((n,n), np.nan); cnt = np.zeros((n,n))
    su = np.zeros((n,n)); sv = np.zeros((n,n))
    np.add.at(cnt, (iy,ix), 1); np.add.at(su, (iy,ix), vec[:,0]); np.add.at(sv, (iy,ix), vec[:,1])
    ok = cnt >= 8
    U1[ok] = su[ok]/cnt[ok]; U2[ok] = sv[ok]/cnt[ok]
    XG, YG = np.meshgrid(xi, yi)
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.6))
    ax[0].hexbin(pts[:,0], pts[:,1], gridsize=45, cmap="Greys", bins="log")
    ax[0].quiver(XG, YG, U1, U2, np.hypot(U1,U2), cmap="viridis", scale_units="xy", angles="xy")
    for b in range(6):   # 개별 토큰 궤적
        t = np.random.randint(0, m.T)
        trj = ((Hs[:, b, t, :] - X.mean(0)) @ P).cpu().numpy()
        ax[0].plot(trj[:,0], trj[:,1], '-o', ms=1.8, lw=.8, alpha=.9)
    ax[0].set_title("PCA 평면 벡터장 f + 토큰 궤적"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    c1 = ax[1].contour(XG, YG, np.nan_to_num(U1, nan=1e3), levels=[0], colors="tab:red")
    c2 = ax[1].contour(XG, YG, np.nan_to_num(U2, nan=1e3), levels=[0], colors="tab:blue")
    ax[1].hexbin(pts[:,0], pts[:,1], gridsize=45, cmap="Greys", bins="log", alpha=.45)
    ax[1].set_title("널클라인  f·PC1=0 (빨강) / f·PC2=0 (파랑)\nFHN 이면 3차(N자) 곡선")
    ax[1].set_xlabel("PC1")
    fig.suptitle(f"[{tag}] 3.3 위상평면 · 널클라인", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_33_phaseplane.png", bbox_inches="tight"); plt.close(fig)
    return float(S[0]/S.sum())

# ---------------------------------------------------------------- 3.4 / 3.6
@torch.no_grad()
def fig_extrapolate(m, x, y, tag, Rmax=64, nb=512):
    """§3.4 R 외삽 = **스텝 크기 dt=1/R_train 고정, 스텝 수만 8→Rmax**.
    (dt 까지 같이 줄이면 총 시간 τ=1 이 불변이라 '외삽'이 아니라 적분 정밀화가 된다.
     그 정밀화 테스트는 fig_refine 에서 따로 본다.)"""
    xb, yb = x[:nb].to(DEV), y[:nb].to(DEV)
    _, tr = m(xb, R=m.R, steps=Rmax, record=True, record_fields=["h"])
    hs = torch.stack(tr["h"])                        # [Rmax+1,B,T,d]
    hmax = hs.norm(dim=-1).amax(dim=(1,2)).cpu().numpy()
    hmean = hs.norm(dim=-1).mean(dim=(1,2)).cpu().numpy()
    incr = (hs[1:]-hs[:-1]).norm(dim=-1).mean(dim=(1,2)).cpu().numpy()
    accs = [ (m.w_cls(hs[r]).mean(1).argmax(1)==yb).float().mean().item() for r in range(Rmax+1) ]
    rad, lmax, kap, gam = m.absorbing_radius()
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
    ax[0].plot(hmax, lw=1.2, label="max_t ‖h_t‖"); ax[0].plot(hmean, lw=1.0, label="mean_t ‖h_t‖")
    ax[0].axhline(rad, ls="--", c="r", label=f"예측 흡수구 √((λ+2κ)/γ)={rad:.1f}")
    ax[0].axvline(m.R, ls=":", c="k", label=f"학습 R={m.R}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("스텝 r (dt=1/%d 고정)" % m.R); ax[0].legend(fontsize=6)
    ax[0].set_title("3.4 시간 외삽: 유계성")
    ax[1].semilogy(np.maximum(incr,1e-12), lw=1.2); ax[1].axvline(m.R, ls=":", c="k")
    ax[1].set_xlabel("스텝 r"); ax[1].set_ylabel("mean ‖h(r+1)-h(r)‖")
    ax[1].set_title("증분 → 0: 고정점 / 진동 유지: 한계주기")
    ax[2].plot(accs, lw=1.2); ax[2].axvline(m.R, ls=":", c="k")
    ax[2].set_xlabel("스텝 r"); ax[2].set_ylabel("accuracy"); ax[2].set_title("3.6 Anytime 정확도 (공유 W_cls)")
    fig.suptitle(f"[{tag}] 3.4/3.6  κ={kap:.1f}, γ={gam:.3f}, λmax(Λsym)={lmax:.2f}", y=1.03)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_34_extrapolate.png", bbox_inches="tight"); plt.close(fig)
    # 후반 증분의 진동성: 고정점이면 단조 감소, 한계주기면 비감소 진동
    tail = incr[max(len(incr)//2,1):]
    rel = float(tail.mean() / (hmean[len(hmean)//2:].mean() + 1e-12))   # 증분/상태 비
    verdict = ("발산" if hmax[-1] > 5*max(hmax[m.R],1e-9) else
               "고정점 수렴" if rel < 0.01 else
               "지속 운동(비고정점)" if rel > 0.05 else "느린 이완")
    return {"h_max": hmax.tolist(), "h_mean": hmean.tolist(), "increment": incr.tolist(),
            "anytime_acc": accs, "absorb_radius": rad, "kappa": kap, "gamma": gam,
            "lmax": lmax, "verdict": verdict, "incr_over_h": rel}

@torch.no_grad()
def fig_refine(m, x, y, tag, Rs=(2,4,8,12,16,24,32,48,64,96), nb=1024):
    """적분 정밀화 테스트: 총 시간 τ=1 을 유지한 채 dt=1/R 만 줄인다.
    학습된 이산 사상이 연속 흐름의 근사인지, 아니면 큰 스텝 자체를 이용하는지 판정."""
    xb, yb = x[:nb].to(DEV), y[:nb].to(DEV)
    accs = []
    for R in Rs:
        _, tr = m(xb, R=R, steps=R, record=True, record_fields=["h"])
        hT = tr["h"][-1]
        accs.append((m.w_cls(hT).mean(1).argmax(1) == yb).float().mean().item())
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax[0].plot(list(Rs), accs, 'o-'); ax[0].axvline(m.R, ls=":", c="k", label=f"학습 R={m.R}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("적분 스텝수 R (τ=1 고정, dt=1/R)")
    ax[0].set_ylabel("accuracy"); ax[0].legend(fontsize=6)
    ax[0].set_title("3.4b 적분 정밀화 (같은 ODE, 더 잘게)")
    ax[1].axis("off")
    ax[1].text(0.02, 0.5, "정확도가 R↑ 에서 유지 → 학습된 것이 연속 벡터장\n"
                          "R↑ 에서 붕괴 → 큰 스텝(dt=1/8) 자체가 계산의 일부\n"
                          f"\n측정: R=8 {accs[list(Rs).index(8)]:.3f} → R={max(Rs)} {accs[-1]:.3f}",
               fontsize=7, va="center")
    fig.suptitle(f"[{tag}] 3.4b 이산화 의존성", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_34b_refine.png", bbox_inches="tight"); plt.close(fig)
    return {"Rs": list(Rs), "acc": accs}


# ---------------------------------------------------------------- 3.4c 주기성
@torch.no_grad()
def fig_periodicity(m, x, tag, steps=160, nb=64, burn=None):
    """증분이 0으로 안 죽을 때, 그것이 한계주기(닫힌 궤도)인지 비주기 배회인지 판정."""
    burn = burn if burn is not None else steps // 2
    _, tr = m(x[:nb].to(DEV), R=m.R, steps=steps, record=True, record_fields=["h"])
    H = torch.stack(tr["h"])[burn:]                    # [L,B,T,d]
    L = H.shape[0]
    X = H.reshape(L, -1, m.d)
    Xc = X - X.mean(0, keepdim=True)                   # 시간평균 제거
    Xn = Xc / (Xc.norm(dim=-1, keepdim=True) + 1e-9)
    lags = range(0, L // 2)
    C = [ (Xn[:L-k] * Xn[k:]).sum(-1).mean().item() for k in lags ]
    C = np.array(C)
    neg = np.where(C < 0)[0]                          # 첫 영교차 이후의 첫 국소최대 = 주기
    per = int(np.argmax(C[neg[0]:]) + neg[0]) if len(neg) else 0
    # 스칼라 관측량의 파워 스펙트럼
    u = torch.randn(m.d, device=DEV); u = u / u.norm()
    s_ = (Xc @ u).cpu().numpy()                        # [L, N]
    S = np.abs(np.fft.rfft(s_ - s_.mean(0), axis=0))**2
    freq = np.fft.rfftfreq(L, d=1.0)
    Pm = S.mean(1)
    lo = np.searchsorted(freq, 1.0/40)                # 40스텝보다 느린 표류는 제외
    fpk = freq[lo:][np.argmax(Pm[lo:])]
    # PCA 평면 위 궤도
    U_, S_, V_ = torch.pca_lowrank(Xc.reshape(-1, m.d), q=3)
    P2 = V_[:, :2]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
    ax[0].plot(list(lags), C, lw=1.2); ax[0].axhline(0, lw=.5, c='k')
    ax[0].axvline(per, ls=':', c='r', label=f"최대 상관 lag={per}")
    ax[0].set_xlabel("lag k (스텝)"); ax[0].set_ylabel("cos<h(r), h(r+k)>")
    ax[0].legend(fontsize=6); ax[0].set_title("3.4c 자기상관: 주기성 판정")
    ax[1].semilogy(freq[1:], Pm[1:], lw=1.0); ax[1].axvline(fpk, ls=':', c='r',
        label=f"주 진동수 {fpk:.3f}/step (주기 {1/fpk:.1f})")
    ax[1].set_xlabel("주파수 (1/스텝)"); ax[1].set_ylabel("power"); ax[1].legend(fontsize=6)
    ax[1].set_title("파워 스펙트럼")
    for i in range(8):
        j = np.random.randint(0, Xc.shape[1])
        o = (Xc[:, j, :] @ P2).cpu().numpy()
        ax[2].plot(o[:, 0], o[:, 1], '-', lw=.8, alpha=.85)
    ax[2].set_title("끌개 궤도 (PCA, 시간평균 제거)\n닫힌 고리 = 한계주기")
    ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2")
    fig.suptitle(f"[{tag}] 3.4c 유계 지속 운동의 정체", y=1.03)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_34c_periodicity.png", bbox_inches="tight"); plt.close(fig)
    return {"autocorr": C.tolist(), "peak_lag": per, "peak_freq": float(fpk),
            "peak_corr": float(C[per]) if per else None}

# ---------------------------------------------------------------- 3.5
@torch.no_grad()
def fig_bifurcation(m, x, tag, Rmax=96, nb=64, cs=None):
    cs = cs if cs is not None else np.concatenate([np.linspace(0.1,2.0,20), np.linspace(2.2,6.0,20)])
    xb = x[:nb].to(DEV)
    decay, Theta = m.kernel()
    lo, hi, fin = [], [], []
    for c in cs:
        h = m.embed_patches(xb) * float(c)
        a_fix = m.attn(h, decay, Theta)[0] if getattr(m, 'freeze_A', False) else None
        tail = []
        for r in range(Rmax):
            h = m.phi(h, 0.5/m.R)
            f, *_ = m.field(h, decay, Theta, a_fix)
            h = h + f/m.R
            h = m.phi(h, 0.5/m.R)
            if r >= Rmax-32: tail.append(h.norm(dim=-1).mean().item())
        tail = np.array(tail); lo.append(tail.min()); hi.append(tail.max()); fin.append(tail[-1])
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0))
    ax[0].fill_between(cs, lo, hi, alpha=.35, label="끌개 진폭 범위 (마지막 32스텝)")
    ax[0].plot(cs, fin, 'k-', lw=1, label="최종값")
    ax[0].set_xlabel("입력 대비 c"); ax[0].set_ylabel("mean ‖h_t‖"); ax[0].legend(fontsize=6)
    ax[0].set_title("3.5 분기 다이어그램")
    amp = np.array(hi)-np.array(lo)
    ax[1].semilogy(cs, np.maximum(amp,1e-12), 'r-')
    ax[1].set_xlabel("입력 대비 c"); ax[1].set_ylabel("max-min (진동 진폭)")
    ax[1].set_title("진동 진폭 > 0 → 한계주기 (Hopf)")
    fig.suptitle(f"[{tag}] 3.5 입력 강도에 대한 끌개 분기", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_35_bifurcation.png", bbox_inches="tight"); plt.close(fig)
    return {"c": cs.tolist(), "lo": lo, "hi": hi, "final": fin}

# ---------------------------------------------------------------- 3.7
def fig_psi_history(tag):
    log = json.load(open(f"/workspace/mnist_model1/runs/{tag}_log.json"))
    steps = log["steps"]
    if not steps: return {}
    P = np.array([s["psi"] for s in steps]); st = [s["step"] for s in steps]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.8))
    for k, lab in [(0,"init"), (len(st)//2,"mid"), (len(st)-1,"final")]:
        ax[0].hist(((P[k]+np.pi)%(2*np.pi))-np.pi, bins=20, alpha=.55, label=f"{lab} (step {st[k]})")
    ax[0].set_xlabel("ψ"); ax[0].legend(fontsize=6); ax[0].set_title("3.7 ψ 히스토그램 (0 수렴 = 대칭 함정)")
    ax[1].plot(st, np.abs(np.sin(P)).mean(1), label="mean |sin ψ|  (비대칭성 지표)")
    ax[1].plot(st, [s["gamma"] for s in steps], label="γ")
    ax[1].plot(st, np.array([s["alpha"] for s in steps]).mean(1), label="mean α")
    ax[1].set_xlabel("step"); ax[1].legend(fontsize=6); ax[1].set_title("파라미터 추이")
    ax[2].plot(st, [s["h_max"][-1] for s in steps], label="max_t‖h_t^(R)‖")
    ax[2].plot(st, [s["rho"][-1] for s in steps], label="ρ^(R)=‖ΠAΠ‖₂")
    ax[2].plot(st, [s["absorb_R"] for s in steps], '--', label="흡수구 반경(예측)")
    ax[2].set_yscale("log"); ax[2].set_xlabel("step"); ax[2].legend(fontsize=6); ax[2].set_title("안정성 지표")
    fig.suptitle(f"[{tag}] 3.7 학습 중 로깅", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}_37_training.png", bbox_inches="tight"); plt.close(fig)
    return {"psi_final_abs_sin": float(np.abs(np.sin(P[-1])).mean()),
            "psi_init_abs_sin": float(np.abs(np.sin(P[0])).mean())}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full")
    ap.add_argument("--idx", type=int, default=0); ap.add_argument("--Rmax", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    m, cfg = load(a.tag)
    x, y = testset(noisy=cfg["noisy"])
    out = {"tag": a.tag, "cfg": cfg}
    fig_state_field(m, x, y, a.tag, idx=a.idx)
    fig_receptive(m, x, y, a.tag, idx=a.idx)
    out["pc1_var"] = fig_phaseplane(m, x, a.tag)
    out["extrap"] = fig_extrapolate(m, x, y, a.tag, Rmax=a.Rmax)
    out["refine"] = fig_refine(m, x, y, a.tag)
    out["period"] = fig_periodicity(m, x, a.tag)
    out["bifurcation"] = fig_bifurcation(m, x, a.tag)
    out.update(fig_psi_history(a.tag))
    json.dump(out, open(f"/workspace/mnist_model1/runs/{a.tag}_interp.json", "w"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("extrap","bifurcation","cfg")}, indent=1))
    e = out["extrap"]
    print(f"anytime acc r=1,2,4,8,16,32,64: " + ", ".join(f"{e['anytime_acc'][i]:.3f}" for i in [1,2,4,8,16,32,64]))
    print(f"max||h|| at r=8/32/64: {e['h_max'][8]:.2f}/{e['h_max'][32]:.2f}/{e['h_max'][64]:.2f}  (bound {e['absorb_radius']:.1f})  판정: {e['verdict']}")
    print("주기성:", out["period"]["peak_lag"], "스텝, 상관", out["period"]["peak_corr"])
    print("refine acc:", [f"R={R}:{a_:.3f}" for R, a_ in zip(out['refine']['Rs'], out['refine']['acc'])])

if __name__ == "__main__":
    main()
