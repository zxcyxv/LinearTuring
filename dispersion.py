"""분산관계 Re λ(k) — 균일 상태 주변 선형화 (SH/튜링형 불안정 판정).

균일 상태 H = 1_T ⊗ h̄ᵀ 에서 어텐션은 Δ 만의 함수가 된다 (정확히 병진불변 합성곱):
    a(Δ) = Σ_j w_j e^{-α_j‖Δ‖₁} cos(ψ_j + θ_j·Δ),   w_j = |ẑ_j|²  (h̄ 방향만의 함수)

[이론] 무한격자 심볼 (S_α(q)=sinh α/(cosh α-cos q)):
    Re Â(k) = Σ_j w_j·½cos ψ_j [S_j(θ_j-k)+S_j(θ_j+k)]   ← 성장률, k≈±θ_j 에 봉우리
    Im Â(k) = Σ_j w_j·½sin ψ_j [S_j(θ_j-k)-S_j(θ_j+k)]   ← ψ=0 이면 0 → 명제 7 재현
  단 1/α ≫ 격자크기면 무한격자합이 유효하지 않으므로 위치 예측용으로만 쓴다.

[실측] 유한 T×T 연산자를 정확히 대각화한다. 두 가지:
    M_frozen : A 를 상수로 본 선형화 (= 고전 RD/튜링 분석의 대상)
    J_full   : 참 야코비안 (autograd) — ∂A/∂h·h 항 포함 (준선형/cross-diffusion 성분)
  고유벡터의 2D FFT 파워로 파수 |k| 를 배정해 Re λ vs |k| 를 그린다.

판정: max_k Re λ 가 k=0 인가 유한 k 인가.
"""
import sys, math
sys.path.insert(0, '/workspace/LinearTuring')
import numpy as np, torch
import interp
from model1 import Model1

DEV = interp.DEV
SCR = '/tmp/claude-0/-workspace-LinearTuring/ac978981-cfca-4f33-817b-3ad8423e9e35/scratchpad'


# ---------------------------------------------------------------- 닫힌 형태 (띠 위치 예측)
def S_alpha(a, q): return torch.sinh(a) / (torch.cosh(a) - torch.cos(q))

def zhat_weights(m, hbar):
    A, B = m.W_C()
    x = torch.einsum('d,hjd->hj', hbar, A); y = torch.einsum('d,hjd->hj', hbar, B)
    n = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()
    x, y = x / (n + m.eps), y / (n + m.eps)
    return x.pow(2) + y.pow(2)                                   # [H,p]

def symbol_inf(m, w, kx, ky):
    al, psi = m.alpha[..., None, None], m.psi[..., None, None]
    tu, tw = m.theta[..., 0, None, None], m.theta[..., 1, None, None]
    Sm = S_alpha(al, tu - kx) * S_alpha(al, tw - ky)
    Sp = S_alpha(al, tu + kx) * S_alpha(al, tw + ky)
    W = w[..., None, None]
    return (W * 0.5 * torch.cos(psi) * (Sm + Sp)).sum(1), (W * 0.5 * torch.sin(psi) * (Sm - Sp)).sum(1)


# ---------------------------------------------------------------- 유한격자 정확 연산자
def ov_blocks(m):
    OV = m.OV()
    out = []
    for h in range(m.H):
        P = torch.zeros(m.dh, m.d, device=OV.device)
        P[:, h * m.dh:(h + 1) * m.dh] = torch.eye(m.dh, device=OV.device)
        out.append(OV[h] @ P)
    return torch.stack(out)                                       # [H,d,d]

def Lam(m):
    if m.lam_mode == 'full': return m.lam
    if m.lam_mode == 'diag': return torch.diag(m.lam)
    return torch.zeros(m.d, m.d, device=DEV)

def vfield(m, H, decay, Theta, AB):
    f, *_ = m.field(H, decay, Theta, None, AB)
    return f - m.gamma * H.pow(2).sum(-1, keepdim=True) * H

def uniform_fixed_point(m, hdir, r0, tries=24):
    """균일 상태가 정지하도록 h_bar 를 완화 (뉴턴법).
    g(h_bar) = mean_t f_t(1_T x h_bar) = 0 을 푼다.
    유한격자라 토큰별 d_t 가 달라 평균장으로 잡는다.
    여러 초기 반경에서 시도해 잔차가 가장 작은 해를 취한다."""
    decay, Theta = m.kernel(); AB = m.W_C()
    u = hdir / hdir.norm()

    def gfun(hb):
        H = hb.view(1, 1, -1).expand(1, m.T, m.d).contiguous()
        return vfield(m, H, decay, Theta, AB).mean(1)[0]

    best = None
    for s_ in torch.linspace(0.15, 2.5, tries):
        hb = (u * r0 * float(s_)).clone()
        it = 0
        for it in range(80):
            with torch.enable_grad():
                J = torch.autograd.functional.jacobian(gfun, hb.clone().requires_grad_(True),
                                                       vectorize=True).detach()
            with torch.no_grad():
                g = gfun(hb)
                res = float(g.norm() / (hb.norm() + 1e-12))
                if res < 1e-11:
                    break
                try:
                    step = torch.linalg.solve(J + 1e-8 * torch.eye(m.d, device=DEV), g)
                except Exception:
                    break
                hb = hb - 0.8 * step
                if not torch.isfinite(hb).all():
                    break
        # 자명해 h̄≈0 은 균일 상태 분석의 대상이 아니므로 배제한다
        if torch.isfinite(hb).all() and float(hb.norm()) > 0.05 * r0:
            with torch.no_grad():
                res = float(gfun(hb).norm() / (hb.norm() + 1e-12))
            if math.isfinite(res) and (best is None or res < best[1]):
                best = (hb.clone(), res, it)
    if best is None:      # 비자명 고정점을 못 찾으면 데이터 반경의 균일 상태를 그대로 쓴다
        hb = (u * r0).clone()
        with torch.no_grad():
            best = (hb, float(gfun(hb).norm() / hb.norm()), -1)
    return best


@torch.no_grad()
def frozen_operator(m, hbar):
    """M_frozen ∈ R^{(T·d)×(T·d)} : A 를 상수로 본 선형화. 정확(근사 없음)."""
    decay, Theta = m.kernel(); AB = m.W_C()
    H = hbar.view(1, 1, -1).expand(1, m.T, m.d).contiguous()
    a, *_ = m.attn(H, decay, Theta, AB)                            # [1,H,T,T]
    a = a[0]
    OVP, L, g = ov_blocks(m), Lam(m), m.gamma
    T, d = m.T, m.d
    M = torch.zeros(T, d, T, d, device=DEV)
    M += torch.einsum('tn,ce->tcne', torch.eye(T, device=DEV), L)
    M += torch.einsum('htn,hce->tcne', a, OVP)
    quart = -g * (hbar.pow(2).sum() * torch.eye(d, device=DEV) + 2 * torch.outer(hbar, hbar))
    M += torch.einsum('tn,ce->tcne', torch.eye(T, device=DEV), quart)
    return M.reshape(T * d, T * d), a

@torch.no_grad()
def uniform_growth_direct(m, hbar):
    """균일 부분공간 span{1_T ⊗ e_c} 로 제한한 d×d 블록의 최대 성장률.
       M_unif = Λ + Σ_m d̄^(m)·W_OV^(m)P_m + quartic ,  d̄ = mean_t Σ_n a_tn."""
    decay, Theta = m.kernel(); AB = m.W_C()
    H = hbar.view(1, 1, -1).expand(1, m.T, m.d).contiguous()
    a, *_ = m.attn(H, decay, Theta, AB)
    dbar = a[0].sum(-1).mean(-1)                                   # [H]
    OVP, L, g = ov_blocks(m), Lam(m), m.gamma
    M = L + torch.einsum('h,hce->ce', dbar, OVP)
    M = M - g * (hbar.pow(2).sum() * torch.eye(m.d, device=DEV) + 2 * torch.outer(hbar, hbar))
    ev = torch.linalg.eigvals(M)
    i = int(ev.real.argmax())
    return float(ev.real[i]), float(ev.imag[i].abs())


def full_jacobian(m, hbar):
    """참 야코비안 J = ∂f/∂H (∂A/∂h 항 포함), autograd."""
    decay, Theta = m.kernel(); AB = m.W_C()
    def F(v):
        return vfield(m, v.view(1, m.T, m.d), decay, Theta, AB).reshape(-1)
    v0 = hbar.view(1, -1).expand(m.T, m.d).reshape(-1).clone()
    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(F, v0, vectorize=True)
    return J.detach()


@torch.no_grad()
def spectrum_vs_k(M, m):
    """고유값 + 고유벡터의 공간 파수 배정 (7×7 격자 2D FFT 파워 중심)."""
    ev, V = torch.linalg.eig(M)                                    # V: [Td, Td]
    G = m.grid
    Vr = V.reshape(m.T, m.d, -1)
    Fv = torch.fft.fft2(Vr.reshape(G, G, m.d, -1), dim=(0, 1))
    P = Fv.abs().pow(2).sum(2)                                     # [G,G,N] 채널 합
    fu = torch.fft.fftfreq(G, d=1.0, device=DEV) * 2 * math.pi
    KX = fu.view(G, 1, 1); KY = fu.view(1, G, 1)
    Kmag = torch.sqrt(KX ** 2 + KY ** 2)
    kbar = (P * Kmag).sum((0, 1)) / P.sum((0, 1)).clamp_min(1e-30)
    p0 = (P[0, 0] / P.sum((0, 1)).clamp_min(1e-30))                # k=0 파워 비중
    return ev.real.cpu().numpy(), ev.imag.cpu().numpy(), kbar.real.cpu().numpy(), p0.real.cpu().numpy()


def band_stats(re, kb, nb=16):
    """|k| 빈별 최대 성장률 프로파일."""
    bins = np.linspace(0, kb.max() + 1e-9, nb + 1)
    bc, prof = [], []
    for i in range(nb):
        s = (kb >= bins[i]) & (kb < bins[i + 1])
        if s.sum() >= 1:
            bc.append(0.5 * (bins[i] + bins[i + 1])); prof.append(re[s].max())
    return np.array(bc), np.array(prof)


def get_hdir(m, cfg, nb=256):
    x, _ = interp.testset(noisy=cfg.get("noisy", False))
    xb = x[:nb].to(DEV)
    decay, Theta = m.kernel(); AB = m.W_C()
    h = m.embed_patches(xb); dt = 1 / m.R
    a_fix = m.attn(h, decay, Theta, AB)[0] if m.freeze_A else None
    for r in range(m.R):
        h = m.phi(h, dt / 2)
        f, *_ = m.field(h, decay, Theta, a_fix, AB)
        h = h + dt * f; h = m.phi(h, dt / 2)
    mean = h.mean((0, 1)); rad = float(h.norm(dim=-1).mean())
    return mean, rad, float(mean.norm()) / rad


def untrained(cfg, seed=1234):
    torch.manual_seed(seed); a = cfg
    m = Model1(d=a["d"], H=a["H"], R=a["R"], gamma_init=a["gamma"], alpha_init=a["alpha"], eps=a["eps"],
               use_ov=not a["no_ov"], lam_mode=a["lam_mode"], orth_wc=not a["no_orth"],
               use_bias_v=not a["no_bias_v"], psi_zero=a.get("psi_zero", False),
               learn_gamma=not a.get("fix_gamma", False), freeze_A=a.get("freeze_A", False)).to(DEV)
    m.eval(); return m


def analyse(label, m, hdir, rad):
    hbar, res, it = uniform_fixed_point(m, hdir, rad)
    gu_dir, gu_im = uniform_growth_direct(m, hbar)
    out = dict(label=label, hbar=float(hbar.norm()), fp_res=res, gamma=float(m.gamma),
               g_uniform_direct=gu_dir, g_uniform_direct_im=gu_im,
               alpha_mean=float(m.alpha.mean()), corr_len=float(1 / m.alpha.mean()),
               theta_absmean=float(m.theta.abs().mean()), cospsi=float(torch.cos(m.psi).mean()))
    for name, M in (("frozen", frozen_operator(m, hbar)[0]), ("full", full_jacobian(m, hbar))):
        re, im, kb, p0 = spectrum_vs_k(M, m)
        bc, prof = band_stats(re, kb)
        i = int(np.argmax(re))
        # k≈0 모드: k=0 파워 비중이 가장 큰 고유벡터들 중 최대 성장률
        unif = p0 > 0.5
        g0 = float(re[unif].max()) if unif.any() else float(re[np.argmin(kb)])
        out[name] = dict(gmax=float(re[i]), kstar=float(kb[i]), imag_at_max=float(abs(im[i])),
                         g_uniform=g0, margin=float(re[i]) - g0,
                         n_unstable=int((re > 0).sum()), n_total=len(re),
                         bc=bc.tolist(), prof=prof.tolist(),
                         re=re.tolist(), kb=kb.tolist(), im=im.tolist())
    return out


if __name__ == "__main__":
    tags = sys.argv[1:] or ["noov"]
    res = {}
    for t in tags:
        m, cfg = interp.load(t)
        hdir, rad, ufrac = get_hdir(m, cfg)
        for lab, mm in ((t, m), (t + "·미학습", untrained(cfg))):
            r = analyse(lab, mm, hdir, rad); r["uniform_frac"] = ufrac; r["data_radius"] = rad
            res[lab] = r
            print(f"[{lab}]  ‖h̄‖={r['hbar']:.3f} (잔차 {r['fp_res']:.1e})  γ={r['gamma']:.4f}  "
                  f"1/⟨α⟩={r['corr_len']:.1f} (격자 {mm.grid})  ⟨|θ|⟩={r['theta_absmean']:.3f}  ⟨cos ψ⟩={r['cospsi']:+.3f}")
            print(f"    균일 부분공간 직접계산: Re λ_unif = {r['g_uniform_direct']:+8.4f}  (|Im|={r['g_uniform_direct_im']:.3f})")
            for nm in ("frozen", "full"):
                q = r[nm]
                print(f"    {nm:6s}  max Re λ = {q['gmax']:+8.4f} @ |k*|={q['kstar']:.3f}  "
                      f"(|Im|={q['imag_at_max']:.3f})   균일모드 {q['g_uniform']:+8.4f}   "
                      f"여유 {q['margin']:+.4f}   불안정 {q['n_unstable']}/{q['n_total']}")
        print()
    torch.save(res, f'{SCR}/disp.pt')
    print("saved", f'{SCR}/disp.pt')
