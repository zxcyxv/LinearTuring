"""noov 전용: 이 변종에서만 정확히 성립하는 반응-확산 구조를 실제로 측정한다.
   ① f = Σ a_tn(h_n − h_t) + d_t h_t 분해의 각 항 크기
   ② 음수 전도도 비율 — 진짜 확산이려면 a_tn ≥ 0 이어야 한다
   ③ 디리클레 에너지 궤적 — 확산이면 감소(평활화), 음수 계수면 증가(첨예화)
   ④ 반경 벡터장 ⟨f, ĥ⟩ vs ‖h‖ — FHN 이면 3차(N자) 곡선이어야 한다 (§3.3 핵심 주장)
   ⑤ 야코비안 비대칭도"""
import sys, json; sys.path.insert(0,'.')
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
import interp
plt.rcParams.update({"figure.dpi":130,"font.size":8,"axes.titlesize":8,"axes.unicode_minus":False})

@torch.no_grad()
def decompose(m, h, decay, Theta):
    """헤드별 확산항/반응항.  noov 는 W_OV = 블록항등이라 분해가 정확하다."""
    a, *_ = m.attn(h, decay, Theta)                       # [B,H,T,T]
    B, T, d = h.shape
    v = h.view(B, T, m.H, m.dh)                           # P_m h
    mixed = torch.einsum('bhtn,bnhc->bthc', a, v)         # Σ_n a_tn h_n[m]
    dt = a.sum(-1)                                        # d_t^(m)  [B,H,T]
    reac = dt.permute(0,2,1).unsqueeze(-1) * v            # d_t h_t
    diff = mixed - reac                                   # Σ_n a_tn (h_n − h_t)
    return a, diff, reac, dt

@torch.no_grad()
def run(tag, nb=64):
    m, cfg = interp.load(tag); x, _ = interp.testset(noisy=cfg["noisy"])
    xb = x[:nb].to(interp.DEV); decay, Theta = m.kernel()
    # 끌개 반경 바깥까지 표본을 얻으려고 초기 대비를 섞는다 (§3.5 와 같은 요령)
    h0 = m.embed_patches(xb)
    cs = torch.linspace(0.3, 3.0, xb.shape[0], device=h0.device).view(-1,1,1)
    h = h0*cs; dt_step = 1.0/m.R
    rec = {"diff":[], "reac":[], "lam":[], "quart":[], "E":[], "negfrac":[], "negmass":[], "hn":[]}
    rad = []                                              # (‖h‖, ⟨f,ĥ⟩) 산점
    for r in range(m.R*3):                                # 3배 길게 돌려 끌개까지
        h = m.phi(h, dt_step/2)
        a, diff, reac, dtm = decompose(m, h, decay, Theta)
        f, *_ = m.field(h, decay, Theta)
        rec["diff"].append(diff.norm(dim=-1).mean().item())
        rec["reac"].append(reac.norm(dim=-1).mean().item())
        rec["lam"].append((h*m.lam).norm(dim=-1).mean().item() if m.lam_mode=="diag"
                          else (h@m.lam.t()).norm(dim=-1).mean().item() if m.lam_mode=="full" else 0.0)
        rec["quart"].append((m.gamma*h.pow(2).sum(-1,keepdim=True)*h).norm(dim=-1).mean().item())
        hbar = h.mean(1, keepdim=True)
        rec["E"].append((h-hbar).pow(2).sum().div(h.shape[0]).item())      # 디리클레 에너지
        rec["negfrac"].append((a<0).float().mean().item())
        rec["negmass"].append((a.clamp(max=0).abs().sum()/a.abs().sum()).item())
        rec["hn"].append(h.norm(dim=-1).mean().item())
        hh = h.reshape(-1, m.d); ff = f.reshape(-1, m.d)
        nrm = hh.norm(dim=-1)
        exc = (ff*hh).sum(-1)/(nrm+1e-9)          # 흥분부 (field 는 4차항을 포함하지 않는다)
        quart = -m.gamma*nrm.pow(3)               # Strang 의 φ 가 담당하는 4차 소산의 반경 성분
        rad.append(torch.stack([nrm, exc, exc+quart], -1).cpu())
        h = h + dt_step*f; h = m.phi(h, dt_step/2)
    return m, rec, torch.cat(rad).numpy()

@torch.no_grad()
def jac_asym(m, tag, n=24):
    x = interp.testset(noisy=False)[0]
    xb = x[:4].to(interp.DEV); decay,Theta = m.kernel(); h = m.embed_patches(xb)
    f = lambda hh: m.field(hh, decay, Theta)[0]
    rs=[]
    for _ in range(n):
        u=torch.randn_like(h); v=torch.randn_like(h); u/=u.norm(); v/=v.norm()
        _,Jv=torch.func.jvp(f,(h,),(v,)); _,Ju=torch.func.jvp(f,(h,),(u,))
        s1=(u*Jv).sum().item(); s2=(v*Ju).sum().item()
        rs.append(abs(s1-s2)/(abs(s1)+abs(s2)+1e-12))
    return float(np.mean(rs))

tags = sys.argv[1:] or ["noov"]
out={}
fig, ax = plt.subplots(1, 4, figsize=(12.6, 2.9))
for tag in tags:
    m, rec, rad = run(tag)
    ja = jac_asym(m, tag)
    R = m.R
    out[tag] = {k:v for k,v in rec.items()}; out[tag]["jac_asym"]=ja
    tt = np.arange(len(rec["diff"]))
    ax[0].plot(tt, rec["diff"], label=f"{tag} 확산항"); ax[0].plot(tt, rec["reac"], '--', label=f"{tag} 반응항")
    ax[1].plot(tt, rec["E"], label=tag)
    ax[2].plot(tt, rec["negmass"], label=f"{tag} 음수질량"); ax[2].plot(tt, rec["negfrac"], ':', label=f"{tag} 음수비율")
    # 반경 벡터장: ‖h‖ 구간별 평균
    nb_=44; lo,hi=np.percentile(rad[:,0],[0.5,99.5]); bins=np.linspace(lo,hi,nb_)
    idx=np.clip(np.digitize(rad[:,0],bins)-1,0,nb_-1)
    mu =np.array([rad[idx==i,2].mean() if (idx==i).sum()>5 else np.nan for i in range(nb_)])
    mue=np.array([rad[idx==i,1].mean() if (idx==i).sum()>5 else np.nan for i in range(nb_)])
    ax[3].plot(bins, mu, 'o-', ms=2.5, label=f"{tag} 전체")
    ax[3].plot(bins, mue, ':', lw=.9, label=f"{tag} 흥분부만")
    zc=np.where(np.diff(np.sign(np.nan_to_num(mu))))[0]
    print(f"   반경 벡터장 영교차 ‖h‖ = " + (", ".join(f"{bins[i]:.1f}" for i in zc) if len(zc) else "없음") +
          f"  (끌개 실측 ‖h‖ ≈ {rec['hn'][-1]:.1f})")
    print(f"[{tag}] R={R} 야코비안 비대칭도={ja:.4f}")
    print(f"   확산/반응 크기비: 초기 {rec['diff'][0]/rec['reac'][0]:.2f} → 끌개 {rec['diff'][-1]/rec['reac'][-1]:.2f}")
    print(f"   음수 전도도: 비율 {rec['negfrac'][-1]:.3f}, 질량분율 {rec['negmass'][-1]:.3f}")
    print(f"   디리클레 에너지: {rec['E'][0]:.1f} → {rec['E'][R]:.1f} → {rec['E'][-1]:.1f}  ({'감소=평활화' if rec['E'][-1]<rec['E'][0] else '증가=첨예화'})")
    print(f"   항 크기(끌개): 확산 {rec['diff'][-1]:.2f} 반응 {rec['reac'][-1]:.2f} Λ {rec['lam'][-1]:.2f} 4차 {rec['quart'][-1]:.2f}")
for a_,t_ in zip(ax, ["① 확산항 vs 반응항 크기","② 디리클레 에너지 (감소=확산)",
                      "③ 음수 전도도 (0이어야 진짜 확산)","④ 반경 벡터장 ⟨f,ĥ⟩ (FHN이면 N자)"]):
    a_.set_title(t_); a_.legend(fontsize=5.5); a_.set_xlabel("스텝 r" if t_[0] in "①②③" else "‖h‖")
ax[3].axhline(0, ls='--', c='k', lw=.6); ax[3].set_xlabel("‖h‖")
fig.suptitle("반응-확산 구조의 실측 — noov 는 분해가 정확히 성립하는 유일한 변종", y=1.03)
fig.tight_layout(); fig.savefig("figs/rd_analysis.png", bbox_inches="tight")
json.dump(out, open("runs/rd_analysis.json","w"))
