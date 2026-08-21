"""반응-확산 판정을 제대로 한다.
   ① 항별 에너지 수지: dE/dt 에 대한 각 항의 기여.
      정리 — 전도도가 음이 아닌 확산 연산자는 절대 E 를 늘릴 수 없다.
      확산항 기여가 양수면 그건 확산이 아니라 에너지 주입원이다.
   ② 튜링 서명: 상태장의 2D 공간 스펙트럼이 k≠0 에서 봉우리를 갖는가.
      (k=0 봉우리 = 균일, 광대역 = 잡음, k*≠0 봉우리 = 튜링 패턴)"""
import sys, json; sys.path.insert(0,'.')
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
import interp
from rd_analysis import decompose
plt.rcParams.update({"figure.dpi":130,"font.size":8,"axes.titlesize":8,"axes.unicode_minus":False})

@torch.no_grad()
def run(tag, nb=128, mult=3):
    m,cfg = interp.load(tag); x,_ = interp.testset(noisy=cfg["noisy"])
    xb = x[:nb].to(interp.DEV); decay,Theta = m.kernel()
    h = m.embed_patches(xb); dt = 1.0/m.R
    G = m.grid; T = m.T
    E=[]; contrib={"확산":[], "반응":[], "Λ":[], "4차":[]}; spec=[]
    for r in range(m.R*mult):
        h = m.phi(h, dt/2)
        a, diff, reac, dtm = decompose(m, h, decay, Theta)
        B = h.shape[0]
        c = h - h.mean(1, keepdim=True)                      # h_t − h̄  (에너지의 편차 성분)
        # 각 항을 [B,T,d] 로 되돌린다 (noov: W_OV = 블록항등이므로 그대로 이어붙이기)
        ov = m.OV()
        gd = torch.einsum('bthc,hdc->btd', diff, ov)
        gr = torch.einsum('bthc,hdc->btd', reac, ov)
        gl = (h @ m.lam.t()) if m.lam_mode=="full" else (h*m.lam) if m.lam_mode=="diag" else torch.zeros_like(h)
        gq = -m.gamma * h.pow(2).sum(-1, keepdim=True) * h
        for k_, term in zip(contrib, [gd, gr, gl, gq]):
            contrib[k_].append((2*(c*term).sum()/B).item())   # dE/dt 기여
        E.append(c.pow(2).sum().div(B).item())
        if r == m.R*mult-1 or r == m.R-1:                     # 학습 깊이와 끌개에서 스펙트럼
            fld = c.reshape(B, G, G, m.d)
            P = torch.fft.fft2(fld, dim=(1,2)).abs().pow(2).mean(dim=(0,3))
            spec.append((r, torch.fft.fftshift(P).cpu().numpy()))
        f,*_ = m.field(h, decay, Theta)
        h = h + dt*f; h = m.phi(h, dt/2)
    return m, E, contrib, spec

tags = sys.argv[1:] or ["noov"]
fig, ax = plt.subplots(1, 2+len(tags), figsize=(4.0*(2+len(tags))*0.85, 3.0))
res={}
for i,tag in enumerate(tags):
    m,E,contrib,spec = run(tag)
    res[tag]={"E":E,"contrib":contrib}
    tt=np.arange(len(E))
    for k_,v in contrib.items(): ax[0].plot(tt, v, label=f"{tag} {k_}")
    ax[1].plot(tt, E, label=tag)
    r_, P = spec[-1]
    G=m.grid; ks=np.fft.fftshift(np.fft.fftfreq(G))*G
    im=ax[2+i].imshow(np.log10(P+1e-9), extent=[ks[0],ks[-1],ks[0],ks[-1]], cmap="magma")
    ax[2+i].set_title(f"{tag} 공간 스펙트럼 log10 (r={r_})"); ax[2+i].set_xlabel("k_w"); ax[2+i].set_ylabel("k_u")
    # 방사 평균으로 k* 찾기
    KU,KW=np.meshgrid(ks,ks); KR=np.sqrt(KU**2+KW**2)
    edges=np.arange(0, KR.max()+0.75, 0.75); prof=[]
    for j in range(len(edges)-1):
        msk=(KR>=edges[j])&(KR<edges[j+1])
        prof.append(P[msk].mean() if msk.sum() else np.nan)
    prof=np.array(prof); ctr=(edges[:-1]+edges[1:])/2
    kstar=ctr[int(np.nanargmax(prof))]
    res[tag]["radial"]={"k":ctr.tolist(),"P":prof.tolist(),"kstar":float(kstar)}
    tot=sum(abs(np.array(v)[-1]) for v in contrib.values())
    print(f"[{tag}] 끌개에서의 dE/dt 기여 (양수=에너지 주입):")
    for k_,v in contrib.items():
        print(f"    {k_:4s} {v[-1]:+10.1f}  ({100*abs(v[-1])/tot:4.1f}%)   {'주입' if v[-1]>0 else '소산'}")
    print(f"    합계 {sum(v[-1] for v in contrib.values()):+.1f}   E={E[-1]:.0f}")
    print(f"    방사 스펙트럼 최대 파수 k* = {kstar:.2f}  (0 이면 균일, 최대 {ctr[-1]:.1f} 면 격자 잡음)")
    print(f"    P(k=0)/P(k*) = {prof[0]/prof[int(np.nanargmax(prof))]:.3f}")
ax[0].axhline(0,ls='--',c='k',lw=.6); ax[0].set_xlabel("스텝 r"); ax[0].set_ylabel("dE/dt 기여")
ax[0].set_title("① 항별 에너지 수지"); ax[0].legend(fontsize=5.5)
ax[1].set_xlabel("스텝 r"); ax[1].set_ylabel("디리클레 에너지 E"); ax[1].set_title("② 전체 에너지"); ax[1].legend(fontsize=6)
fig.suptitle("확산항은 에너지를 넣는가 빼는가 · 튜링 특징 파장은 있는가", y=1.03)
fig.tight_layout(); fig.savefig("figs/turing.png", bbox_inches="tight")
json.dump(res, open("runs/turing.json","w"))
