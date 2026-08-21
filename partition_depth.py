"""분할이 재귀 깊이에 따라 무엇을 기준으로 삼는지 추적한다.
   각 r 에서:  좌절도 · 분할 v_t · 다음 세 지표
     ① |corr(v, 패치밝기)|            — 저수준(획/배경)
     ② 클래스내 |corr| − 클래스간 |corr| — 의미론적 (부호 불변)
     ③ 샘플간 |corr| 평균              — 입력 의존성 (낮을수록 데이터 특이적)"""
import sys, json; sys.path.insert(0,'.')
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
import interp
plt.rcParams.update({"figure.dpi":130,"font.size":8,"axes.titlesize":8,"axes.unicode_minus":False})

@torch.no_grad()
def track(tag, nb=256, mult=3):
    m,cfg=interp.load(tag); x,y=interp.testset(noisy=cfg['noisy'], n=nb)
    xb,yb=x[:nb].to(interp.DEV), y[:nb].numpy()
    dec,Th=m.kernel(); h=m.embed_patches(xb); dt=1/m.R; T=m.T; G,P=m.grid,m.patch
    bright=xb.view(nb,1,G,P,G,P).permute(0,2,4,3,5,1).reshape(nb,G*G,P*P).mean(-1).cpu().numpy()
    same = yb[:,None]==yb[None,:]; iu=np.triu_indices(nb,1)
    out={"frus":[], "bright":[], "cls":[], "xsample":[], "gap":[]}
    for r in range(m.R*mult+1):
        a,*_=m.attn(h,dec,Th); A=a.mean(1); As=0.5*(A+A.transpose(-1,-2))
        D=As.abs().sum(-1); Di=(D+1e-9).pow(-0.5)
        L=torch.eye(T,device=As.device)-Di.unsqueeze(-1)*As*Di.unsqueeze(-2)
        ev,evec=torch.linalg.eigh(L); V=evec[:,:,0].cpu().numpy()
        out["frus"].append(float(ev[:,0].mean()))
        out["gap"].append(float((ev[:,1]/(ev[:,0]+1e-9)).median()))
        out["bright"].append(float(np.mean([abs(np.corrcoef(V[b],bright[b])[0,1]) for b in range(nb)])))
        C=np.abs(np.corrcoef(V))
        out["xsample"].append(float(C[iu].mean()))
        out["cls"].append(float(C[same&(np.arange(nb)[:,None]<np.arange(nb)[None,:])].mean()
                               - C[(~same)&(np.arange(nb)[:,None]<np.arange(nb)[None,:])].mean()))
        h=m.phi(h,dt/2); f,*_=m.field(h,dec,Th); h=h+dt*f; h=m.phi(h,dt/2)
    return m,out

tags=sys.argv[1:] or ["noov"]
fig,ax=plt.subplots(1,4,figsize=(12.4,2.9)); res={}
for tag in tags:
    m,o=track(tag); res[tag]=o; tt=np.arange(len(o["frus"]))
    ax[0].plot(tt,o["frus"],label=tag); ax[1].plot(tt,o["bright"],label=tag)
    ax[2].plot(tt,o["cls"],label=tag);  ax[3].plot(tt,o["xsample"],label=tag)
    print(f"[{tag}] r:            " + " ".join(f"{v:5d}" for v in [0,2,4,6,8,12,16,24]))
    for k,lab in [("frus","좌절도       "),("bright","|corr| 밝기  "),
                  ("cls","클래스내−간 "),("xsample","샘플간 |corr|")]:
        print(f"   {lab} " + " ".join(f"{o[k][i]:5.3f}" for i in [0,2,4,6,8,12,16,24]))
for a_,t_ in zip(ax,["좌절도 (0=균형)","① 분할 vs 패치밝기 |corr|",
                     "② 클래스내 − 클래스간 유사도","③ 샘플간 |corr| (낮을수록 입력 특이적)"]):
    a_.set_title(t_); a_.set_xlabel("스텝 r"); a_.legend(fontsize=6); a_.axvline(8,ls=':',c='k',lw=.7)
ax[2].axhline(0,ls='--',c='k',lw=.6)
fig.suptitle("분할의 기준은 깊이에 따라 바뀌는가",y=1.03)
fig.tight_layout(); fig.savefig("figs/partition_depth.png",bbox_inches="tight")
json.dump(res,open("runs/partition_depth.json","w"))
