"""모델 간 동역학 비교: dt 쓸기 속도 · anytime 정확도 · 자기상관."""
import sys, json, os; sys.path.insert(0,'.')
import numpy as np, matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":130,"font.size":8,"axes.titlesize":8,"axes.unicode_minus":False})
LBL={'full':'full (완전 Model 1)','noov':'noov (순수 좌곱셈)','noisy':'noisy (잡음 MNIST)',
     'psi0':'psi0 (ψ≡0 대칭)','kern':'kern (커널 lr×30)','randR':'randR (R~U(4,16))',
     'gradflow':'gradflow (ψ=0,W_OV=I,Λ=0,b=0)','lamonly':'lamonly (gradflow+Λ)',
     'frozenA':'frozenA (A 고정, dt 무작위)','frozenA8':'frozenA8 (A 고정, R=8)'}
tags=[t for t in LBL if os.path.exists(f'runs/{t}_interp.json')]
fig,ax=plt.subplots(1,3,figsize=(10.4,3.1))
for t in tags:
    o=json.load(open(f'runs/{t}_interp.json')); e=o['extrap']; rf=o['refine']
    ax[0].plot(e['anytime_acc'],lw=1.1,label=LBL[t])
    ax[1].plot(rf['Rs'],rf['acc'],'o-',ms=3,lw=1.1)
    d=f'runs/{t}_dtsweep.json'
    if os.path.exists(d):
        s=json.load(open(d)); k=sorted(s,key=int)
        ax[2].plot([int(x) for x in k],[s[x]['speed_late'] for x in k],'o-',ms=3,lw=1.1)
ax[0].axvline(8,ls=':',c='k'); ax[0].set_xlabel("스텝 r (dt=1/8 고정)"); ax[0].set_ylabel("accuracy")
ax[0].set_title("anytime 정확도"); ax[0].legend(fontsize=5.5,loc='upper right')
ax[1].set_xscale('log'); ax[1].axvline(8,ls=':',c='k'); ax[1].set_xlabel("적분 스텝수 R (τ=1)")
ax[1].set_ylabel("accuracy"); ax[1].set_title("적분 정밀화 저항")
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel("1/dt")
ax[2].set_ylabel("τ∈[6,8] 상대속도"); ax[2].set_title("연속극한 속도: 0 이면 고정점")
ax[2].set_ylim(1e-3,10)
fig.suptitle("모델별 동역학 비교",y=1.03); fig.tight_layout()
fig.savefig("figs/compare_dynamics.png",bbox_inches="tight"); print("saved", tags)
