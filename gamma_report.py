"""γ 쓸기 정리: λ · 정확도 · 끌개 반경 · 입력 강건성."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":130,"font.size":8,"axes.titlesize":8,"axes.unicode_minus":False})
import glob,re
G=sorted(float(re.match(r"runs/g([\d.]+)_interp\.json",f).group(1)) for f in glob.glob("runs/g[0-9]*_interp.json"))
tags=[f"g{g}" for g in G]
NB=sorted(float(re.match(r"runs/nb_g([\d.]+)_attractor\.json",f).group(1)) for f in glob.glob("runs/nb_g*_attractor.json"))
rows=[]
rob={r['tag']:r for r in json.load(open("runs/robustness.json"))} if os.path.exists("runs/robustness.json") else {}
for g,t in zip(G,tags):
    if not os.path.exists(f'runs/{t}_interp.json'): continue
    o=json.load(open(f'runs/{t}_interp.json')); L=json.load(open(f'runs/{t}_log.json'))
    a=json.load(open(f'runs/{t}_attractor.json')) if os.path.exists(f'runs/{t}_attractor.json') else {}
    e=o['extrap']
    rows.append(dict(g=g,tag=t,acc=L['epochs'][-1]['test_acc'],lam=a.get('ftle'),
                     h=e['h_max'][64],speed=e.get('incr_over_h'),refine=o['refine']['acc'][-1],
                     rec=max(a['autocorr'][2:]) if a.get('autocorr') else None,
                     flip=rob.get(t,{}).get('flip',[None]*6)[4]))
fig,ax=plt.subplots(1,4,figsize=(12.4,2.9))
gs=[r['g'] for r in rows]
ax[0].semilogx(gs,[r['acc'] for r in rows],'o-'); ax[0].set_xlabel("γ (4차 소산)"); ax[0].set_ylabel("test 정확도")
ax[0].set_title("정확도")
ax[1].semilogx(gs,[r['lam'] for r in rows],'o-',c='tab:red'); ax[1].axhline(0,ls='--',c='k',lw=.7)
ax[1].set_xlabel("γ"); ax[1].set_ylabel("λ (FTLE)"); ax[1].set_title("리아푸노프 지수")
ax[2].semilogx(gs,[r['h'] for r in rows],'o-',c='tab:green'); ax[2].set_xlabel("γ")
ax[2].set_ylabel("끌개 ‖h‖"); ax[2].set_title("끌개 반경"); ax[2].set_yscale('log')
gg=np.array(gs); ref=rows[0]['h']*np.sqrt(gg[0]/gg); ax[2].plot(gs,ref,'--',c='gray',lw=.8,label="기울기 -1/2 (게이지 예측)")
ax[2].legend(fontsize=6)
ax[3].semilogx(gs,[r['flip'] for r in rows],'o-',c='tab:purple',label="예측 뒤집힘 (σ=0.4)")
ax[3].semilogx(gs,[r['rec'] for r in rows],'s--',c='tab:brown',label="자기상관 재발")
ax[3].set_xlabel("γ"); ax[3].legend(fontsize=6); ax[3].set_title("강건성 · 궤도 조직화")
if NB:
    nb=[json.load(open(f"runs/nb_g{g}_attractor.json"))['ftle'] for g in NB]
    ax[1].semilogx(NB,nb,'s--',c='tab:blue',label="b 제거 (게이지 예측: 평평)")
    ax[1].legend(fontsize=6)
from matplotlib.ticker import FixedLocator, FixedFormatter
for a_ in ax:
    a_.xaxis.set_major_locator(FixedLocator(gs)); a_.xaxis.set_minor_locator(FixedLocator([]))
    a_.xaxis.set_major_formatter(FixedFormatter([("%g"%g) for g in gs]))
    a_.tick_params(axis="x", labelsize=6)
ax[2].yaxis.set_major_locator(FixedLocator([0.25,1,4,16])); ax[2].yaxis.set_minor_locator(FixedLocator([]))
ax[2].yaxis.set_major_formatter(FixedFormatter(["0.25","1","4","16"]))
fig.suptitle("4차 소산 γ 쓸기 — 카오스를 죽이면 무엇을 잃는가",y=1.03); fig.tight_layout()
fig.savefig("figs/gamma_sweep.png",bbox_inches="tight")
json.dump(rows,open("runs/gamma_rows.json","w"))
for r in rows: print(r)
