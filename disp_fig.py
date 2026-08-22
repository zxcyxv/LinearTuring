"""분산관계 그림: Re λ vs |k| (학습 vs 미학습), 그리고 frozen vs full 야코비안."""
import sys, math
sys.path.insert(0,'/workspace/LinearTuring')
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":130,"font.size":8,"axes.titlesize":8.5,"axes.unicode_minus":False})
SCR='/tmp/claude-0/-workspace-LinearTuring/ac978981-cfca-4f33-817b-3ad8423e9e35/scratchpad'
res = torch.load(f'{SCR}/disp.pt', weights_only=False)

tags = [t for t in res if not t.endswith('·미학습')]
n = len(tags)
fig, ax = plt.subplots(2, n, figsize=(2.55*n, 5.0), squeeze=False)
for i, t in enumerate(tags):
    for row, key in enumerate(("frozen", "full")):
        A = ax[row, i]
        for lab, style in ((t, dict(c='crimson', s=5, alpha=.55)),
                           (t+'·미학습', dict(c='steelblue', s=5, alpha=.35))):
            if lab not in res: continue
            q = res[lab][key]
            A.scatter(q['kb'], q['re'], **style, lw=0,
                      label=('학습' if lab==t else '미학습'))
            bc, pr = np.array(q['bc']), np.array(q['prof'])
            A.plot(bc, pr, c=style['c'], lw=1.4, alpha=.9)
            A.axhline(res[lab]['g_uniform_direct'], c=style['c'], ls=':', lw=1.0)
        A.axhline(0, c='k', lw=.6)
        q = res[t][key]
        A.plot([q['kstar']], [q['gmax']], marker='v', c='k', ms=6, ls='none')
        A.annotate(f"|k*|={q['kstar']:.2f}", (q['kstar'], q['gmax']),
                   textcoords='offset points', xytext=(4,6), fontsize=7)
        A.set_title(f"{t} — {'A 고정 선형화' if key=='frozen' else '참 야코비안'}")
        A.set_xlabel("공간 파수 |k|"); A.set_ylabel("Re λ (성장률)")
        if i==0 and row==0: A.legend(fontsize=7, loc='lower right')
        A.grid(alpha=.25, lw=.4)
fig.suptitle("균일 상태 주변 분산관계 — 점선 = 균일모드 성장률(직접계산), ▼ = 최대 성장 파수", y=.995, fontsize=9)
fig.tight_layout()
fig.savefig('/workspace/LinearTuring/figs/dispersion.png', bbox_inches='tight')
print("saved figs/dispersion.png")

# 요약표
print(f"\n{'모델':16s} {'‖h̄‖':>7s} {'균일모드':>9s} {'max Reλ':>9s} {'|k*|':>6s} {'|Im|':>7s} {'여유':>8s} {'판정':>6s}")
for t in tags:
    for lab in (t, t+'·미학습'):
        if lab not in res: continue
        r = res[lab]; q = r['full']
        turing = (r['g_uniform_direct'] < 0) and (q['gmax'] > 0)
        print(f"{lab:16s} {r['hbar']:7.3f} {r['g_uniform_direct']:+9.3f} {q['gmax']:+9.3f} "
              f"{q['kstar']:6.2f} {q['imag_at_max']:7.3f} {q['gmax']-r['g_uniform_direct']:+8.3f} "
              f"{'튜링' if turing else '—':>6s}")
