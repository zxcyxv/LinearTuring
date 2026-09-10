"""모순(I) 상태가 정착인가 이동중인가: 위반 수 궤적 · 후반 churn · 동결 비율."""
import sys, numpy as np
from state_transitions import valid, UI
def nviol(D):                                   # [NB,81] → 위반 유닛 수 [NB]
    U=D[:,UI]; return (np.sort(U,axis=2)!=np.arange(1,10)).any(2).sum(1)
def run(p):
    d=np.load(p); P,X,Y=d["P"],d["X"],d["Y"]; SEG,NB,_=P.shape; D=P.astype(int)-1; blank=(X==0)
    C=(D==Y[None]).all(2); V=np.array([valid(D[s]) for s in range(SEG)])&~C; I=~C&~V
    last=SEG-1; Il=I[last]; Vl=V[last]
    print(f"===== {p.split('/')[-1]}   seg{SEG}:  C {C[last].sum()}  V {Vl.sum()}  I {Il.sum()} =====")
    print("[비-정답 퍼즐의 위반 유닛 수 평균 (27개 중)]  — 그 세그먼트에서 I 인 퍼즐만")
    print("   seg :  " + "  ".join(f"{s:>4d}" for s in [1,4,8,16,32,64,96,128] if s<=SEG))
    print("   위반:  " + "  ".join(f"{nviol(D[s-1])[I[s-1]].mean():4.1f}" for s in [1,4,8,16,32,64,96,128] if s<=SEG))
    print("   n   :  " + "  ".join(f"{I[s-1].sum():4d}" for s in [1,4,8,16,32,64,96,128] if s<=SEG))
    # seg128 에 I 인 퍼즐들의 위반 수 궤적 (같은 퍼즐 집합 고정)
    print(f"[seg{SEG} 에 I 인 퍼즐 {Il.sum()}개의 위반 수 궤적]")
    print("   seg :  " + "  ".join(f"{s:>4d}" for s in [16,32,64,96,112,128] if s<=SEG))
    print("   위반:  " + "  ".join(f"{nviol(D[s-1])[Il].mean():4.1f}" for s in [16,32,64,96,112,128] if s<=SEG))
    # 후반 churn 과 동결
    W=16; ch=np.zeros(NB)
    for s in range(SEG-W,SEG):
        chg=(D[s]!=D[s-1])&blank; ch+=chg.sum(1)/np.maximum(blank.sum(1),1)
    ch/=W
    frozen=np.array([(D[SEG-W:,b]==D[SEG-W,b]).all() for b in range(NB)])
    for lab,m in [("I",Il),("V",Vl),("C",C[last])]:
        if m.sum(): print(f"[seg{SEG} {lab} {m.sum():3d}개] 마지막 16seg churn 평균 {ch[m].mean():.4f}   완전 동결 {frozen[m].sum()}개 ({100*frozen[m].mean():.0f}%)")
    v=nviol(D[last])[Il]
    if Il.any():
        print(f"[seg{SEG} I 의 위반 수 분포] 1~2: {int((v<=2).sum())}  3~5: {int(((v>=3)&(v<=5)).sum())}  6~10: {int(((v>=6)&(v<=10)).sum())}  11+: {int((v>=11).sum())}   틀린칸 평균 {(D[last]!=Y).sum(1)[Il].mean():.1f}")
    print()
for p in sys.argv[1:]: run(p)
