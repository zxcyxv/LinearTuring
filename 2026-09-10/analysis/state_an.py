"""세그먼트별 예측 P[SEG,NB,81](토큰 2..10=숫자) 에서 상태 {C 정답, V 무모순·오답, I 모순} 의 전이 구조."""
import sys, numpy as np
UNITS=([[r*9+c for c in range(9)] for r in range(9)]+[[r*9+c for r in range(9)] for c in range(9)]
       +[[(br+a)*9+(bc+b) for a in range(3) for b in range(3)] for br in (0,3,6) for bc in (0,3,6)])
UI=np.array(UNITS)
def valid(D):
    ok=((D>=1)&(D<=9)).all(1); U=D[:,UI]
    return ok&(np.sort(U,axis=2)==np.arange(1,10)).all(2).all(1)
def analyze(P,X,Y,name):
    SEG,NB,_=P.shape; D=P.astype(int)-1; blank=(X==0)
    C=(D==Y[None]).all(2)                                   # [SEG,NB]
    V=np.array([valid(D[s]) for s in range(SEG)])&~C
    I=~C&~V
    S=np.where(C,0,np.where(V,1,2))                         # 0=C 1=V 2=I
    nwrong=(D!=Y[None]).sum(2)                              # [SEG,NB] 틀린 칸 수
    print(f"===== {name}  ({NB} 퍼즐, {SEG} seg) =====")
    print(f"{'seg':>4} {'C':>4} {'V':>4} {'I':>4} | 틀린칸 평균: V {'':>3} I {'':>3} | 빈칸 오류율")
    for s in [1,2,4,8,16,32,64,96,128]:
        if s>SEG: continue
        k=s-1; v=V[k]; i=I[k]
        er=(D[k]!=Y)[blank].mean()
        print(f"{s:4d} {C[k].sum():4d} {V[k].sum():4d} {I[k].sum():4d} | {nwrong[k][v].mean() if v.any() else float('nan'):8.1f} {nwrong[k][i].mean() if i.any() else float('nan'):6.1f} | {er:.4f}")
    # V 가 경유지인가 함정인가
    last=SEG-1
    solved=C[last]
    everV=V.any(0); everV_before=np.array([V[:np.argmax(C[:,b]) if C[:,b].any() else SEG, b].any() for b in range(NB)])
    print(f"\n[seg{SEG} 정답 퍼즐 {solved.sum()}개] 그 전에 V 를 거친 적: {(everV_before&solved).sum()}  ({100*(everV_before&solved).mean()/max(solved.mean(),1e-9):.1f}%)")
    print(f"[seg16 상태 → seg{SEG} 상태]")
    for a,an in [(0,'C'),(1,'V'),(2,'I')]:
        m=S[15]==a
        if m.sum(): print(f"   seg16 {an} ({m.sum():3d}) → C {int((S[last][m]==0).sum()):3d}  V {int((S[last][m]==1).sum()):3d}  I {int((S[last][m]==2).sum()):3d}")
    # 연속 세그 전이 행렬 (풀링)
    T=np.zeros((3,3),int)
    for s in range(SEG-1):
        for a in range(3):
            for b in range(3): T[a,b]+=int(((S[s]==a)&(S[s+1]==b)).sum())
    print("[연속 세그먼트 전이 확률 (행=현재, 열=다음: C V I)]")
    for a,an in enumerate("CVI"):
        row=T[a]; tot=row.sum()
        print(f"   {an}: "+"  ".join(f"{x/max(tot,1):.4f}" for x in row)+f"   (n={tot})")
    # V 상태의 지속 길이 & V 에서 C 로 간 경우 V→C 직전 틀린칸 수
    vc=[]; 
    for b in range(NB):
        for s in range(SEG-1):
            if S[s,b]==1 and S[s+1,b]==0: vc.append(nwrong[s,b])
    if vc: print(f"[V→C 전이 {len(vc)}회] 전이 직전 V 의 틀린칸 수 평균 {np.mean(vc):.1f}  중앙 {np.median(vc):.0f}")
    ic=[]
    for b in range(NB):
        for s in range(SEG-1):
            if S[s,b]==2 and S[s+1,b]==0: ic.append(nwrong[s,b])
    if ic: print(f"[I→C 전이 {len(ic)}회] 전이 직전 I 의 틀린칸 수 평균 {np.mean(ic):.1f}  중앙 {np.median(ic):.0f}")
    # 단서 변경 여부 (V 에서)
    clue=(X!=0)
    cc=[]
    for s in [15,last]:
        v=V[s]; 
        if v.any():
            changed=np.array([(D[s,b][clue[b]]!=X[b][clue[b]]).sum() for b in np.where(v)[0]])
            print(f"[seg{s+1} V {v.sum()}개] 바뀐 단서 수 평균 {changed.mean():.2f}, 0개 {int((changed==0).sum())}, 1개 {int((changed==1).sum())}, 2+ {int((changed>=2).sum())}")
    print()
if __name__=="__main__":
    for p in sys.argv[1:]:
        d=np.load(p); analyze(d["P"],d["X"],d["Y"],p.split("/")[-1])
