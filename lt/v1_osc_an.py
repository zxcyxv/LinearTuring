import numpy as np, sys, os
SP=os.environ.get("LT_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"runs","analysis"))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from solver import propagate, PEERS
d=np.load(os.path.join(SP,"v1_osc.npz")); X,Y,P=d["X"],d["Y"],d["P"]; NB=X.shape[0]
try:
    from scipy.stats import spearmanr
    def sp(a,b): return spearmanr(a,b).correlation
except Exception:
    def sp(a,b):
        ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
        return np.corrcoef(ra,rb)[0,1]
blank=(X==0)
# --- 진동 세기 (1 − 집중도) ---
OSC={"칸위상 seg16":1-d["ru16"],"칸위상(r̂²가중) seg16":1-d["rw16"],"칸위상 seg128":1-d["ruF"],"칸위상(r̂²가중) seg128":1-d["rwF"]}
def peer_osc(rp):
    o=np.zeros((NB,81))
    for t in range(81): o[:,t]=1-rp[:,t,PEERS[t]].mean(1)
    return o
OSC["피어쌍위상차 seg16"]=peer_osc(d["rp16"]); OSC["피어쌍위상차 seg128"]=peer_osc(d["rpF"])
# --- 난이도 ---
D1=np.full((NB,81),-1.0); stuck=np.zeros(NB,bool)
for b in range(NB):
    rnd,_,g,st=propagate(X[b].astype(int)); stuck[b]=st
    r=rnd.astype(float); mx=r.max()
    r[(rnd<0)&blank[b]]=mx+1          # 전파로 못 푼 칸 = 가장 어려운 등급
    D1[b]=r
Yp=(Y+1)
chg=(P[1:]!=P[:-1])                                        # [SEG-1,NB,81]
D2=np.where(chg.any(0), chg.shape[0]-np.argmax(chg[::-1],0), 0)   # 마지막으로 바뀐 seg (1-based, 0=한 번도 안 바뀜)
D2_16=np.where(chg[:15].any(0), 15-np.argmax(chg[:15][::-1],0), 0)
D3F=(P[-1]!=Yp).astype(float); D3_16=(P[15]!=Yp).astype(float)
DIF={"D1 솔버 라운드":D1,"D2 커밋seg(128)":D2.astype(float),"D2 커밋seg(16)":D2_16.astype(float),"D3 오답@128":D3F,"D3 오답@16":D3_16}
print(f"퍼즐 {NB}  빈칸 {blank.sum()}  전파로 막힌 퍼즐 {stuck.sum()}   seg128 완답 {(P[-1]==Yp).all(1).sum()}  seg16 완답 {(P[15]==Yp).all(1).sum()}")
print(f"빈칸 오답 비율: seg16 {D3_16[blank].mean():.3f}  seg128 {D3F[blank].mean():.3f}")
# --- 상관: 빈칸만, 풀링 + 퍼즐별 ---
print("\n=== Spearman (빈칸만) — 풀링 ρ  |  퍼즐별 ρ 평균±sd (n) ===")
hdr="진동지표".ljust(26)+"".join(k.ljust(22) for k in DIF); print(hdr)
for ok,o in OSC.items():
    row=ok.ljust(26)
    for dk,dd in DIF.items():
        pooled=sp(o[blank],dd[blank])
        per=[]
        for b in range(NB):
            mb=blank[b]; a=o[b,mb]; c=dd[b,mb]
            if mb.sum()>=10 and np.std(c)>0 and np.std(a)>0: per.append(sp(a,c))
        per=np.array(per)
        row+=f"{pooled:+.3f} | {per.mean():+.3f}±{per.std():.2f}({len(per)})".ljust(22)
    print(row)
# --- 구간 평균: 피어쌍 seg128 / 칸위상 seg128 을 솔버 라운드·정오로 ---
for key in ["피어쌍위상차 seg128","칸위상 seg128","피어쌍위상차 seg16"]:
    o=OSC[key]; print(f"\n=== {key}: 난이도 구간별 평균 (빈칸만) ===")
    rr=D1[blank]; oo=o[blank]
    for r in sorted(set(rr.astype(int))):
        if r==0: continue
        msk=rr==r; print(f"  솔버 라운드 {r:2d} (n={msk.sum():5d}): {oo[msk].mean():.4f}")
    for lab,dd in [("오답@128",D3F),("오답@16",D3_16)]:
        c=dd[blank]; print(f"  {lab}=0: {oo[c==0].mean():.4f} (n={int((c==0).sum())})   {lab}=1: {oo[c==1].mean():.4f} (n={int((c==1).sum())})")
    cs=D2[blank]
    for lo,hi in [(0,0),(1,8),(9,16),(17,32),(33,64),(65,128)]:
        msk=(cs>=lo)&(cs<=hi)
        if msk.sum(): print(f"  커밋 seg {lo:3d}~{hi:3d} (n={msk.sum():5d}): {oo[msk].mean():.4f}")
# 단서 vs 빈칸 (참고)
o=OSC["피어쌍위상차 seg128"]; print(f"\n참고: 단서 칸 진동 {o[~blank].mean():.4f}  vs 빈칸 {o[blank].mean():.4f}")
