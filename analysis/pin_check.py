import torch, numpy as np
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT
torch.set_grad_enabled(False); m=load_lt(CKPT_DEFAULT,bilinear=True,batch_size=128); inner=m.inner; core=inner.core
inp,lab,_=load_test(); N=len(inp); G=(lab.cuda()-2); bl=(inp==1); pm=torch.tensor(peer_mask(),device='cuda'); rng=np.random.default_rng(0)
def solve(x):
    P=torch.zeros(N,81,dtype=torch.long,device='cuda')
    for b in range(0,N,128): P[b:b+128]=logits(m,rollout(m,make_batch(x[b:b+128],x[b:b+128])))[:,:,2:11].argmax(-1)
    return P
P0=solve(inp); ex0=((P0==G)|~bl).all(-1); print(f"베이스: 완답 {int(ex0.sum())}")
# 핀 토큰 검증: 토큰 = 숫자+2, 주어진 칸과 같은 인코딩
assert int(((inp-2)[~bl]==G[~bl]).float().mean()*100)==100, "주어진 칸 인코딩 불일치"
results={}
for K in [1,3,7,15]:
    X=inp.clone(); pins=torch.zeros(N,81,dtype=torch.bool,device='cuda')
    for i in range(N):
        cand=torch.where(bl[i])[0].cpu().numpy(); sel=rng.choice(cand,min(K,len(cand)),replace=False); pins[i,sel]=True
    X[pins]=(G+2)[pins]                                 # 정답을 주어진 칸 토큰으로
    P=solve(X); blk=bl&~pins; ex=((P==G)|~blk).all(-1)
    copied=float(((P==G)[pins]).float().mean())          # 핀 칸을 모델이 그대로 출력하는가
    lost=(ex0&~ex); gained=(~ex0&ex)
    results[K]=(lost,gained,P,X,pins)
    print(f"K={K:2d}: 핀 후 완답 {int(ex.sum())}  잃음 {int(lost.sum())}  얻음 {int(gained.sum())}  | 핀 칸 복사율 {copied:.4f}  | 깨끗한(베이스 완답) 퍼즐 {int(ex0.sum())} 중 잃음 비율 {float(lost.sum()/ex0.sum()):.3f}")
# K=1 에서 잃은 표본 세부
lost,gained,P,X,pins=results[1]; idx=torch.where(lost)[0][:4].tolist()
print(f"\n=== K=1 잃은 퍼즐 {int(lost.sum())}개 중 4개 세부 (핀=정답 1칸)")
r_=np.arange(81)//9; c_=np.arange(81)%9
for i in idx:
    t=int(torch.where(pins[i])[0]); wrong=torch.where((P[i]!=G[i])&bl[i]&~pins[i])[0].tolist()
    peers_of_pin=[w for w in wrong if pm[t,w]]
    # 궤적: 핀 유무로 첫 분기 스텝
    A={}
    for tag,x in [("무핀",inp[i:i+1]),("핀",X[i:i+1])]:
        xx=x.repeat(128,1); tr=[]
        def hook(loop,blk,st,h,a):
            if st=='post_step': tr.append(logits(m,h)[0,:,2:11].argmax(-1).cpu().numpy())
        rollout(m,make_batch(xx,xx),hook=hook); A[tag]=np.stack(tr)
    d=np.where((A["무핀"]!=A["핀"]).any(1))[0]; first=int(d[0]) if len(d) else -1
    firstcells=np.where(A["무핀"][first]!=A["핀"][first])[0] if first>=0 else []
    print(f"퍼즐 {i}: 핀 r{r_[t]}c{c_[t]}={int(G[i,t])+1}  → 새 오답 {len(wrong)}칸 {[f'r{r_[w]}c{c_[w]}({int(P[i,w])+1}/{int(G[i,w])+1})' for w in wrong[:8]]}  핀의 피어인 오답 {len(peers_of_pin)}/{len(wrong)}")
    print(f"   궤적 첫 분기 스텝 {first}, 그 스텝에 갈린 칸 {[f'r{r_[w]}c{c_[w]}' for w in firstcells[:6]]}  (핀의 피어? {[bool(pm[t,w]) for w in firstcells[:6]]})")
    # 첫 분기 칸의 스텝별 argmax (무핀 vs 핀), 8스텝 간격
    if len(firstcells):
        w=int(firstcells[0]); print(f"   칸 r{r_[w]}c{c_[w]} 정답 {int(G[i,w])+1}: 무핀 {' '.join(str(v+1) for v in A['무핀'][::8,w])} | 핀 {' '.join(str(v+1) for v in A['핀'][::8,w])}")
