"""자기교정 루프 (원판 lt.py 경로): 위반 칸 탐지 → 상태 초기화 → 재전개 → 유효 격자면 채택 (오답 채택 0 보장은 유효성=정답 유일성).
  탐지 = 손 규칙(행·열·박스 피어 중 같은 argmax 가 있는 빈칸). 라벨은 완답 집계에만.
  범위 unit: 위반 칸의 유닛 빈칸 전부 초기화 / cell: 위반 칸만 / random: 위반 칸 수만큼 무작위 빈칸 (대조군)
  η, λ: 결합 기억 EMA (η=1, λ=0 이면 없음).  STDP.md §6 결과: 947 → cell 1047 / random 1062 / cell+기억 1198 / random+기억 1188
사용: python selfcorrect.py [N=2048] [ROUNDS=3] [unit|cell|random] [η] [λ] [잡음]"""
import torch, numpy as np, sys
from common import load_lt, load_test, make_batch, logits, peer_mask, CKPT_DEFAULT
torch.set_grad_enabled(False); N=int(sys.argv[1]) if len(sys.argv)>1 else 2048; ROUNDS=int(sys.argv[2]) if len(sys.argv)>2 else 3; SCOPE=sys.argv[3] if len(sys.argv)>3 else "unit"; ETA=float(sys.argv[4]) if len(sys.argv)>4 else 1.0; LAM=float(sys.argv[5]) if len(sys.argv)>5 else 0.0; NOISE=float(sys.argv[6]) if len(sys.argv)>6 else 0.0
m=load_lt(CKPT_DEFAULT,bilinear=True,batch_size=128); inner=m.inner; core=inner.core; K=8; L=16; fc=core.kernel_fast(); AB=core.W_C()
inp,lab,_=load_test(N); G=(lab.cuda()-2); bl=(inp==1); pm=torch.tensor(peer_mask(),device='cuda'); PMI=pm|torch.eye(81,dtype=torch.bool,device='cuda'); givdig=(inp-2).long()
INIT=inner.init_hidden.reshape(-1)[:core.d]
def rollout(h0,x):
    inj=inner._injection(make_batch(x,x)); h=h0.clone(); w=None
    for _ in range(L):
        for _ in range(K):
            h=inner._boundary(h)+inner.inj_gate*inj; hp=core.phi(h,.5); a,*_=core.attn_fast(hp,*fc,AB=AB)
            w=a.clone() if w is None else w+ETA*(a-w)
            f,*_=core.field(hp,None,None,a_fixed=(1-LAM)*a+LAM*w,AB=AB,fast_ctx=fc); h=core.phi(hp+f,.5)
    return h
def board(h): return logits(m,h)[:,:,2:11].argmax(-1)
def validmask(P): fin=torch.where(bl,P,inp-2); return ~((fin[:,:,None]==fin[:,None,:])&pm[None]).any(-1).any(-1)
H=torch.zeros(N,81,core.d,device='cuda'); P=torch.zeros(N,81,dtype=torch.long,device='cuda')
for b in range(0,N,128): H[b:b+128]=rollout(INIT.expand(128,81,-1),inp[b:b+128]); P[b:b+128]=board(H[b:b+128])
accepted=validmask(P); final=P.clone(); ex0=int(((final==G)|~bl).all(-1).sum()); print(f"기준: 완답 {ex0} 유효 {int(accepted.sum())}  (N={N}, 범위={SCOPE}, η={ETA}, λ={LAM}, 잡음={NOISE})")
for rnd in range(ROUNDS):
    todo=(~accepted); 
    if todo.sum()==0: break
    d=torch.where(bl,P,givdig); V=((d[:,:,None]==d[:,None,:])&pm[None]).any(-1)&bl            # 위반 칸
    tgt=((torch.einsum('bt,tn->bn',V.float(),PMI.float())>0)&bl) if SCOPE=="unit" else (V if SCOPE=="cell" else (torch.rand(N,81,device="cuda")<(V.sum(1,keepdim=True).float()/bl.sum(1,keepdim=True).float()))&bl)         # 유닛 전체 / 위반 칸만
    h=H.clone(); h[tgt]=INIT; newacc=0
    if NOISE>0: h=h+NOISE*torch.randn_like(h)*tgt[...,None]
    for b in range(0,N,128):
        hb=rollout(h[b:b+128],inp[b:b+128]); Pb=board(hb); vb=validmask(Pb) if False else None
        fin=torch.where(bl[b:b+128],Pb,inp[b:b+128]-2); vb=~((fin[:,:,None]==fin[:,None,:])&pm[None]).any(-1).any(-1)
        for j in range(len(Pb)):
            i=b+j
            if accepted[i]: continue
            if vb[j]: accepted[i]=True; final[i]=Pb[j]; newacc+=1
            else: H[i]=hb[j]; P[i]=Pb[j]
    ex=int(((final==G)|~bl).all(-1).sum()); va=int(accepted.sum())
    print(f"라운드 {rnd}: 대상 {int(todo.sum())}  위반칸/퍼즐 {V[todo].sum(1).float().mean():.1f}  초기화칸/퍼즐 {tgt[todo].sum(1).float().mean():.1f}  새 채택 {newacc}  → 완답 {ex} (+{ex-ex0}) 유효 {va}  오답 채택 {va-ex}", flush=True)
