import numpy as np, torch, math, sys
from common import load_test, peer_mask
torch.manual_seed(0); rng=np.random.default_rng(0)
n=int(sys.argv[1]) if len(sys.argv)>1 else 256
inp,lab,_=load_test(n); N=len(inp); I=inp.cpu().numpy(); G=lab.cpu().numpy()-2
PM=torch.tensor(peer_mask(),device='cuda',dtype=torch.float32); Gt=torch.tensor(G,device='cuda')
def solve(Xdig, K=1.0, dt=0.1, steps=2000, kappa=0.0):
    """상태 = 슬롯 로짓 u_t ∈ R^9 (분포 q = softmax u). 에너지 E = K Σ_peer q_t·q_n (같은 슬롯 확률) − κ Σ q·q (결정 우물). 로짓 경사 하강 = 선형 배제의 국소 동역학"""
    bl=torch.tensor(Xdig<0,device='cuda'); giv=torch.nn.functional.one_hot(torch.tensor(np.where(Xdig<0,0,Xdig),device="cuda").long(),9).float()
    u=torch.zeros(N,81,9,device='cuda')
    for s in range(steps):
        q=torch.softmax(u,-1); q=torch.where(bl[...,None],q,giv)
        msg=K*torch.einsum('tn,bnd->btd',PM,q)                     # 피어들의 슬롯 점유 합 = 배제 메시지
        grad_q=msg-2*kappa*q                                        # ∂E/∂q
        grad_u=q*(grad_q-(q*grad_q).sum(-1,keepdim=True))           # softmax 야코비안
        u=u-dt*grad_u*9
    q=torch.softmax(u,-1); P=q.argmax(-1)
    return int(((P==Gt)|~bl).all(-1).sum()), round(float(((P==Gt)&bl).sum()/bl.sum()),3)
for k in [1,5,20,40]:
    X=G.copy()
    for i in range(N): X[i, rng.choice(81,k,replace=False)]=-1
    print(f"정답격자 {k:2d}칸 비움: 분포 하강 {solve(X)}   +결정우물 κ=0.5 {solve(X,kappa=0.5)}", flush=True)
X=np.where(I==1,-1,I-2)
print(f"실제 퍼즐 {N}개: 분포 하강 {solve(X)}   +우물 {solve(X,kappa=0.5)}   K=3 우물 {solve(X,K=3.0,kappa=0.5,steps=4000)}", flush=True)
