import numpy as np, torch, math
from common import load_test, peer_mask
rng=np.random.default_rng(0); inp,lab,_=load_test(256); N=len(inp); G=lab.cpu().numpy()-2
PM=torch.tensor(peer_mask(),device='cuda',dtype=torch.float32); slots=torch.arange(9,device='cuda')*2*math.pi/9; Gt=torch.tensor(G,device='cuda')
def pair_energy(dphi,M):   # 디리클레 척력 에너지 (M 조화)
    ms=torch.arange(1,M+1,device='cuda').float(); return torch.cos(dphi[...,None]*ms).sum(-1)/M
for k in [1,5,20]:
    X=G.copy()
    for i in range(N): X[i, rng.choice(81,k,replace=False)]=-1
    bl=torch.tensor(X<0,device='cuda'); gp=torch.tensor(np.where(X<0,0,X),device='cuda').float()*2*math.pi/9
    for M in [1,4,8]:
        # 좌표별 전역 탐색(반복): 각 빈칸에 대해 9슬롯 에너지 평가 → 최소 슬롯 선택, 나머지 고정. 5회 sweep
        phi=gp.clone(); phi=torch.where(bl,slots[torch.randint(0,9,(N,81),device='cuda')],gp)
        for sweep in range(5):
            for t in range(81):
                cand=slots[None,:]                                           # [1,9]
                d=cand[:,:,None]-phi[:,None,:]                               # [N,9,81] φ_cand − φ_n
                E=(PM[t][None,None,:]*pair_energy(d,M)).sum(-1)              # [N,9]
                best=slots[E.argmin(-1)]; phi[:,t]=torch.where(bl[:,t],best,phi[:,t])
        P=torch.cos(phi[...,None]-slots).argmax(-1)
        print(f"{k:2d}칸 비움, M={M}: 전역(좌표별 최소) 완답 {int(((P==Gt)|~bl).all(-1).sum())}/{N}  빈칸셀 {float(((P==Gt)&bl).sum()/bl.sum()):.3f}")
