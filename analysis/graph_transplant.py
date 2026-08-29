"""학습된 어텐션 그래프의 이식 시험: 원판 체크포인트의 a_tn (헤드별·스텝별) 을 손으로 짠 9-슬롯 분포 동역학(배제만)의 결합으로 넣는다.
  결과(STDP.md §6): 손 피어 그래프 6/0.493, step-8 a(피어-음 헤드 합) 5/0.500 — 동등. step 32/128 은 0.42/0.32 (내용이 섞여 흐려짐).
사용: python graph_transplant.py"""
import numpy as np, torch, math
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT
torch.set_grad_enabled(False); torch.manual_seed(0)
m=load_lt(CKPT_DEFAULT,bilinear=True,batch_size=128); inner=m.inner; core=inner.core; fc=core.kernel_fast(); AB=core.W_C()
inp,lab,_=load_test(256); N=len(inp); I=inp.cpu().numpy(); G=lab.cpu().numpy()-2; Gt=torch.tensor(G,device='cuda'); bl=torch.tensor(I==1,device='cuda')
PM=torch.tensor(peer_mask(),device='cuda',dtype=torch.float32)
# 학습된 그래프 두 시점
def graph_at(stage):
    A=torch.zeros(N,81,81,device='cuda')
    for b in range(0,N,128):
        x=inp[b:b+128]; n=len(x); inj=inner._injection(make_batch(x,x)); h=inner.init_hidden.expand(n,81,-1).clone()
        if stage=='init':
            h=inner._boundary(h)+inner.inj_gate*inj; a,*_=core.attn_fast(core.phi(h,.5),*fc,AB=AB)
        else:
            rec={}
            def hook(loop,blk,st,hh,aa):
                if st=='post_step' and aa is not None: rec['a']=aa
            rollout(m,make_batch(x,x),hook=hook); a=rec['a']
        A[b:b+n]=a.sum(1)          # 헤드 합
    return A
def solve(K, dt=0.1, steps=2000, scale=1.0):
    giv=torch.nn.functional.one_hot(torch.tensor(np.where(I==1,0,I-2),device='cuda').long(),9).float(); u=torch.zeros(N,81,9,device='cuda')
    for s in range(steps):
        q=torch.softmax(u,-1); q=torch.where(bl[...,None],q,giv)
        msg=scale*torch.einsum('btn,bnd->btd',K,q); grad_u=q*(msg-(q*msg).sum(-1,keepdim=True)); u=u-dt*grad_u*9
    P=torch.softmax(u,-1).argmax(-1)
    return int(((P==Gt)|~bl).all(-1).sum()), round(float(((P==Gt)&bl).sum()/bl.sum()),3)
print("손 그래프(피어=1):", solve(PM[None].expand(N,-1,-1)))
for st in ['init','final']:
    A=graph_at(st); Ad=A.clone(); Ad.diagonal(dim1=1,dim2=2).zero_()
    print(f"학습 그래프 [{st}]: 피어 평균 a {float((A*PM).sum()/(PM.sum()*N)):+.3f}, 비피어 평균 {float((A*(1-PM)).sum()/((1-PM).sum()*N)):+.3f}")
    for sc in [1.0,3.0,10.0]: print(f"   K = −a (헤드합), 배율 {sc}: ", solve(-Ad, scale=sc))
    print(f"   K = −a 를 피어에만 (마스크 곱):", solve(-Ad*PM[None], scale=3.0), "   K = 부호만 (−sign a):", solve(-torch.sign(Ad), scale=1.0))
print("\n=== 헤드별 / 중간 시점 ===")
def graph_heads(step_at):
    A=torch.zeros(N,8,81,81,device='cuda')
    for b in range(0,N,128):
        x=inp[b:b+128]; n=len(x); rec={}
        def hook(loop,blk,st,hh,aa):
            if st=='post_step' and loop*8+blk+1==step_at: rec['a']=aa.clone()
        rollout(m,make_batch(x,x),hook=hook); A[b:b+n]=rec['a']
    return A
for step_at in [8,32,128]:
    A=graph_heads(step_at); pm_mean=[(float((A[:,h]*PM).sum()/(PM.sum()*N))) for h in range(8)]
    neg=[h for h in range(8) if pm_mean[h]<-0.05]; Asel=A[:,neg].sum(1); Asel.diagonal(dim1=1,dim2=2).zero_()
    print(f"step {step_at:3d}: 헤드별 피어 평균 a {[round(v,2) for v in pm_mean]} → 음인 헤드 {neg}")
    print(f"   K=−a (음 헤드만 합): {solve(-Asel,scale=3.0)}   피어 마스크 곱: {solve(-Asel*PM[None],scale=3.0)}   음 헤드의 −sign: {solve(-torch.sign(Asel),scale=1.0)}")
