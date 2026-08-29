"""증강 합집합: 논리적으로 동일한 변형(숫자 치환 × 전치/밴드/스택 셔플)을 M번 돌려 유효 격자(=유일해=정답)를 채택. 오답 채택 0.
원판 123k: 947 → 1411 (10회). STDP.md §6.8.  사용: python aug_union.py [M=10]"""
import sys, torch, numpy as np
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT
torch.set_grad_enabled(False); M=int(sys.argv[1]) if len(sys.argv)>1 else 10
m=load_lt(CKPT_DEFAULT,bilinear=True,batch_size=128); inp,lab,_=load_test(); N=len(inp); bl=(inp==1); pm=torch.tensor(peer_mask(),device='cuda'); rng=np.random.default_rng(7)
def solve(x):
    P=torch.zeros(N,81,dtype=torch.long,device='cuda')
    for b in range(0,N,128): P[b:b+128]=logits(m,rollout(m,make_batch(x[b:b+128],x[b:b+128])))[:,:,2:11].argmax(-1)
    return P
def valid(P,x):
    b=(x==1); fin=torch.where(b,P,x.long()-2); return ~((fin[:,:,None]==fin[:,None,:])&pm[None]).any(-1).any(-1)
def band_shuffle(): bands=rng.permutation(3); return np.concatenate([bands[b]*3+rng.permutation(3) for b in range(3)])
acc=valid(solve(inp),inp); print(f"기준 유효(=완답) {int(acc.sum())}")
for k in range(M):
    kind=['perm','transpose','rows','cols'][k%4]; x=inp.clone(); perm=torch.tensor(rng.permutation(9),device='cuda'); nb=~bl; x[nb]=(perm[(inp.long()-2)[nb]]+2).to(inp.dtype)
    if kind=='transpose': x=x.view(N,9,9).transpose(1,2).reshape(N,81).contiguous()
    elif kind=='rows': x=x.view(N,9,9)[:,torch.tensor(band_shuffle(),device='cuda'),:].reshape(N,81).contiguous()
    elif kind=='cols': x=x.view(N,9,9)[:,:,torch.tensor(band_shuffle(),device='cuda')].reshape(N,81).contiguous()
    v=valid(solve(x),x); new=int((v&~acc).sum()); acc|=v; print(f"증강 {k+1:2d} ({kind}): 유효 {int(v.sum())}  누적 {int(acc.sum())} (+{new})", flush=True)
