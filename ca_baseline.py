"""같은 CA 과제에 대한 외부 기준선.
   biGRU  : 순환망이 정말 못 하는지 직접 확인 (우리 모델이 양방향이므로 양방향으로 맞춤)
   CNN-k  : 커널 3짜리 k층 — 과제와 구조가 일치하는 '정답 귀납편향'. 상한 역할.
   CNN-2  : 층수를 k보다 적게 준 경우 — 깊이 부족이 실제로 병목인지 확인."""
import argparse, json, time
import torch, torch.nn as nn, torch.nn.functional as F
from ca_task import batch

class BiGRU(nn.Module):
    def __init__(s, d=128, L=2):
        super().__init__(); s.emb=nn.Embedding(2,d)
        s.rnn=nn.GRU(d,d//2,num_layers=L,batch_first=True,bidirectional=True)
        s.out=nn.Linear(d,2)
    def forward(s,x): return s.out(s.rnn(s.emb(x))[0])

class CNN(nn.Module):
    def __init__(s, d=128, L=3):
        super().__init__(); s.emb=nn.Embedding(2,d)
        s.net=nn.Sequential(*sum([[nn.Conv1d(d,d,3,padding=1),nn.GELU()] for _ in range(L)],[]))
        s.out=nn.Linear(d,2)
    def forward(s,x): return s.out(s.net(s.emb(x).transpose(1,2)).transpose(1,2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch",required=True); ap.add_argument("--rule",type=int,default=110)
    ap.add_argument("--k",type=int,default=4); ap.add_argument("--T",type=int,default=32)
    ap.add_argument("--L",type=int,default=3); ap.add_argument("--steps",type=int,default=2000)
    ap.add_argument("--bs",type=int,default=256); ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    dev="cuda"; torch.manual_seed(a.seed); g=torch.Generator(device=dev).manual_seed(a.seed)
    m={"gru":BiGRU,"cnn":CNN}[a.arch](L=a.L).to(dev)
    n=sum(p.numel() for p in m.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,a.lr,total_steps=a.steps,pct_start=0.15)
    t0=time.time()
    for i in range(a.steps):
        x,y=batch(a.bs,a.T,a.k,a.rule,dev,g)
        F.cross_entropy(m(x).reshape(-1,2),y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step(); opt.zero_grad()
    with torch.no_grad():
        x,y=batch(8192,a.T,a.k,a.rule,dev,g); p=m(x).argmax(-1)
        cell=(p==y).float().mean().item(); ex=(p==y).all(-1).float().mean().item()
    tag=f"{a.arch}L{a.L}_r{a.rule}_k{a.k}"
    print(f"[{tag}] 파라미터 {n:,} 셀 {cell:.4f} 전체일치 {ex:.4f} ({time.time()-t0:.0f}s)")
    json.dump({"arch":a.arch,"L":a.L,"rule":a.rule,"k":a.k,"params":n,"cell":cell,"exact":ex},
              open(f"runs/base_{tag}.json","w"))
if __name__=="__main__": main()
