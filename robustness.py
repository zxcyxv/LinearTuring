"""카오스가 대가를 치르는가: 입력 섭동에 대한 예측 안정성 vs 리아푸노프 지수.
   λ>0 은 초기조건 민감 의존이므로, 입력 잡음이 예측을 뒤집는 비율이 λ 와 함께 커져야 한다."""
import sys, json, os; sys.path.insert(0,'.')
import numpy as np, torch, interp
SIGMAS=[0.0,0.05,0.1,0.2,0.4,0.8]
@torch.no_grad()
def run(tag, nb=1024):
    m,cfg=interp.load(tag); x,y=interp.testset(noisy=cfg['noisy'])
    xb,yb=x[:nb].to(interp.DEV),y[:nb].to(interp.DEV)
    base=m(xb).argmax(1)
    acc,flip=[],[]
    g=torch.Generator(device=interp.DEV).manual_seed(0)
    for s in SIGMAS:
        p=m(xb+s*torch.randn(xb.shape,device=interp.DEV,generator=g)).argmax(1)
        acc.append((p==yb).float().mean().item()); flip.append((p!=base).float().mean().item())
    return {"tag":tag,"sigmas":SIGMAS,"acc":acc,"flip":flip,"clean":acc[0]}
out=[run(t) for t in sys.argv[1:]]
json.dump(out,open("runs/robustness.json","w"))
for r in out:
    print(f"[{r['tag']:7s}] clean {r['clean']:.4f} | acc@σ " +
          " ".join(f"{s}:{a:.3f}" for s,a in zip(SIGMAS[1:],r['acc'][1:])) +
          " | flip@σ=0.1 {:.3f} σ=0.4 {:.3f}".format(r['flip'][2],r['flip'][4]))
