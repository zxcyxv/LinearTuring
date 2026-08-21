"""사양이 직접 지목한 토이 과제로 절제 실험을 반복한다.
   parity: 길이 T 의 이진열의 XOR — 전 토큰의 곱이라 T차 다항식이 필요하다.
           R 번의 3차 재귀는 최대 3^R 차를 만들 수 있으므로 원리적으로 표현 가능하고,
           '재귀의 h-비선형성이 실제로 쓰이는가' 를 MNIST 보다 훨씬 강하게 압박한다."""
import argparse, json, math, time
import numpy as np, torch, torch.nn.functional as F
from model1 import Model1

def batch(bs, T, dev, gen):
    x = torch.randint(0, 2, (bs, T), device=dev, generator=gen)
    return x, x.sum(1) % 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--H", type=int, default=4); ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--freeze_A", action="store_true"); ap.add_argument("--rand_R", default="")
    ap.add_argument("--kernel_lr_mult", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
    pos = torch.stack([torch.arange(a.T).float(), torch.zeros(a.T)], 1)      # 1D 배치
    m = Model1(d=a.d, H=a.H, R=a.R, n_classes=2, positions=pos, vocab=2,
               freeze_A=a.freeze_A).to(dev)
    kn = {"psi","theta","alpha_raw"}
    kp = [p for n,p in m.named_parameters() if n in kn]; rp=[p for n,p in m.named_parameters() if n not in kn]
    opt = torch.optim.AdamW([{"params":rp,"lr":a.lr},{"params":kp,"lr":a.lr*a.kernel_lr_mult}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt,[a.lr,a.lr*a.kernel_lr_mult],total_steps=a.steps,pct_start=0.15)
    rR = [int(v) for v in a.rand_R.split(",")] if a.rand_R else None
    net = torch.compile(m) if a.compile else m
    log=[]; t0=time.time()
    for i in range(a.steps):
        x,y = batch(a.bs, a.T, dev, gen)
        Rb = int(torch.randint(rR[0],rR[1]+1,(1,)).item()) if rR else None
        loss = F.cross_entropy(net(x, R=Rb), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
        if (i+1) % 500 == 0:
            with torch.no_grad():
                xe,ye = batch(4096, a.T, dev, gen)
                acc = (m(xe).argmax(1)==ye).float().mean().item()
            log.append({"step":i+1,"loss":loss.item(),"acc":acc})
            print(f"  [{a.tag}] step {i+1:5d} loss {loss.item():.4f} eval_acc {acc:.4f}", flush=True)
    with torch.no_grad():
        xe,ye = batch(16384, a.T, dev, gen); final=(m(xe).argmax(1)==ye).float().mean().item()
    print(f"[{a.tag}] T={a.T} freeze_A={a.freeze_A} FINAL parity acc = {final:.4f}  ({time.time()-t0:.0f}s)")
    torch.save({"model":m.state_dict(),"args":vars(a)}, f"runs/{a.tag}.pt")
    json.dump({"args":vars(a),"log":log,"final":final}, open(f"runs/{a.tag}_seq.json","w"))

if __name__ == "__main__":
    main()
