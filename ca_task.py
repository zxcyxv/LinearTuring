"""셀룰러 오토마타 k스텝 예측.
   이웃 그래프는 고정, 갱신은 상태에 대해 비선형 → A 고정본이 무너져야 하는 (b) 경로.
   Rule 90 은 GF(2) 선형이라 닫힌 해가 있고, Rule 110 은 없다 — 둘을 같이 돌려 비교한다."""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from model1 import Model1

def step(s, rule):
    """s: [B,T] int64, 고정(0) 경계"""
    l = torch.roll(s, 1, 1); l[:, 0] = 0
    r = torch.roll(s, -1, 1); r[:, -1] = 0
    idx = 4*l + 2*s + r
    tbl = torch.tensor([(rule >> i) & 1 for i in range(8)], device=s.device)
    return tbl[idx]

def batch(bs, T, k, rule, dev, gen):
    x = torch.randint(0, 2, (bs, T), device=dev, generator=gen)
    y = x.clone()
    for _ in range(k): y = step(y, rule)
    return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--k", type=int, default=1); ap.add_argument("--rule", type=int, default=90)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--H", type=int, default=4); ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--freeze_A", action="store_true")
    ap.add_argument("--no_ov", action="store_true", help="noov 구조 (W_OV=블록항등, Λ 대각)")
    ap.add_argument("--boundary_wo", action="store_true", help="τ 블록 경계에서만 W_O")
    ap.add_argument("--wo_mode", default="plain", choices=["plain","residual","orth","contract","perhead"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=int, default=1, help="총 적분 시간 배수: steps = tau*R (dt=1/R 고정)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--kernel_lr_mult", type=float, default=30.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
    pos = torch.stack([torch.arange(a.T).float(), torch.zeros(a.T)], 1)
    m = Model1(d=a.d, H=a.H, R=a.R, n_classes=2, positions=pos, vocab=2,
               freeze_A=a.freeze_A, pool=False, boundary_wo=a.boundary_wo, wo_mode=a.wo_mode,
               use_ov=not a.no_ov, lam_mode="diag" if a.no_ov else "full").to(dev)
    kn={"psi","theta","alpha_raw"}
    kp=[p for n,p in m.named_parameters() if n in kn]; rp=[p for n,p in m.named_parameters() if n not in kn]
    opt=torch.optim.AdamW([{"params":rp,"lr":a.lr},{"params":kp,"lr":a.lr*a.kernel_lr_mult}],weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,[a.lr,a.lr*a.kernel_lr_mult],total_steps=a.steps,pct_start=0.15)
    net = torch.compile(m) if a.compile else m
    log=[]; t0=time.time()
    for i in range(a.steps):
        x,y = batch(a.bs, a.T, a.k, a.rule, dev, gen)
        loss = F.cross_entropy(net(x, R=a.R, steps=a.R*a.tau).reshape(-1,2), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
        if (i+1)%500==0:
            with torch.no_grad():
                xe,ye = batch(2048, a.T, a.k, a.rule, dev, gen); p_=m(xe, R=a.R, steps=a.R*a.tau).argmax(-1)
                cell=(p_==ye).float().mean().item(); exact=(p_==ye).all(-1).float().mean().item()
            log.append({"step":i+1,"loss":loss.item(),"cell":cell,"exact":exact})
            print(f"  [{a.tag}] step {i+1:5d} loss {loss.item():.4f} 셀 {cell:.4f} 전체일치 {exact:.4f}", flush=True)
    with torch.no_grad():
        xe,ye = batch(8192, a.T, a.k, a.rule, dev, gen); p_=m(xe, R=a.R, steps=a.R*a.tau).argmax(-1)
        cell=(p_==ye).float().mean().item(); exact=(p_==ye).all(-1).float().mean().item()
    print(f"[{a.tag}] rule={a.rule} k={a.k} T={a.T} tau={a.tau} freeze_A={a.freeze_A} FINAL 셀 {cell:.4f} 전체일치 {exact:.4f} ({time.time()-t0:.0f}s)")
    torch.save({"model":m.state_dict(),"args":vars(a)}, f"runs/{a.tag}.pt")
    json.dump({"args":vars(a),"log":log,"cell":cell,"exact":exact}, open(f"runs/{a.tag}_ca.json","w"))

if __name__=="__main__": main()
