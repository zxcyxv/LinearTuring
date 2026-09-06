"""v3 스텝 벤치: fwd+bwd+opt, 배치 128, 합성 데이터. compile 모드 × bmm 병합 유무."""
import importlib.util, sys, time, argparse, torch
ap=argparse.ArgumentParser(); ap.add_argument("--mode", default="default"); ap.add_argument("--merged", type=int, default=0); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--nopersist", type=int, default=1)
a=ap.parse_args()
spec=importlib.util.spec_from_file_location("v3","/workspace/LinearTuring/kaggle/train_kaggle_v3.py"); v3=importlib.util.module_from_spec(spec); sys.modules["v3"]=v3; spec.loader.exec_module(v3)
if a.nopersist: torch._inductor.config.triton.persistent_reductions = False
if a.merged:
    def attn_xy(self, xy, kc):
        x, y = xy; cosA, sinA, cosB, sinB = kc
        qx = x*cosA - y*sinA; qy = x*sinA + y*cosA; kx = x*cosB - y*sinB; ky = x*sinB + y*cosB
        q = torch.cat([qx, qy], -1)                                   # [B,T,H,2p]
        a  = torch.einsum('bthj,bnhj->bhtn', q, torch.cat([kx, ky], -1))
        ap = torch.einsum('bthj,bnhj->bhtn', q, torch.cat([ky, -kx], -1))
        return a, ap
    v3.LT_Inner.attn_xy = attn_xy
cfg=dict(v3.CFG); cfg.update(batch_size=128, seq_len=81, num_puzzle_identifiers=1)
torch.manual_seed(0)
with torch.device("cuda"):
    base=v3.ACTLossHead(v3.LT(cfg), "stablemax_cross_entropy", q_weight=0.5); base.train()
    model = base if a.mode=="eager" else torch.compile(base, mode=(None if a.mode=="default" else a.mode), dynamic=False)
    opts,_=v3.create_optimizers(base, cfg, 1)
    x=torch.randint(1,11,(128,81),dtype=torch.int32); y=torch.randint(2,11,(128,81)); y[x>1]=x[x>1].long()
    batch=dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(128,dtype=torch.int32))
    carry=None
    def step():
        global carry
        if carry is None: carry=base.initial_carry(batch)
        carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=set())
        (loss/128).backward()
        for o in opts:
            for g in o.param_groups: g["lr"]=1e-4
            o.step(); o.zero_grad()
    t0=time.time()
    for _ in range(12): step()
    torch.cuda.synchronize(); tc=time.time()-t0
    t0=time.time()
    for _ in range(a.steps): step()
    torch.cuda.synchronize(); dt=(time.time()-t0)/a.steps
    print(f"mode={a.mode:16s} merged={a.merged}  {dt*1000:7.1f} ms/step  {1/dt:5.2f} it/s   (warmup+compile {tc:.0f}s)  peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
