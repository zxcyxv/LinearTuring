"""v1.7 학습 1스텝(세그먼트 fwd + bwd + opt.step)의 실측 벤치.

사용:
  python bench_v17.py                                  # 기본 (compile default)
  python bench_v17.py --no-compile
  python bench_v17.py --mode max-autotune-no-cudagraphs
  python bench_v17.py --tf32
  python bench_v17.py --profile                        # 연산자별 CUDA 시간
"""
import argparse, os, sys, time, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"

ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--no-compile", action="store_true")
ap.add_argument("--mode", default="default")
ap.add_argument("--tf32", action="store_true")
ap.add_argument("--no-persist", action="store_true", default=True)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--warmup", type=int, default=8)
ap.add_argument("--profile", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args()

s = importlib.util.spec_from_file_location("tv", os.path.join(ROOT, "2026-09-09/analysis/train_v17.py"))
tv = importlib.util.module_from_spec(s); s.loader.exec_module(tv)

if a.tf32:
    torch.set_float32_matmul_precision("high")
if a.no_persist:
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False

cfg = dict(tv.CFG)
cfg.update(batch_size=a.bs, seq_len=81, num_puzzle_identifiers=1, loops=16)
with torch.device("cuda"):
    base = tv.ACTLossHead(tv.LT(cfg), "stablemax_cross_entropy", q_weight=cfg["q_weight"])
base.train()
opts, lrs = tv.create_optimizers(base, cfg, world_size=1)
model = base if a.no_compile else torch.compile(base, dynamic=False, mode=(None if a.mode == "default" else a.mode))

g = torch.Generator().manual_seed(0)
x = torch.randint(2, 11, (a.bs, 81), generator=g).int().cuda()
y = torch.randint(2, 11, (a.bs, 81), generator=g).long().cuda()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(a.bs, dtype=torch.int32, device="cuda"))

with torch.device("cuda"):
    carry = base.initial_carry(batch)

def one_step(carry):
    carry, loss, _, _, _ = model(return_keys=set(), carry=carry, batch=batch)
    loss.backward()
    for o in opts:
        o.step(); o.zero_grad(set_to_none=True)
    return carry

for _ in range(a.warmup):
    carry = one_step(carry)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

if a.profile:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5): carry = one_step(carry)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22))
else:
    t0 = time.time()
    for _ in range(a.iters):
        carry = one_step(carry)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / a.iters
    mem = torch.cuda.max_memory_allocated() / 1e9
    lbl = a.tag or f"{'eager' if a.no_compile else 'compile:'+a.mode}{' tf32' if a.tf32 else ''}"
    print(f"{lbl:42s}  {dt*1000:7.1f} ms/step   {1/dt:5.2f} it/s   peak {mem:.2f} GB")
