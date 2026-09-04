import sys, os, time, torch, argparse
sys.path.insert(0, "/workspace/LinearTuring/2026-09-04/analysis")
ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--fix", default="none")
ap.add_argument("--exp", type=float, default=4.0)   # none | nopersist | coord | maxauto
a0 = ap.parse_args()

import torch._inductor.config as ic
if a0.fix == "nopersist":
    ic.triton.persistent_reductions = False
elif a0.fix == "coord":
    ic.coordinate_descent_tuning = True
elif a0.fix == "maxauto":
    ic.max_autotune = True
    ic.max_autotune_pointwise = True

import bench_train as B
args = argparse.Namespace(bs=a0.bs, grid=9, d=832, heads=8, blocks=8, layers=2, expansion=a0.exp,
                          vocab=11, stdp=1, target="faithful", window="psi", compile=True,
                          patch_emb=False)
torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = True
head, T = B.build(args); batch = B.make_batch(args, T); opt = B.make_opt(head)
with torch.device('cuda'): carry = head.initial_carry(batch)
for _ in range(8): carry = B.one_step(head, opt, carry, batch, args.bs)
torch.cuda.synchronize()
ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(15)]
for i in range(15):
    ev[i][0].record(); carry = B.one_step(head, opt, carry, batch, args.bs); ev[i][1].record()
torch.cuda.synchronize()
ms = sorted(a.elapsed_time(b) for a, b in ev)
np_=sum(p.numel() for p in head.parameters())
print(f"RESULT bs={a0.bs} exp={a0.exp} params={np_/1e6:5.2f}M fix={a0.fix:10s} {ms[len(ms)//2]:8.2f} ms  ({1000/ms[len(ms)//2]:.2f} it/s)", flush=True)
