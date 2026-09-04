"""bs32 compiled 에서 GPU 가 90% 유휴인 490ms 가 CPU 의 무엇인지 특정한다."""
import sys, os, time, torch, argparse
sys.path.insert(0, "/workspace/LinearTuring/2026-09-04/analysis")
import bench_train as B

ap = argparse.ArgumentParser(); ap.add_argument("--bs", type=int, default=32); a0 = ap.parse_args()
args = argparse.Namespace(bs=a0.bs, grid=9, d=832, heads=8, blocks=8, layers=2, expansion=4.0,
                          vocab=11, stdp=1, target="faithful", window="psi", compile=True,
                          patch_emb=False)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
head, T = B.build(args)
batch = B.make_batch(args, T)
opt = B.make_opt(head)
with torch.device('cuda'):
    carry = head.initial_carry(batch)
for _ in range(8):
    carry = B.one_step(head, opt, carry, batch, args.bs)
torch.cuda.synchronize()

# 1) 메모리 할당자 통계 (retry 는 cudaFree 전체동기화를 뜻한다)
m0 = torch.cuda.memory_stats()
t = time.perf_counter()
for _ in range(10):
    carry = B.one_step(head, opt, carry, batch, args.bs)
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / 10 * 1e3
m1 = torch.cuda.memory_stats()
print(f"[bs{args.bs}] step {dt:.1f} ms")
for k in ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free",
          "num_sync_all_streams", "allocation.all.allocated", "segment.all.allocated"):
    d = m1.get(k, 0) - m0.get(k, 0)
    print(f"    {k:32s} +{d}   (총 {m1.get(k,0)})")

# 2) CPU 프로파일 — 자기 CPU 시간 상위
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
    for _ in range(3):
        carry = B.one_step(head, opt, carry, batch, args.bs)
    torch.cuda.synchronize()
print(pr.key_averages().table(sort_by="self_cuda_time_total", row_limit=22,
                              max_name_column_width=70))
