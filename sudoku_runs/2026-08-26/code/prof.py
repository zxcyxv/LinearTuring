import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
BS=128
def mk(ckpt):
    cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, loops=16, causal=False, boundary_mlp=True, ckpt=ckpt, blocks_per_seg=8, block_inj=True)
    with torch.device("cuda"): m = LT(cfg)
    return m.train()
batch = dict(inputs=torch.randint(1,11,(BS,81),dtype=torch.int32,device="cuda"), labels=torch.randint(2,11,(BS,81),dtype=torch.int32,device="cuda"), puzzle_identifiers=torch.zeros(BS,dtype=torch.int32,device="cuda"))
def bench(m, name, amp=False, n=10):
    with torch.device("cuda"): carry = m.initial_carry(batch)
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            c, out = m(carry, batch); loss = F.cross_entropy(out["logits"].float().reshape(-1,11), batch["labels"].long().reshape(-1))
        loss.backward()
    for _ in range(3): step()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): step()
    torch.cuda.synchronize(); dt=(time.time()-t)/n
    print(f"{name:40s} {dt*1000:7.1f} ms/step  ({1/dt:.2f} it/s)  mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True); torch.cuda.reset_peak_memory_stats()
m = mk(True);  bench(m, "baseline ckpt=True fp32")
m = mk(False); bench(m, "ckpt=False fp32")
bench(m, "ckpt=False bf16 autocast", amp=True)
# matrix_exp 단독 비용
S = torch.randn(8,832,832,device="cuda",requires_grad=True)
torch.cuda.synchronize(); t=time.time()
for _ in range(10):
    Q = torch.matrix_exp(S - S.transpose(-1,-2)); Q.sum().backward()
torch.cuda.synchronize(); print(f"matrix_exp fwd+bwd (8x832²): {(time.time()-t)/10*1000:.1f} ms")
