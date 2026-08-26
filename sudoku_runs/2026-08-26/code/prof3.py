"""compile+bf16 실제 스텝에서 W_C 파라미터화만 교체: matrix_exp vs Cayley. (학습 런과 GPU 공유 — 절대값은 오염, 상대 비교용)"""
import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
from model1 import Model1
BS=128
cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, loops=16, causal=False, boundary_mlp=True, ckpt=False, blocks_per_seg=8, block_inj=True, bilinear=True, amp=True)
batch = dict(inputs=torch.randint(1,11,(BS,81),dtype=torch.int32,device="cuda"), labels=torch.randint(2,11,(BS,81),dtype=torch.int32,device="cuda"), puzzle_identifiers=torch.zeros(BS,dtype=torch.int32,device="cuda"))
def W_C_cayley(self):
    S = self.wc_skew - self.wc_skew.transpose(-1, -2); I = torch.eye(self.d, device=S.device, dtype=S.dtype)
    Q = torch.linalg.solve(I - S, I + S); AB = Q[:, :self.dh, :]
    return AB[:, :self.p, :], AB[:, self.p:, :]
def bench(name, n=15):
    with torch.device("cuda"): m = LT(cfg); m.train()
    mc = torch.compile(m, dynamic=False)
    with torch.device("cuda"): carry = m.initial_carry(batch)
    def step():
        c, out = mc(carry, batch); F.cross_entropy(out["logits"].reshape(-1,11), batch["labels"].long().reshape(-1)).backward()
    for _ in range(3): step()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): step()
    torch.cuda.synchronize(); dt=(time.time()-t)/n; print(f"{name:20s} {dt*1000:6.1f} ms/step ({1/dt:.1f} it/s)", flush=True)
bench("matrix_exp"); Model1.W_C = W_C_cayley; bench("cayley"); Model1.W_C = Model1.W_C  # noqa
