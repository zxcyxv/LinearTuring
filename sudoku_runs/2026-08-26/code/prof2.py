import sys, time, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
BS=128
cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, loops=16, causal=False, boundary_mlp=True, ckpt=False, blocks_per_seg=8, block_inj=True)
# ── (1) bf16 추론 정합: R1B8 체크포인트, 512 퍼즐 ──
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")[:512]).to(torch.int32); lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")[:512]).to(torch.int32)
st = torch.load("/workspace/LinearTuring/sudoku_runs/2026-08-23/checkpoints/R1B8_step9765.pt", map_location='cuda', weights_only=False)
sd = {(k.replace("model.", "", 1) if k.startswith("model.") else k): v for k, v in st["model_state_dict"].items()}
with torch.device("cuda"): m = LT(cfg)
m.load_state_dict(sd, strict=True); m.eval()
with torch.no_grad():
    for amp in (False, True):
        ok=[]
        for i in range(0,512,BS):
            b=slice(i,i+BS); batch={k:v.cuda() for k,v in dict(inputs=inp[b],labels=lab[b],puzzle_identifiers=torch.zeros(BS,dtype=torch.int32)).items()}
            with torch.device("cuda"): carry=m.initial_carry(batch)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=amp):
                for _ in range(16): carry,out=m(carry,batch)
            ok.append((out["logits"].float().argmax(-1).to(torch.int32).cpu()==lab[b]).numpy())
        ok=np.concatenate(ok); print(f"R1B8 추론 amp={amp}: 셀 {ok.mean():.4f} 완답 {ok.all(1).sum()}/512", flush=True)
# ── (2) 학습 스텝 속도: compile ──
m.train(); batch = dict(inputs=torch.randint(1,11,(BS,81),dtype=torch.int32,device="cuda"), labels=torch.randint(2,11,(BS,81),dtype=torch.int32,device="cuda"), puzzle_identifiers=torch.zeros(BS,dtype=torch.int32,device="cuda"))
def bench(mod, name, amp, n=10):
    with torch.device("cuda"): carry = m.initial_carry(batch)
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            c, out = mod(carry, batch); loss = F.cross_entropy(out["logits"].float().reshape(-1,11), batch["labels"].long().reshape(-1))
        loss.backward()
    for _ in range(3): step()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): step()
    torch.cuda.synchronize(); dt=(time.time()-t)/n; print(f"{name:40s} {dt*1000:7.1f} ms/step ({1/dt:.2f} it/s)", flush=True)
bench(m, "eager fp32", False); bench(m, "eager bf16", True)
try:
    mc = torch.compile(m, dynamic=False)
    bench(mc, "compile fp32", False); bench(mc, "compile bf16", True)
except Exception as e: print("compile 실패:", repr(e)[:300])
# ── (3) matrix_exp vs Cayley ──
S = torch.randn(8,832,832,device="cuda",requires_grad=True); I = torch.eye(832,device="cuda")
for name, fn in (("matrix_exp", lambda A: torch.matrix_exp(A)), ("cayley solve", lambda A: torch.linalg.solve(I - A, I + A))):
    for _ in range(2): fn(S - S.transpose(-1,-2)).sum().backward()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(10): fn(S - S.transpose(-1,-2)).sum().backward()
    torch.cuda.synchronize(); print(f"{name:14s} fwd+bwd 8x832²: {(time.time()-t)/10*1000:.1f} ms")
