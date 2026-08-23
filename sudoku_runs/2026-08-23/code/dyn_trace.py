"""동역학 이론 기반 추적: ①Potts 부호 구조 ②진폭 채널(사건 정렬) ③에너지 수지 시계열."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
torch.set_grad_enabled(False)
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")).to(torch.int32)
solved = np.load("/workspace/LinearTuring/sudoku_runs/2026-08-23/interp_solved.npz")["solved_idx"]
S = solved[:128]; BS = 128
cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, seg_steps=0,
           loops=16, grid=9, ckpt=False, boundary_mlp=True, forward_dtype="float32",
           causal=False, blocks_per_seg=16, block_inj=True)
with torch.device("cuda"): lt = LT(cfg)
st = torch.load("/workspace/LinearTuring/refs/URM/checkpoints/lt1k_R1B16/step_19530.pt", map_location="cuda", weights_only=False)
lt.load_state_dict({k.replace("model.","",1) if k.startswith("model.") else k: v for k,v in st["model_state_dict"].items()}, strict=False)
lt.eval(); inner = lt.inner; m = inner.core
batch = dict(inputs=inp[S].cuda(), labels=lab[S].cuda(), puzzle_identifiers=torch.zeros(BS, dtype=torch.int32, device="cuda"))
lab_n, inp_n = lab[S].numpy(), inp[S].numpy()
# 피어·동일숫자 마스크
pm = np.zeros((81,81), bool)
for t in range(81):
    r,c = t//9, t%9
    for u in range(81):
        if u!=t and (u//9==r or u%9==c or (u//9//3==r//3 and u%9//3==c//3)): pm[t,u]=True
peer = torch.tensor(pm, device="cuda")
samed = torch.tensor(lab_n[:,:,None]==lab_n[:,None,:], device="cuda") & (~torch.eye(81,dtype=bool,device="cuda"))
fc = m.kernel_fast(); AB = m.W_C(); dt = 1.0
injv = inner._injection(batch); h = inner.init_hidden.expand(BS, 81, 832).clone()
amp = np.zeros((BS,16,81)); dfield = np.zeros((BS,16,81)); am = np.zeros((BS,16,81), np.int8)
E_inj = np.zeros(16); E_diss = np.zeros(16); negfrac = np.zeros(16)
potts = torch.zeros(16, 8, 4, device="cuda")   # loop × head × [피어다름, 피어같음(불가∅), 비피어같음, 비피어다름]
for loop in range(16):
    for blk in range(16):
        g_, u_ = inner.b_gate_up(h).chunk(2, dim=-1)
        h = inner.b_carry * h + inner.b_down(F.silu(g_) * u_)
        h = h + inner.inj_gate * injv
        h = m.phi(h, dt/2)
        f, a, *_ = m.field(h, None, None, None, AB, fast_ctx=fc)
        h = h + dt*f; h = m.phi(h, dt/2)
    amp[:, loop] = h.norm(dim=-1).cpu().numpy()
    dfield[:, loop] = a.sum(3).sum(1).cpu().numpy()             # Σ_m d_t^(m)
    am[:, loop] = m.w_cls(h).argmax(-1).to(torch.int8).cpu().numpy()
    diff2 = torch.cdist(h, h).pow(2)                            # [B,81,81]
    a_all = a.sum(1)                                            # 헤드 합 수송 계수
    E_inj[loop] = (a_all.clamp(max=0).abs()*diff2).sum().item()/BS
    E_diss[loop] = (a_all.clamp(min=0)*diff2).sum().item()/BS
    negfrac[loop] = (a<0).float().mean().item()
    for hh in range(8):
        av = a[:, hh]
        potts[loop, hh, 0] = av[peer.expand(BS,81,81) & ~samed].mean()
        potts[loop, hh, 2] = av[~peer.expand(BS,81,81) & samed].mean()
        potts[loop, hh, 3] = av[~peer.expand(BS,81,81) & ~samed & ~torch.eye(81,dtype=bool,device="cuda")].mean()
commit = 16 - np.cumprod((am == lab_n[:,None,:])[:, ::-1, :], 1).astype(bool).sum(1)

print("── ① Potts 부호 구조 (loop 15, 헤드별 평균 a) ──")
print(f"{'헤드':>4} {'피어(≠숫자)':>10} {'비피어(=숫자)':>12} {'비피어(≠숫자)':>12}")
for hh in range(8):
    p_ = potts[15, hh].cpu().numpy()
    print(f"{hh:>4} {p_[0]:>+10.4f} {p_[2]:>+12.4f} {p_[3]:>+12.4f}")
print(f"\n전 간선 a<0 비율: loop0 {negfrac[0]:.3f} → loop15 {negfrac[15]:.3f}   [등록 예측: MNIST ~0.48 보다 높음]")

print("\n── ③ 에너지 수지 (음수간선 주입 vs 양수간선 소산, 샘플당) ──")
print("loop:  " + " ".join(f"{i:>7d}" for i in [0,1,3,7,15]))
print("주입:  " + " ".join(f"{E_inj[i]:>7.0f}" for i in [0,1,3,7,15]))
print("소산:  " + " ".join(f"{E_diss[i]:>7.0f}" for i in [0,1,3,7,15]))

print("\n── ② 진폭 채널: 확정 사건 정렬 (offset = loop − 확정loop) ──")
sel = (inp_n == 1) & (commit >= 2) & (commit <= 13)
offs = range(-2, 3)
rows_a, rows_d = [], []
for o in offs:
    va, vd, cnt = 0.0, 0.0, 0
    for p in range(BS):
        for t in np.where(sel[p])[0]:
            lp = commit[p, t] + o
            if 0 <= lp < 16:
                va += amp[p, lp, t]; vd += dfield[p, lp, t]; cnt += 1
    rows_a.append(va/cnt); rows_d.append(vd/cnt)
print("offset:  " + " ".join(f"{o:>7d}" for o in offs))
print("‖h_t‖:  " + " ".join(f"{v:>7.3f}" for v in rows_a))
print("d_t:     " + " ".join(f"{v:>+7.3f}" for v in rows_d))
base_a = amp[sel[:, None, :].repeat(16, axis=1) if False else 0]  # (미사용)
print(f"(전 빈칸 전 loop 평균 ‖h‖ = {amp[:, :, :][np.repeat((inp_n==1)[:,None,:],16,1)].mean():.3f})")
