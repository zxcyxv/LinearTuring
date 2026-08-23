"""B16(step 19530)이 '완답한' 퍼즐만 대상: 커밋 시점·소거 곡선·헤드-피어 정렬."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
torch.set_grad_enabled(False)
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")).to(torch.int32)
depth = np.load(f"{D}/cell_depth.npy")
N, BS = len(inp), 128

cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, seg_steps=0,
           loops=16, grid=9, ckpt=False, boundary_mlp=True, forward_dtype="float32",
           causal=False, blocks_per_seg=16, block_inj=True)
with torch.device("cuda"):
    lt = LT(cfg)
st = torch.load("/workspace/LinearTuring/refs/URM/checkpoints/lt1k_R1B16/step_19530.pt",
                map_location="cuda", weights_only=False)
sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in st["model_state_dict"].items()}
ms, us = lt.load_state_dict(sd, strict=False)
lt.eval(); inner = lt.inner; m = inner.core

# ── 1패스: 완답 퍼즐 식별 (일반 forward) ─────────────────────
def batchify(idx):
    n = len(idx)
    b = dict(inputs=torch.zeros(BS,81,dtype=torch.int32), labels=torch.zeros(BS,81,dtype=torch.int32),
             puzzle_identifiers=torch.zeros(BS, dtype=torch.int32))
    b["inputs"][:n] = inp[idx]; b["labels"][:n] = lab[idx]
    return {k: v.cuda() for k, v in b.items()}, n
solved = np.load("/workspace/LinearTuring/sudoku_runs/2026-08-23/interp_solved.npz")["solved_idx"]
print(f"완답 퍼즐: {len(solved)}/2048")
S = solved[:128]                                   # 분석 표본

# ── 2패스: 수동 전개 — 스텝별 argmax·엔트로피·헤드 피어질량 기록 ──
# 피어 마스크 (행·열·박스, 자기 제외)
pm = np.zeros((81, 81), bool)
for t in range(81):
    r, c = t // 9, t % 9
    for u in range(81):
        if u == t: continue
        r2, c2 = u // 9, u % 9
        if r2 == r or c2 == c or (r2//3 == r//3 and c2//3 == c//3): pm[t, u] = True
peer = torch.tensor(pm, device="cuda")
TSTEPS = 16 * 16                                    # loops × blocks
P = len(S)
am_traj = np.zeros((P, TSTEPS, 81), np.int8)        # argmax 궤적
en_traj = np.zeros((P, TSTEPS, 81), np.float16)     # 엔트로피 궤적
head_peer = torch.zeros(TSTEPS, 8, device="cuda")   # |a| 피어 질량비
head_negpeer = torch.zeros(TSTEPS, 8, device="cuda")# 피어 위 음수 질량비
nb = 0
for i in range(0, P, BS):
    idx = S[i:i+BS]
    batch, n = batchify(list(idx))
    fc = m.kernel_fast(); AB = m.W_C(); dt = 1.0
    injv = inner._injection(batch)
    h = inner.init_hidden.expand(BS, 81, 832).clone()
    t = 0
    for loop in range(16):
        for blk in range(16):
            g_, u_ = inner.b_gate_up(h).chunk(2, dim=-1)
            h = inner.b_carry * h + inner.b_down(F.silu(g_) * u_)
            h = h + inner.inj_gate * injv
            h = m.phi(h, dt / 2)
            f, a, *_ = m.field(h, None, None, None, AB, fast_ctx=fc)
            h = h + dt * f
            h = m.phi(h, dt / 2)
            aa = a.abs()                              # [B,H,81,81]
            tot = aa.sum((0, 2, 3)) + 1e-9
            head_peer[t] += aa[:, :, peer].sum((0, 2)) / tot * n / BS
            neg = (a.clamp(max=0).abs())[:, :, peer].sum((0, 2))
            head_negpeer[t] += neg / (aa[:, :, peer].sum((0, 2)) + 1e-9) * n / BS
            lg = m.w_cls(h)[:n, :, 1:]                # 값 1..9 만 (PAD·빈칸 토큰 제외 아님 — vocab 11: 0 PAD,1 빈칸? labels 2..10)
            pr = lg.softmax(-1)
            am_traj[i:i+n, t] = (lg.argmax(-1) + 1).to(torch.int8).cpu().numpy()
            en_traj[i:i+n, t] = (-(pr * (pr + 1e-9).log()).sum(-1)).to(torch.float16).cpu().numpy()
            t += 1
    nb += 1
head_peer /= nb; head_negpeer /= nb
fin = lab.numpy()[S]                                # 정답 (토큰 공간 2..10 — am_traj 와 동일 공간)

# ── 분석 A: 커밋 시점 (마지막으로 정답에서 이탈한 다음 스텝) ──
correct = (am_traj == fin[:, None, :])              # [P,T,81]
commit = np.full((P, 81), TSTEPS, np.int32)
rev = correct[:, ::-1, :]
never_wrong_after = np.cumprod(rev, axis=1).astype(bool)
for p in range(P):
    for c in range(81):
        k = never_wrong_after[p, :, c].sum()
        commit[p, c] = TSTEPS - k                    # 이 스텝부터 끝까지 정답 유지
giv = (inp.numpy()[S] > 1)
dep = depth[S]
classes = [("주어짐", giv), ("전파 1-2", (dep>=1)&(dep<3)), ("전파 3-5", (dep>=3)&(dep<6)),
           ("전파 6+", dep>=6), ("탐색", dep==-1)]
print("\n── A. 커밋 시점 (256 스텝 중, 완답 퍼즐만) ──")
print(f"{'클래스':10s} {'중앙값':>6s} {'평균':>7s} {'90%ile':>7s}")
for nm, msk in classes:
    v = commit[msk]
    print(f"{nm:10s} {np.median(v):6.0f} {v.mean():7.1f} {np.percentile(v,90):7.0f}")
r = np.corrcoef(dep[dep>0].astype(float), commit[dep>0])[0,1]
print(f"전파확정 칸에서 corr(전파깊이, 커밋시점) = {r:.3f}")

# ── 분석 B: 소거 곡선 (loop 말 기준 16점, 클래스별 평균 엔트로피) ──
loop_end = en_traj[:, 15::16, :]                    # [P,16,81]
print("\n── B. 엔트로피 (loop 말, nats) ──")
print("loop:      " + " ".join(f"{i:5d}" for i in [0,3,7,11,15]))
for nm, msk in classes:
    e = loop_end[:, :, :][:, :, np.zeros(81,bool) | False]  # placeholder
    e = loop_end[:, :, np.where(msk.any(0) if msk.ndim>1 else msk)[0]] if msk.ndim==1 else None
    # 클래스 마스크가 퍼즐별로 다르므로 마스크 평균으로
    em = (loop_end * msk[:, None, :]).sum((0,2)) / (msk.sum() + 1e-9)
    print(f"{nm:10s} " + " ".join(f"{em[i]:5.2f}" for i in [0,3,7,11,15]))

# ── 분석 C: 헤드-피어 정렬 ──
hp = head_peer.mean(0).cpu().numpy(); hn = head_negpeer.mean(0).cpu().numpy()
print("\n── C. 헤드별 |a| 질량의 피어(행·열·박스 20칸) 비중 [무작위 기대 20/80=0.25] ──")
print("헤드:      " + " ".join(f"{i:6d}" for i in range(8)))
print("피어비중:  " + " ".join(f"{v:6.3f}" for v in hp))
print("피어중음수:" + " ".join(f"{v:6.3f}" for v in hn))
np.savez("/workspace/LinearTuring/sudoku_runs/2026-08-23/interp_solved.npz",
         commit=commit, dep=dep, giv=giv, head_peer=head_peer.cpu().numpy(),
         head_negpeer=head_negpeer.cpu().numpy(), solved_idx=S)
print("\n→ sudoku_runs/2026-08-23/interp_solved.npz")
