"""① 블록 간 어텐션 그래프 변화 Δa/‖a‖·상태 고정점 ‖h_{k+1}−h_k‖/‖h‖ (loop 15)  ② 경계 SwiGLU 의 선형 대체 프로브:
   (h_in → sw_out) 쌍을 릿지 회귀로 맞춘 W 로 경계를 h + W h 로 치환(=LSH 형식) 후 성능. 학습 없음.
사용: python dyn2.py <ckpt> <cfg_json> <out_json>"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
torch.set_grad_enabled(False)
ck, over, out = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
st = torch.load(ck, map_location='cuda', weights_only=False)
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith(("model.", "_orig_mod.")) else k): v for k, v in st["model_state_dict"].items()}
BS = 128
cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
           R=8, loops=16, causal=False, boundary_mlp=True, ckpt=False, blocks_per_seg=1, block_inj=False); cfg.update(over)
with torch.device("cuda"): m = LT(cfg)
m.load_state_dict(sd, strict=True); m.eval(); inner = m.inner; core = inner.core
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")[:1024]).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")[:1024]).to(torch.int32)
NB, nstep = cfg["blocks_per_seg"], cfg["R"]; dt = 1.0 / cfg["R"]; d = 832
fc = core.kernel_fast(); AB = core.W_C()
def swiglu(h):
    g, u = inner.b_gate_up(h).chunk(2, -1); return inner.b_down(F.silu(g) * u)
def run(batch, mode="full", W=None, rec=None):
    h = inner.init_hidden.expand(BS, 81, -1).clone(); inj = inner._injection(batch)
    for loop in range(16):
        a_prev = None; h_prev_blk = None
        for k in range(NB):
            if k == 0 or cfg.get("block_inj"):
                if mode == "full": sw = swiglu(h)
                elif mode == "lin": sw = h @ W
                if rec is not None: rec.append((h.reshape(-1, d).cpu(), swiglu(h).reshape(-1, d).cpu()))
                h = inner.b_carry * h + sw + inner.inj_gate * inj
            for s in range(nstep):
                hh = core.phi(h, dt / 2); f, a, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc)
                if loop == 15 and a_prev is not None: DA.append(((a - a_prev).norm() / a.norm()).item())
                a_prev = a; h = core.phi(hh + dt * f, dt / 2)
            if loop == 15:
                if h_prev_blk is not None: DH.append(((h - h_prev_blk).norm() / h.norm()).item())
                h_prev_blk = h
            if loop == 15 and k == NB - 1: pass
        if loop == 14: h14 = h.clone()
    DSEG.append(((h - h14).norm() / h.norm()).item())
    return core.w_cls(h).argmax(-1).to(torch.int32).cpu()
def evaluate(mode, W=None, lo=512, hi=1024, rec=None):
    ok = []
    for i in range(lo, hi, BS):
        b = slice(i, i + BS)
        batch = {k: v.cuda() for k, v in dict(inputs=inp[b], labels=lab[b], puzzle_identifiers=torch.zeros(BS, dtype=torch.int32)).items()}
        p = run(batch, mode, W, rec); ok.append((p == lab[b]).numpy())
    ok = np.concatenate(ok); return {"cell": float(ok.mean()), "exact": int(ok.all(1).sum()), "N": len(ok)}
res = {}
DA, DH, DSEG = [], [], []; pairs = []
res["full"] = evaluate("full", lo=0, hi=512, rec=pairs)          # 앞 512: 회귀 데이터 수집
res["da_step_loop15"] = float(np.mean(DA)) if DA else None; res["dh_block_loop15"] = float(np.mean(DH)) if DH else None
res["dh_seg14to15"] = float(np.mean(DSEG))
X = torch.cat([p[0] for p in pairs]).cuda(); Y = torch.cat([p[1] for p in pairs]).cuda()
sub = torch.randperm(len(X), device="cuda")[:400000]; X, Y = X[sub], Y[sub]
for lam in (1e-2, 1.0):
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(d, device="cuda"), X.T @ Y)
    r2 = 1 - ((X @ W - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    DA, DH, DSEG = [], [], []
    res[f"lin_ridge{lam}"] = {"R2_fit": float(r2), **evaluate("lin", W)}
DA, DH, DSEG = [], [], []; res["full_heldout"] = evaluate("full")
DA, DH, DSEG = [], [], []; res["identity_heldout"] = evaluate("lin", torch.zeros(d, d, device="cuda"))
print(json.dumps(res, ensure_ascii=False)); json.dump(res, open(out, "w"), indent=1)
