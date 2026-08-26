"""보강: ① 선형 대체 프로브 float64 (held-out) ② loop별(0,1,2,15) 블록 간 Δa/‖a‖·Δh ③ 블록 절단(keep k) ④ 어텐션 놕아웃(a=0)."""
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
def run(batch, mode="full", W=None, keep=None, rec=None, trace=None):
    h = inner.init_hidden.expand(BS, 81, -1).clone(); inj = inner._injection(batch)
    for loop in range(16):
        a_prev = h_prev = None
        for k in range(NB if keep is None else keep):
            if k == 0 or cfg.get("block_inj"):
                sw = swiglu(h) if mode != "lin" else h @ W
                if rec is not None and loop % 5 == 0: rec.append((h.reshape(-1, d).double().cpu(), swiglu(h).reshape(-1, d).double().cpu()))
                h = inner.b_carry * h + sw + inner.inj_gate * inj
            for s in range(nstep):
                hh = core.phi(h, dt / 2)
                if mode == "noattn":
                    f = hh @ core.lam.t()  # 수송·b 항 제거 (a=0)
                else:
                    f, a, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc)
                    if trace is not None and a_prev is not None: trace.setdefault(f"da_L{loop}", []).append(((a - a_prev).norm() / a.norm()).item())
                    a_prev = a
                h = core.phi(hh + dt * f, dt / 2)
            if trace is not None:
                if h_prev is not None: trace.setdefault(f"dh_L{loop}", []).append(((h - h_prev).norm() / h.norm()).item())
                h_prev = h
                trace.setdefault(f"acc_L{loop}", []).append((core.w_cls(h).argmax(-1).to(torch.int32).cpu() == batch["labels"].cpu()).float().mean().item())
    return core.w_cls(h).argmax(-1).to(torch.int32).cpu()
def evaluate(mode="full", W=None, keep=None, lo=512, hi=1024, rec=None, trace=None):
    ok = []
    for i in range(lo, hi, BS):
        b = slice(i, i + BS)
        batch = {k: v.cuda() for k, v in dict(inputs=inp[b], labels=lab[b], puzzle_identifiers=torch.zeros(BS, dtype=torch.int32)).items()}
        p = run(batch, mode, W, keep, rec, trace); ok.append((p == lab[b]).numpy())
    ok = np.concatenate(ok); return {"cell": float(ok.mean()), "exact": int(ok.all(1).sum()), "N": len(ok)}
res = {}; pairs = []; tr = {}
res["full"] = evaluate(lo=0, hi=512, rec=pairs, trace=tr)
res["trace"] = {k: (float(np.mean(v)) if k.startswith("d") else [round(x, 4) for x in np.array(v).reshape(-1, 4).mean(0)] if False else float(np.mean(v))) for k, v in tr.items() if k in ("da_L0","da_L5","da_L10","da_L15","dh_L0","dh_L5","dh_L10","dh_L15","acc_L0","acc_L1","acc_L2","acc_L3","acc_L5","acc_L15")}
X = torch.cat([p[0] for p in pairs]).cuda(); Y = torch.cat([p[1] for p in pairs]).cuda()
sub = torch.randperm(len(X), device="cuda")[:300000]; X, Y = X[sub], Y[sub]
Xm, Ym = X.mean(0), Y.mean(0); Xc, Yc = X - Xm, Y - Ym
for lam in (1e-3, 1e-1):
    W = torch.linalg.solve(Xc.T @ Xc + lam * len(X) * torch.eye(d, device="cuda", dtype=torch.float64), Xc.T @ Yc)
    r2 = float(1 - ((Xc @ W - Yc) ** 2).sum() / (Yc ** 2).sum())
    Wf = W.float(); bias = (Ym - Xm @ W).float()
    # 편향 포함 아핀 치환: sw ≈ h W + bias
    class Aff: pass
    res[f"affine_ridge{lam}"] = {"R2": r2, **evaluate("lin", W=Wf)}  # bias 는 전역 상수라 주입 게이트 b 와 동류 — 생략 시 영향 확인용

res["heldout_full"] = evaluate()
res["noattn"] = evaluate("noattn")
if NB > 1:
    res["keep"] = {k: evaluate(keep=k) for k in (1, 2, 3, 4, 6) if k < NB}
print(json.dumps(res, ensure_ascii=False)); json.dump(res, open(out, "w"), indent=1)
