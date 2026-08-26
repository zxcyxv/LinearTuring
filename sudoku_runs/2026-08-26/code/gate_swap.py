"""학습 없이 R1B8 체크포인트의 경계 게이트 함수만 교체: silu → x/2 (쌍선형), x/2+x²/4 (2차), relu, identity(x), 0.5 상수(=선형)."""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
torch.set_grad_enabled(False)
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")[:512]).to(torch.int32); lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")[:512]).to(torch.int32)
st = torch.load("/workspace/LinearTuring/sudoku_runs/2026-08-23/checkpoints/R1B8_step9765.pt", map_location='cuda', weights_only=False)
sd = {(k.replace("model.", "", 1) if k.startswith("model.") else k): v for k, v in st["model_state_dict"].items()}
BS=128; cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, loops=16, causal=False, boundary_mlp=True, ckpt=False, blocks_per_seg=8, block_inj=True)
with torch.device("cuda"): m = LT(cfg)
m.load_state_dict(sd, strict=True); m.eval(); inner = m.inner; core = inner.core; fc = core.kernel_fast(); AB = core.W_C()
acts = {"silu(원판)": F.silu, "x/2 (쌍선형)": lambda x: x/2, "x/2+x²/4 (2차)": lambda x: x/2 + x*x/4, "x/2+x²/4−x⁴/48": lambda x: x/2 + x*x/4 - x**4/48,
        "relu": F.relu, "gelu": F.gelu, "x (항등 게이트)": lambda x: x, "상수 0.5 (=선형 경계)": lambda x: torch.full_like(x, 0.5)}
res = {}
for name, act in acts.items():
    ok = []
    for i in range(0, 512, BS):
        b = slice(i, i+BS); batch = {k: v.cuda() for k, v in dict(inputs=inp[b], labels=lab[b], puzzle_identifiers=torch.zeros(BS, dtype=torch.int32)).items()}
        h = inner.init_hidden.expand(BS, 81, -1).clone(); inj = inner._injection(batch)
        for loop in range(16):
            for k in range(8):
                g, u = inner.b_gate_up(h).chunk(2, -1); h = inner.b_carry*h + inner.b_down(act(g)*u) + inner.inj_gate*inj
                hh = core.phi(h, 0.5); f, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc); h = core.phi(hh + f, 0.5)
        p = core.w_cls(h).argmax(-1).to(torch.int32).cpu(); ok.append((p == lab[b]).numpy())
    ok = np.concatenate(ok); res[name] = {"cell": round(float(ok.mean()), 4), "exact": int(ok.all(1).sum())}
    print(f"{name:24s} 셀 {res[name]['cell']:.4f}  완답 {res[name]['exact']}/512", flush=True)
json.dump(res, open("gate_swap_R1B8.json", "w"), indent=1, ensure_ascii=False)
