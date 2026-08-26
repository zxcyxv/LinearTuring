"""경계 SwiGLU 가 토큰 내 '커밋'(후보 첨예화) 연산인가: 경계 전/후 상태를 w_cls 로 읽은 로짓 엔트로피·argmax 변화·정답률 (loop 0,1,15)."""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
torch.set_grad_enabled(False)
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")[:256]).to(torch.int32).cuda(); lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")[:256]).to(torch.int32).cuda()
st = torch.load("/workspace/LinearTuring/sudoku_runs/2026-08-23/checkpoints/R1B8_step9765.pt", map_location='cuda', weights_only=False)
sd = {(k.replace("model.", "", 1) if k.startswith("model.") else k): v for k, v in st["model_state_dict"].items()}
cfg = dict(batch_size=256, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, loops=16, causal=False, boundary_mlp=True, ckpt=False, blocks_per_seg=8, block_inj=True)
with torch.device("cuda"): m = LT(cfg)
m.load_state_dict(sd, strict=True); m.eval(); inner = m.inner; core = inner.core; fc = core.kernel_fast(); AB = core.W_C()
batch = dict(inputs=inp, labels=lab, puzzle_identifiers=torch.zeros(256, dtype=torch.int32, device="cuda"))
h = inner.init_hidden.expand(256, 81, -1).clone(); inj = inner._injection(batch); blank = (inp == 1); res = {}
def read(h):
    lg = core.w_cls(h)[..., 2:11]; p = lg.softmax(-1); ent = -(p*p.clamp_min(1e-9).log()).sum(-1)
    return ent[blank].mean().item(), ((lg.argmax(-1)+2) == lab)[blank].float().mean().item(), lg.argmax(-1)
for loop in range(16):
    for k in range(8):
        e0, a0, g0 = read(h)
        g, u = inner.b_gate_up(h).chunk(2, -1); hb = inner.b_carry*h + inner.b_down(F.silu(g)*u)
        e1, a1, g1 = read(hb); h = hb + inner.inj_gate*inj
        e2, a2, g2 = read(h)
        hh = core.phi(h, 0.5); f, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc); h = core.phi(hh + f, 0.5)
        e3, a3, g3 = read(h)
        if loop in (0, 1, 15) and k in (0, 3, 7):
            res[f"L{loop}_b{k}"] = {"ent_pre": round(e0,3), "ent_postSwiGLU": round(e1,3), "ent_postInj": round(e2,3), "ent_postAttn": round(e3,3),
                                   "acc_pre": round(a0,3), "acc_postSwiGLU": round(a1,3), "acc_postAttn": round(a3,3),
                                   "flip_SwiGLU": round((g1 != g0)[blank].float().mean().item(),3), "flip_Attn": round((g3 != g2)[blank].float().mean().item(),3)}
print(json.dumps(res, indent=1)); json.dump(res, open("sharpen_R1B8.json","w"), indent=1)
