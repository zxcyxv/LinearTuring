"""블록 동역학 진단: 블록별 상태 점프·어텐션 재배선량·SwiGLU 기여·게이트 통계 + 추론시 경계 놕아웃(항등 치환).
사용: python dyn.py <ckpt> <cfg_json> <out_json>"""
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
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")[:512]).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")[:512]).to(torch.int32)
NB, nstep = cfg["blocks_per_seg"], cfg["R"]; dt = 1.0 / cfg["R"]
stats = {k: np.zeros(NB) for k in ("dh_flow", "dh_bnd", "da", "swi_ratio", "gate_neg", "gate_sat")}
def run(batch, knock=False, rec=None):
    """inner.forward 를 수동 전개 (블록 순서: 경계→주입→흐름). knock=True 면 SwiGLU 기여를 0 (경계=항등)."""
    h = inner.init_hidden.expand(BS, 81, -1).clone()
    inj = inner._injection(batch); fc = core.kernel_fast(); AB = core.W_C()
    last = None
    for loop in range(16):
        for k in range(NB):
            if k == 0 or cfg.get("block_inj"):
                gate, up = inner.b_gate_up(h).chunk(2, -1); sw = inner.b_down(F.silu(gate) * up)
                h_b = inner.b_carry * h + (0 if knock else sw)
                if rec is not None and loop == 15:
                    stats["dh_bnd"][k] += (h_b - h).norm() / h.norm(); stats["swi_ratio"][k] += sw.norm() / h.norm()
                    stats["gate_neg"][k] += (gate < 0).float().mean(); stats["gate_sat"][k] += (gate.abs() > 3).float().mean()
                h = h_b + inner.inj_gate * inj
            h0 = h; a_prev = None
            for _ in range(nstep):
                hh = core.phi(h, dt / 2)
                f, a, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc)
                if rec is not None and loop == 15 and a_prev is not None: stats["da"][k] += (a - a_prev).norm() / a.norm() / max(nstep - 1, 1)
                a_prev = a
                h = core.phi(hh + dt * f, dt / 2)
            if rec is not None and loop == 15: stats["dh_flow"][k] += (h - h0).norm() / h0.norm()
            last = (a, a_prev)
        if rec is not None and loop == 15 and NB == 1: pass
    return core.w_cls(h).argmax(-1).to(torch.int32).cpu()
res = {}
for knock in (False, True):
    ok = []
    for i in range(0, 512, BS):
        b = slice(i, i + BS)
        batch = {k: v.cuda() for k, v in dict(inputs=inp[b], labels=lab[b], puzzle_identifiers=torch.zeros(BS, dtype=torch.int32)).items()}
        p = run(batch, knock=knock, rec=(None if knock else True)); ok.append((p == lab[b]).numpy())
    ok = np.concatenate(ok); res["knock" if knock else "full"] = {"cell": float(ok.mean()), "exact": int(ok.all(1).sum()), "N": 512}
nb = 512 // BS
for k in stats: res[k] = (stats[k] / nb).round(4).tolist()
# 블록간 어텐션 그래프 차이 (dt=1 판: 스텝=블록이므로 블록 k 와 k+1 의 a 비교) — 마지막 loop, 마지막 배치
print(json.dumps(res, ensure_ascii=False)); json.dump(res, open(out, "w"), indent=1)
