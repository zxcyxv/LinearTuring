"""어텐션 그래프의 입력 의존도·시간 변화: 스도쿠(R1B8, M1) vs CA(b110k8_woC_t4_long).
   지표: (i) 스텝/블록 간 Δa/‖a‖  (ii) 배치 내 퍼즐 간 편차 ‖a−ā‖/‖a‖ (0 이면 입력 무관 정적 그래프)
        (iii) |a| 의 피어 마스크 상관 (스도쿠)."""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM'); sys.path.insert(0, '/workspace/LinearTuring')
from models.lt.lt import LT
from model1 import Model1
torch.set_grad_enabled(False); res = {}
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")[:128]).to(torch.int32).cuda()
r = torch.arange(81)//9; c = torch.arange(81)%9; bx = (r//3)*3 + c//3
peer = ((r[:,None]==r[None]) | (c[:,None]==c[None]) | (bx[:,None]==bx[None])).float().cuda(); peer.fill_diagonal_(0)
def sudoku(name, ck, over):
    st = torch.load(ck, map_location='cuda', weights_only=False)
    sd = {(k.replace("model.", "", 1) if k.startswith("model.") else k): v for k, v in st["model_state_dict"].items()}
    cfg = dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
               R=8, loops=16, causal=False, boundary_mlp=True, ckpt=False, blocks_per_seg=1, block_inj=False); cfg.update(over)
    with torch.device("cuda"): m = LT(cfg)
    m.load_state_dict(sd, strict=True); m.eval(); inner = m.inner; core = inner.core
    NB, nstep = cfg["blocks_per_seg"], cfg["R"]; dt = 1.0/cfg["R"]; fc = core.kernel_fast(); AB = core.W_C()
    batch = dict(inputs=inp, labels=inp, puzzle_identifiers=torch.zeros(128, dtype=torch.int32, device="cuda"))
    h = inner.init_hidden.expand(128, 81, -1).clone(); inj = inner._injection(batch); out = {}
    for loop in range(16):
        A = []
        for k in range(NB):
            if k == 0 or cfg.get("block_inj"):
                g, u = inner.b_gate_up(h).chunk(2, -1); h = inner.b_carry*h + inner.b_down(F.silu(g)*u) + inner.inj_gate*inj
            for s in range(nstep):
                hh = core.phi(h, dt/2); f, a, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc); A.append(a); h = core.phi(hh+dt*f, dt/2)
        if loop in (0, 15):
            A = torch.stack(A)                       # [S,B,H,T,T]
            da = ((A[1:]-A[:-1]).flatten(1).norm(dim=1)/A[1:].flatten(1).norm(dim=1)).mean().item() if len(A) > 1 else None
            am = A[-1].mean(0, keepdim=True); dep = ((A[-1]-am).norm()/A[-1].norm()).item()
            absa = A[-1].abs().mean(0)               # [H,T,T]
            corr = [float(np.corrcoef(absa[hh_].flatten().cpu(), peer.flatten().cpu())[0,1]) for hh_ in range(absa.shape[0])]
            out[f"L{loop}"] = {"da_step": da, "input_dep": dep, "peer_corr_per_head": [round(x,3) for x in corr]}
    res[name] = out
sudoku("R1B8", "/workspace/LinearTuring/sudoku_runs/2026-08-23/checkpoints/R1B8_step9765.pt", {"R":1,"blocks_per_seg":8,"block_inj":True})
sudoku("M1", "/workspace/LinearTuring/sudoku_runs/2026-08-23/checkpoints/M1_d832H8_step13671.pt", {"R":8,"blocks_per_seg":1})
# --- CA ---
from ca_task import batch as cabatch
st = torch.load("/workspace/LinearTuring/runs/b110k8_woC_t4_long.pt", map_location="cuda", weights_only=False); a_ = st["args"]
pos = torch.stack([torch.arange(a_["T"]).float(), torch.zeros(a_["T"])], 1)
m = Model1(d=a_["d"], H=a_["H"], R=a_["R"], n_classes=2, positions=pos, vocab=2, pool=False, boundary_wo=a_["boundary_wo"], wo_mode=a_["wo_mode"],
           use_ov=not a_["no_ov"], sheaf=a_["sheaf"], lam_mode=("diag" if (a_["no_ov"] or a_["sheaf"]) else "full") if a_["lam"]=="auto" else a_["lam"]).cuda()
m.load_state_dict(st["model"], strict=False); m.eval()
gen = torch.Generator(device="cuda").manual_seed(0); x, y = cabatch(512, a_["T"], a_["k"], a_["rule"], "cuda", gen)
logits, traj = m(x, R=a_["R"], steps=a_["R"]*a_["tau"], record=True, record_fields=("a",))
A = torch.stack(traj["a"]); acc = (logits.argmax(-1)==y).float().mean().item()
da = ((A[1:]-A[:-1]).flatten(1).norm(dim=1)/A[1:].flatten(1).norm(dim=1)).mean().item()
deps = [((A[s]-A[s].mean(0,keepdim=True)).norm()/A[s].norm()).item() for s in range(len(A))]
res["CA_b110k8_woC_t4_long"] = {"cell": acc, "da_step_mean": da, "input_dep_first_last": [deps[0], deps[-1]], "input_dep_mean": float(np.mean(deps))}
print(json.dumps(res, ensure_ascii=False, indent=1)); json.dump(res, open("graph_dep.json","w"), indent=1)
