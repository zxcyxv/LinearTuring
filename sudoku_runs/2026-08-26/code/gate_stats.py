"""경계 게이트 사전활성 g 의 분포 (loop 0,3,15 · 블록 0,7): std, |g|>1,>2,>3 비율, ‖h‖ 원소 std. 활성화 없음."""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
torch.set_grad_enabled(False)
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")[:128]).to(torch.int32).cuda(); lab = inp
for name, ck, bil in [("3일차 R1B8 @9765", "/workspace/LinearTuring/sudoku_runs/2026-08-23/checkpoints/R1B8_step9765.pt", False),
                      ("swiglu_ok @3906", "/workspace/LinearTuring/refs/URM/checkpoints/R1B8_swiglu_ok/step_3906.pt", False),
                      ("bilin_ok @3906", "/workspace/LinearTuring/refs/URM/checkpoints/R1B8_bilin_ok/step_3906.pt", True)]:
    st = torch.load(ck, map_location='cuda', weights_only=False); sd = {(k.replace("_orig_mod.","").replace("model.","",1) if k.startswith(("model.","_orig_mod.")) else k): v for k, v in st["model_state_dict"].items()}
    cfg = dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, loops=16, causal=False, boundary_mlp=True, ckpt=False, blocks_per_seg=8, block_inj=True, bilinear=bil)
    with torch.device("cuda"): m = LT(cfg)
    m.load_state_dict(sd, strict=True); m.eval(); inner = m.inner; core = inner.core; fc = core.kernel_fast(); AB = core.W_C()
    batch = dict(inputs=inp, labels=lab, puzzle_identifiers=torch.zeros(128, dtype=torch.int32, device="cuda"))
    h = inner.init_hidden.expand(128, 81, -1).clone(); inj = inner._injection(batch); rows = []
    for loop in range(16):
        for k in range(8):
            g, u = inner.b_gate_up(h).chunk(2, -1)
            if loop in (0, 3, 15) and k in (0, 7):
                rows.append(f"  L{loop} b{k}: h원소 std {h.std().item():.3f} ‖h_t‖ {h.norm(dim=-1).mean().item():.2f} | g std {g.std().item():.3f} |g|>1 {(g.abs()>1).float().mean().item():.3f} >2 {(g.abs()>2).float().mean().item():.3f} >3 {(g.abs()>3).float().mean().item():.4f} | u std {u.std().item():.3f}")
            h = inner.b_carry*h + inner.b_down(inner._act(g)*u) + inner.inj_gate*inj
            hh = core.phi(h, 0.5); f, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc); h = core.phi(hh + f, 0.5)
    print(name, " W_g 행노름 평균", inner.b_gate_up.weight[:2304].norm(dim=1).mean().item()); print("\n".join(rows))
