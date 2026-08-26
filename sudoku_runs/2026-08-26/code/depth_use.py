"""깊이 사용도: 세그별 정확도 곡선 + 세그당 블록 절단(keep k) — 체크포인트 비교."""
import sys, json, numpy as np, torch
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
torch.set_grad_enabled(False)
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")[:512]).to(torch.int32).cuda(); lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")[:512]).to(torch.int32).cuda()
blank = inp == 1
def run_ckpt(name, ck, bil):
    st = torch.load(ck, map_location='cuda', weights_only=False); sd = {(k.replace("_orig_mod.","").replace("model.","",1) if k.startswith(("model.","_orig_mod.")) else k): v for k, v in st["model_state_dict"].items()}
    cfg = dict(batch_size=512, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, R=1, loops=16, causal=False, boundary_mlp=True, ckpt=False, blocks_per_seg=8, block_inj=True, bilinear=bil)
    with torch.device("cuda"): m = LT(cfg)
    m.load_state_dict(sd, strict=True); m.eval(); inner=m.inner; core=inner.core; fc=core.kernel_fast(); AB=core.W_C()
    batch = dict(inputs=inp, labels=lab, puzzle_identifiers=torch.zeros(512, dtype=torch.int32, device="cuda")); inj = inner._injection(batch)
    def run(keep=8, loops=16):
        h = inner.init_hidden.expand(512, 81, -1).clone(); segc=[]; sege=[]
        for loop in range(loops):
            for k in range(keep):
                h = inner._boundary(h) + inner.inj_gate*inj
                hh = core.phi(h, 0.5); f, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc); h = core.phi(hh + f, 0.5)
            p = core.w_cls(h).argmax(-1); ok = (p == lab) | ~blank
            segc.append(((p==lab)&blank).sum().item()/blank.sum().item()); sege.append(ok.all(-1).sum().item())
        return segc, sege
    segc, sege = run()
    print(f"== {name}\n  세그별 빈칸 정확도: " + " ".join(f"{x:.3f}" for x in segc)); print("  세그별 완답/512:   " + " ".join(str(x) for x in sege))
    print("  블록 절단 keep→(빈칸acc, 완답): " + "  ".join(f"k{k}:({run(keep=k)[0][-1]:.3f},{run(keep=k)[1][-1]})" for k in (1,2,4,8)))
    print("  loops 연장 32 → 마지막:", (lambda r: (round(r[0][-1],3), r[1][-1]))(run(loops=32)))
run_ckpt("3일차 R1B8 @9765 (SwiGLU)", "/workspace/LinearTuring/sudoku_runs/2026-08-23/checkpoints/R1B8_step9765.pt", False)
run_ckpt("bilin_ok @46872", "/workspace/LinearTuring/refs/URM/checkpoints/R1B8_bilin_ok/step_46872.pt", True)
