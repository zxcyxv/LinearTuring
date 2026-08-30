"""한 퍼즐을 규칙 없이(stdp1, w 세그먼트 초기화) 오래 돌리며 세그먼트별 오답 수·특정 칸들의 숫자·로짓 추적.
사용: python analysis/one_puzzle_long.py [퍼즐=57] [세그먼트=256] [칸들=2,4,20,21,76,12,26]"""
import sys, os, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); PZ = int(sys.argv[1]) if len(sys.argv) > 1 else 57; S = int(sys.argv[2]) if len(sys.argv) > 2 else 256
CELLS = [int(c) for c in (sys.argv[3] if len(sys.argv) > 3 else "2,4,20,21,76,12,26").split(",")]
inp, lab, _ = load_test(512); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
b0 = (PZ // 128) * 128; x = inp[b0:b0 + 128]; i = PZ - b0; h = inner.init_hidden.expand(128, 81, -1).clone(); rc = lambda c: f"{c//9+1}행{c%9+1}열"
print(f"퍼즐 {PZ}: 칸 " + "  ".join(f"{c}({rc(c)}, 정답 {int(G[PZ,c])+1})" for c in CELLS))
print("세그  오답수 유효?  " + "  ".join(f"칸{c:>2d}" for c in CELLS) + "   | 칸2 로짓 7/9    칸4 로짓 9/7    뒤집힌 칸수")
prevP = None; solved_at = None
for s in range(S):
    w = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
        for _ in range(8):
            h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
        lg = inner.w_cls(h).float()[:, :, 2:11]
    h = h.float(); P = lg.argmax(-1); wrong = ((P[i] != G[PZ]) & bl[PZ]); nw = int(wrong.sum())
    fin = torch.where(bl[PZ], P[i], x[i] - 2); valid = not ((fin[:, None] == fin[None, :]) & pm).any()
    flips = int(((P[i] != prevP[i]) & bl[PZ]).sum()) if prevP is not None else 0; prevP = P.clone()
    if solved_at is None and nw == 0: solved_at = s + 1
    if s + 1 <= 24 or (s + 1) % 8 == 0 or nw == 0 and (solved_at == s + 1):
        vals = "  ".join(f"  {int(P[i,c])+1}{'*' if P[i,c]!=G[PZ,c] else ' '}" for c in CELLS)
        print(f"{s+1:>4d}  {nw:>4d}   {'유효' if valid else ' - '}  {vals}   | {lg[i,2,6]:+6.1f}/{lg[i,2,8]:+6.1f}   {lg[i,4,8]:+6.1f}/{lg[i,4,6]:+6.1f}    {flips}")
print("해결 세그먼트:", solved_at if solved_at else f"{S} 까지 미해결", "| 마지막 오답 수:", nw)
