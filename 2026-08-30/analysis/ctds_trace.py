# 맞바꿈(오답 2칸) 퍼즐 추적: 워밍업 16 뒤 ρ 로 g 성장. 세그먼트별 오답 칸·충돌 상대 값, g, 오답 수
import sys, os, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis')); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", "core/minimal.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); RHO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05; CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
inp, lab, _ = load_test(512); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda"); OFFD = (~torch.eye(81, dtype=torch.bool, device="cuda")).float()
ck = torch.load("checkpoints/R1B8_min_stdp1_step228501.pt", map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eta = torch.sigmoid(inner.eta_raw).float(); eps = inner.config.eps; HEADS = [0, 2, 4]; FLAG = 0.216; BETA = 1 / 16; DECAY = 0.02; WARM = 16; S = 48
b = 0; x = inp[b:b + 128]; n = 128; h = inner.init_hidden.expand(n, 81, -1).clone(); g = torch.zeros(n, 81, 81, device="cuda"); vbar = torch.zeros_like(g)
hist = []  # (P, g) per segment
for s in range(S):
    w = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
        for _ in range(8):
            pr = torch.softmax(inner.w_cls(h).float()[:, :, 2:11], -1); pp = torch.einsum('btd,bnd->btn', pr, pr)
            af = inner.attn(h, AB, kc).float(); R = (-af[:, HEADS].sum(1)).clamp_min(0); viol = (pp * R - FLAG).clamp_min(0) * OFFD
            if s >= WARM:
                vbar = vbar + BETA * (viol - vbar); g = torch.minimum(torch.where(vbar > 0.01, g * (1 + RHO * vbar) + RHO * vbar, g * (1 - DECAY)), torch.full_like(g, CAP))
            h = inner.boundary(h); h = h + inner.inj_gate * inj; a = inner.attn(h, AB, kc)
            vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps); agree = torch.einsum('bthc,bnhc->bhtn', vn, vn); Gm = a * agree
            w = Gm if w is None else w + eta * (Gm - w); wf = w.float(); a_eff = wf - (g * pp).unsqueeze(1) * (-wf).clamp_min(0)
            o = torch.einsum('bhtn,bnhc->bthc', a_eff.to(vv.dtype), vv); f = torch.einsum('bthc,hcd->btd', o, inner.w_sh); h = inner.phi(h + f)
        lg = inner.w_cls(h).float()
    h = h.float(); hist.append((lg[:, :, 2:11].argmax(-1).clone(), g.clone(), torch.softmax(lg[:, :, 2:11], -1).max(-1).values.clone()))
P16 = hist[WARM - 1][0]; wrong16 = ((P16 != G[:n]) & bl[:n]); nw = wrong16.sum(1)
cand = [i for i in range(n) if nw[i] in (2, 3, 4)][:5]
print(f"ρ={RHO} cap={CAP}  워밍업 16 시점 오답 칸 수 분포(128 퍼즐): 0:{int((nw==0).sum())} 1-2:{int(((nw>=1)&(nw<=2)).sum())} 3-4:{int(((nw>=3)&(nw<=4)).sum())} 5+:{int((nw>=5).sum())}")
for i in cand:
    W = torch.where(wrong16[i])[0].tolist()
    print(f"\n퍼즐 {i}: 16 시점 오답 {len(W)}칸  " + "  ".join(f"칸{c}(정답 {int(G[i,c])+1}, 예측 {int(P16[i,c])+1})" for c in W))
    # 충돌 상대: 같은 예측값을 가진 피어
    partners = {c: [int(q) for q in torch.where(pm[c] & (P16[i] == P16[i, c]))[0]] for c in W}
    print("  충돌 상대(피어, 같은 값): " + "  ".join(f"칸{c}→{partners[c]}" for c in W))
    cells = sorted(set(W + sum(partners.values(), [])))
    print("  세그   오답수  " + " ".join(f"칸{c:>2d}(정{int(G[i,c])+1})" for c in cells) + "   |  g(오답-상대)")
    for s in range(WARM - 1, S, 2):
        P, gg, conf = hist[s]; nwr = int(((P[i] != G[i]) & bl[i]).sum())
        vals = " ".join(f"   {int(P[i,c])+1}{'*' if P[i,c]!=G[i,c] else ' '}({conf[i,c]:.2f})" for c in cells)
        gs = " ".join(f"{float(gg[i,c,q]):.1f}" for c in W for q in partners[c][:2])
        print(f"  {s+1:>4d}   {nwr:>4d}   {vals}   | {gs}")
