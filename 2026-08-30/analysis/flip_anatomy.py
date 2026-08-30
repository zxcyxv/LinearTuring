"""퍼즐 57 의 뒤집힘을 정확한 항등식으로 귀속. 근사·지표 없음.
  블록:  h_mid = h_in + Δ경계 + Δ주입 ;  h_out = c·(h_mid + f),  c = 1/√(1+γ‖h_mid+f‖²) > 0
  읽기가 선형이므로 두 숫자 사이 여유 m = W_cls[a]·h − W_cls[b]·h 에 대해
        m_out = c·( m_in + Δm_경계 + Δm_주입 + Σ_n Δm_n ),   Δm_n = 보낸 칸 n 의 기여
  c>0 이므로 부호를 바꾸는 것은 세 항뿐. 각 블록에서 어느 항이 얼마를 옮겼는지 그대로 출력한다.
사용: python analysis/flip_anatomy.py [퍼즐=57] [시작세그=100] [끝세그=120]"""
import sys, os, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False)
PZ = int(sys.argv[1]) if len(sys.argv) > 1 else 57; S0 = int(sys.argv[2]) if len(sys.argv) > 2 else 100; S1 = int(sys.argv[3]) if len(sys.argv) > 3 else 120; K = 8
inp, lab, _ = load_test(512); G = (lab - 2).long(); bl = inp == 1; pm = peer_mask()
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
               stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; Wc = inner.w_cls.weight[2:11].float(); x = inp[:128]; i = PZ
AB = inner.W_C(); kc = inner.kernel(); inj_v = inner.injection(make_batch(x, x)).float()
CELLS = [21, 2, 4, 20]; WRONG = {21: 2, 2: 8, 4: 6, 20: 6}; RIGHT = {c: int(G[PZ, c]) for c in CELLS}   # 16 시점의 틀린 숫자(0-based), 정답
rc = lambda c: f"{c//9+1}행{c%9+1}열"
h = inner.init_hidden.expand(128, 81, -1).clone().float()
rows = {c: [] for c in CELLS}
for s in range(S1):
    w = None
    for blk in range(K):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hb = inner.boundary(h); hmid = hb + inner.inj_gate * inj_v
            a = inner.attn(hmid, AB, kc); vv = torch.einsum('btd,hcd->bthc', hmid, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
            Gm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn); w = Gm if w is None else w + torch.sigmoid(inner.eta_raw) * (Gm - w)
            o = torch.einsum('bhtn,bnhc->bthc', w.to(vv.dtype), vv); f = torch.einsum('bthc,hcd->btd', o, inner.w_sh)
            hout = inner.phi(hmid + f)
        hb = hb.float(); hmid = hmid.float(); f = f.float(); hout = hout.float()
        if S0 <= s < S1:
            for c in CELLS:
                d_w, d_r = WRONG[c], RIGHT[c]; u = (Wc[d_w] - Wc[d_r])                      # 여유 방향 (틀린 − 정답)
                m_in = float(h[i, c] @ u); m_b = float((hb[i, c] - h[i, c]) @ u); m_j = float((hmid[i, c] - hb[i, c]) @ u)
                # 보낸 칸별 기여: f_c = Σ_n Wᵀ(w_cn W v_n)
                on = torch.einsum('hn,nhc->nhc', w[i, :, c, :].float(), vv[i].float()); fn = torch.einsum('nhc,hcd->nd', on, inner.w_sh.float())
                dm = fn @ u                                                                   # [81]
                m_f = float(dm.sum()); m_out = float(hout[i, c] @ u); cc = m_out / (m_in + m_b + m_j + m_f + 1e-30)
                pr = torch.tensor(pm[c], device="cuda"); wrongset = torch.zeros(81, dtype=torch.bool, device="cuda"); wrongset[CELLS] = True
                rows[c].append((s + 1, blk, m_in, m_b, m_j, m_f, float(dm[c]), float(dm[wrongset & (torch.arange(81, device="cuda") != c)].sum()),
                                float(dm[pr & ~wrongset].sum()), float(dm[~pr & ~wrongset & (torch.arange(81, device="cuda") != c)].sum()), m_out, cc))
        h = hout
    if (s + 1) % 4 == 0 or S0 <= s < S1:
        P = (h[i] @ Wc.T).argmax(-1); nw = int(((P != G[PZ]) & bl[PZ]).sum())
        if S0 <= s < S1: print(f"seg {s+1:>3d}: 오답 {nw:>2d}  " + " ".join(f"칸{c}={int(P[c])+1}" for c in CELLS), flush=True)
print()
for c in CELLS:
    print(f"===== 칸 {c} ({rc(c)}) 여유 m = 로짓[{WRONG[c]+1}] − 로짓[{RIGHT[c]+1}]  (양수 = 틀린 쪽 우세). 블록별 정확 분해")
    print("  seg.blk |    m_in |   Δ경계 |   Δ주입 |  Δ전달 = 자기 + 오답고리 + 피어(맞음) + 비피어 |    m_out |   Φ배율")
    for r in rows[c][::4]:
        print(f"  {r[0]:>3d}.{r[1]}  | {r[2]:>7.2f} | {r[3]:>7.2f} | {r[4]:>7.2f} | {r[5]:>7.2f} = {r[6]:>6.2f} + {r[7]:>6.2f} + {r[8]:>6.2f} + {r[9]:>6.2f} | {r[10]:>7.2f} | {r[11]:>6.3f}")
    print()
