"""칸 21 이 왜 하필 seg 112 에 넘어갔나. 정확한 항등식만 사용.
  m = W_cls[3]·h₂₁ − W_cls[7]·h₂₁ ;  블록마다 m_out = c·(m_in + Δ경계 + Δ주입 + Σ_n Δm_n)
  Δm_n = Σ_h w^h_{21,n} · (u · Wᵀ_h W_h v^h_n)  → 보낸 칸별로, 그리고 (결합 w) × (값 사영 s) 로 한 번 더 분해.
사용: python analysis/why_112.py [세그=120]"""
import sys, os, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); PZ = 57; S = int(sys.argv[1]) if len(sys.argv) > 1 else 120; K = 8; T = 21
inp, lab, _ = load_test(512); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
               stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; Wc = inner.w_cls.weight[2:11].float(); x = inp[:128]; i = PZ
AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x)).float()
u = Wc[2] - Wc[6]                                      # 여유 방향: 숫자3 − 숫자7
CYC = [20, 4, 2]; COR3 = [12, 26]                      # 오답 고리 / 칸21 의 3 을 막는 맞는 칸
peer = pm[T].clone(); others = peer.clone(); others[CYC] = False; others[COR3] = False
nonpeer = ~pm[T].clone(); nonpeer[T] = False
h = inner.init_hidden.expand(128, 81, -1).clone().float()
REC = []
for s in range(S):
    w = None
    for blk in range(K):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hb = inner.boundary(h); hmid = hb + inner.inj_gate * inj
            a = inner.attn(hmid, AB, kc); vv = torch.einsum('btd,hcd->bthc', hmid, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
            Gm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn); w = Gm if w is None else w + torch.sigmoid(inner.eta_raw) * (Gm - w)
            o = torch.einsum('bhtn,bnhc->bthc', w.to(vv.dtype), vv); f = torch.einsum('bthc,hcd->btd', o, inner.w_sh)
            hout = inner.phi(hmid + f)
        hb, hmid, hout = hb.float(), hmid.float(), hout.float(); wf = w[i].float(); vf = vv[i].float()
        m_in = float(h[i, T] @ u); m_b = float((hb[i, T] - h[i, T]) @ u); m_j = float((hmid[i, T] - hb[i, T]) @ u)
        on = torch.einsum('hn,nhc->nhc', wf[:, T, :], vf); fn = torch.einsum('nhc,hcd->nd', on, inner.w_sh.float()); dm = fn @ u   # [81]
        # 결합 × 값 사영 분해: s_n = u·WᵀW v_n (헤드 합 전 각 헤드), w_n = w_{T,n}
        sn_h = torch.einsum('nhc,hcd,d->nh', vf, inner.w_sh.float(), u)      # [81,H]
        wn_h = wf[:, T, :].T                                                  # [81,H]
        P = (hout[i] @ Wc.T).argmax(-1)
        REC.append(dict(seg=s + 1, blk=blk, m_in=m_in, m_b=m_b, m_j=m_j, m_out=float(hout[i, T] @ u),
                        self=float(dm[T]), cyc=float(dm[CYC].sum()), cor3=float(dm[COR3].sum()), oth=float(dm[others].sum()), npr=float(dm[nonpeer].sum()),
                        w20=float(wn_h[20].sum()), s20=float(sn_h[20].sum()), w12=float(wn_h[12].sum()), s12=float(sn_h[12].sum()),
                        w26=float(wn_h[26].sum()), s26=float(sn_h[26].sum()), d12=int(P[12]), d26=int(P[26]), d76=int(P[76]), d20=int(P[20]), d21=int(P[T]),
                        hn=float(hout[i, T].norm())))
        h = hout
import json; json.dump(REC, open(os.path.join(ROOT, "results", "json", "why_112.json"), "w"))
R = {k: np.array([r[k] for r in REC]) for k in REC[0]}
seg = R["seg"]; b0 = R["blk"] == 0
print("세그먼트별(블록0) 칸21 여유와 항 분해.  cor3 = 맞는 3(칸12,26)의 밀림, cyc = 오답고리(20,4,2)의 밀림")
print(" seg |  m_in | Δ경계 |   자기 |   cyc |  cor3 |   기타피어 | 비피어 |  m_out | 21의값 | 12/26/76")
for s0 in list(range(10, 100, 10)) + list(range(100, 121, 2)):
    k = np.where(b0 & (seg == s0))[0]
    if len(k) == 0: continue
    k = k[0]
    print(f" {s0:>3d} | {R['m_in'][k]:>5.2f} | {R['m_b'][k]:>5.2f} | {R['self'][k]:>6.2f} | {R['cyc'][k]:>5.2f} | {R['cor3'][k]:>5.2f} | {R['oth'][k]:>10.2f} | {R['npr'][k]:>6.2f} | {R['m_out'][k]:>6.2f} | {R['d21'][k]+1:>5d} | {R['d12'][k]+1}/{R['d26'][k]+1}/{R['d76'][k]+1}")
print("\ncor3 항의 (결합 w) × (값 사영 s) 분해 — 변한 것이 결합인가 값인가")
print(" seg |   w12 |   s12 |  곱=Δm12 |   w26 |   s26 |  곱=Δm26 |   w20 |   s20 |  곱=Δm20")
for s0 in list(range(10, 100, 10)) + list(range(100, 121, 2)):
    k = np.where(b0 & (seg == s0))[0]
    if len(k) == 0: continue
    k = k[0]
    print(f" {s0:>3d} | {R['w12'][k]:>5.2f} | {R['s12'][k]:>5.2f} | {R['w12'][k]*R['s12'][k]:>8.2f} | {R['w26'][k]:>5.2f} | {R['s26'][k]:>5.2f} | {R['w26'][k]*R['s26'][k]:>8.2f} | {R['w20'][k]:>5.2f} | {R['s20'][k]:>5.2f} | {R['w20'][k]*R['s20'][k]:>8.2f}")
# 여유가 음수였던 모든 블록: 언제, 얼마나 오래
neg = R["m_out"] < 0
runs = []; st = None
for j in range(len(neg)):
    if neg[j] and st is None: st = j
    if (not neg[j]) and st is not None: runs.append((seg[st], R['blk'][st], j - st)); st = None
if st is not None: runs.append((seg[st], R['blk'][st], len(neg) - st))
print(f"\n칸21 여유가 음수였던 구간 (시작 seg.blk, 지속 블록수): " + ", ".join(f"{a}.{b}×{c}" for a, b, c in runs))
