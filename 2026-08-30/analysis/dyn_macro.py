"""퍼즐 하나의 거시 동역학 (stdp1, w 세그먼트 초기화, 규칙 없음): 블록마다 겹침·충돌·에너지/컬·여유를 기록하고 DMD 를 돌린다.
사용: python analysis/dyn_macro.py [퍼즐=57] [세그먼트=120] [출력 npz]"""
import sys, os, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); PZ = int(sys.argv[1]) if len(sys.argv) > 1 else 57; S = int(sys.argv[2]) if len(sys.argv) > 2 else 120; OUT = sys.argv[3] if len(sys.argv) > 3 else f"results/json/dyn_macro_{PZ}.npz"
inp, lab, _ = load_test(512); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; K = 8; b0 = (PZ // 128) * 128; x = inp[b0:b0 + 128]; i = PZ - b0; h = inner.init_hidden.expand(128, 81, -1).clone()
Wc = inner.w_cls.weight[2:11].float(); T = S * K
P_hist = np.zeros((T, 81), int); gap = np.zeros((T, 81)); corr_margin = np.zeros((T, 81)); E_sym = np.zeros(T); f_sym = np.zeros(T); f_anti = np.zeros(T); nconf = np.zeros(T, int)
H = np.zeros((T, 81 * 832), np.float32); W_snap = {}; cells = [2, 4, 20, 21, 76, 12, 26]
k = 0
for s in range(S):
    w = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
        for _ in range(K):
            h = inner.boundary(h); h = h + inner.inj_gate * inj
            a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
            Gm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn); wm = Gm if w is None else w + torch.sigmoid(inner.eta_raw) * (Gm - w)
            # 에너지/컬 (퍼즐 i 만, float32): 대칭·반대칭 결합으로 나눈 전달장
            wf = wm[i].float(); vf = vv[i].float()                                          # [H,T,T], [T,H,C]
            ws = 0.5 * (wf + wf.transpose(-1, -2)); wa = 0.5 * (wf - wf.transpose(-1, -2))
            E_sym[k] = float(-0.5 * torch.einsum('htn,thc,nhc->', ws, vf, vf))
            fs = torch.einsum('htn,nhc->thc', ws, vf); fa = torch.einsum('htn,nhc->thc', wa, vf); f_sym[k] = float(fs.norm()); f_anti[k] = float(fa.norm())
            h, w = inner.step(h, AB, kc, w, None, None)
            lg = inner.w_cls(h).float()[i, :, 2:11]; P = lg.argmax(-1); t2 = lg.topk(2, -1).values
            P_hist[k] = P.cpu().numpy(); gap[k] = (t2[:, 0] - t2[:, 1]).cpu().numpy(); corr_margin[k] = (lg.max(-1).values - lg.gather(-1, G[PZ][:, None]).squeeze(-1)).cpu().numpy()
            fin = torch.where(bl[PZ], P, x[i] - 2); nconf[k] = int(((fin[:, None] == fin[None, :]) & pm).sum() // 2)
            H[k] = h[i].float().reshape(-1).cpu().numpy()
            if (s + 1) in (16, 100, 112, 113, 114, 115, 120) and (k % K == K - 1): W_snap[s + 1] = wm[i].float().cpu().numpy()
            k += 1
    h = h.float()
P16 = P_hist[16 * K - 1]; blank = bl[PZ].cpu().numpy(); Gp = G[PZ].cpu().numpy()
q_sol = ((P_hist == Gp[None]) & blank[None]).sum(1) / blank.sum(); q_w16 = ((P_hist == P16[None]) & blank[None]).sum(1) / blank.sum()
np.savez_compressed(OUT, P=P_hist, gap=gap, corr_margin=corr_margin, E_sym=E_sym, f_sym=f_sym, f_anti=f_anti, nconf=nconf, q_sol=q_sol, q_w16=q_w16, cells=np.array(cells), G=Gp, blank=blank, K=K,
                    **{f"W{s}": v for s, v in W_snap.items()})
np.save(OUT.replace(".npz", "_H.npy"), H)
print("saved", OUT, "| 해결 세그먼트:", (np.where(q_sol == 1)[0][0] // K + 1) if (q_sol == 1).any() else "미해결")
