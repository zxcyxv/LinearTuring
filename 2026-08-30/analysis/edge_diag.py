# 간선량 진단: 16 세그먼트 끝에서 (피어 × 같은 argmax) 등 4 부류별 헤드별 a, 값 코사인(centered/raw), 헤드 p·p
import sys, os, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis')); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", "core/minimal.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); inp, lab, _ = load_test(512); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load("checkpoints/R1B8_min_stdp1_step228501.pt", map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eta = torch.sigmoid(inner.eta_raw).float(); eps = inner.config.eps
acc = {}
def add(k, v): acc.setdefault(k, []).append(v)
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); w = None
    for s in range(16):
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for _ in range(8):
                h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
            a = inner.attn(h, AB, kc).float()
            vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh).float()
            lg = inner.w_cls(h).float()
        h = h.float(); w = w.float()
    vn = vv / (vv.norm(dim=-1, keepdim=True) + eps); cos_raw = torch.einsum('bthc,bnhc->bhtn', vn, vn)
    vc = vv - vv.mean(1, keepdim=True); vcn = vc / (vc.norm(dim=-1, keepdim=True) + eps); cos_c = torch.einsum('bthc,bnhc->bhtn', vcn, vcn)
    p = torch.softmax(lg[:, :, 2:11], -1); pp = torch.einsum('btd,bnd->btn', p, p)
    P = lg[:, :, 2:11].argmax(-1); same = P[:, :, None] == P[:, None, :]; peer = pm[None].expand(n, 81, 81)
    wrong = (P != G[b:b + n]) & bl[b:b + n]; anyw = wrong[:, :, None] | wrong[:, None, :]
    offd = ~torch.eye(81, dtype=torch.bool, device="cuda")[None]
    classes = {"peer&same(충돌)": peer & same & offd, "peer&same&오답포함": peer & same & offd & anyw, "peer&diff": peer & ~same & offd, "nonpeer&same": ~peer & same & offd, "nonpeer&diff": ~peer & ~same & offd}
    for name, msk in classes.items():
        cnt = float(msk.sum())
        add(name, dict(cnt=cnt / n, a=[float((a[:, hh] * msk).sum() / cnt) for hh in range(8)], cos_raw=float((cos_raw.mean(1) * msk).sum() / cnt), cos_c=float((cos_c.mean(1) * msk).sum() / cnt), pp=float((pp * msk).sum() / cnt),
                       G=[float(((a[:, hh] * cos_c[:, hh]) * msk).sum() / cnt) for hh in range(8)], amag=[float((a[:, hh].abs() * msk).sum() / cnt) for hh in range(8)]))
for name, L in acc.items():
    k = len(L); agg = lambda key: np.mean([d[key] for d in L], 0)
    print(f"\n{name}: 쌍/퍼즐 {agg('cnt'):.1f} | cos_raw {agg('cos_raw'):.3f} cos_centered {agg('cos_c'):.3f} p·p {agg('pp'):.3f}")
    print("  a 헤드별   ", np.round(agg('a'), 3)); print("  |a| 헤드별 ", np.round(agg('amag'), 3)); print("  a·cos_c   ", np.round(agg('G'), 3))
