# 후보 위반 신호(모델 내부)의 정밀도/재현율 — 2026-08-30/2026-08-30/analysis/ctds.py 의 탐지기 선택 근거. 사용: python 2026-08-30/analysis/edge_signal.py (저장소 루트에서)
# 후보 위반 신호들의 정밀도/재현율 (512 퍼즐, 16 세그먼트 끝, reset 모드)
import sys, os, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis')); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", "core/minimal.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); inp, lab, _ = load_test(512); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load("checkpoints/R1B8_min_stdp1_step228501.pt", map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; offd = ~torch.eye(81, dtype=torch.bool, device="cuda")[None]
sig = {}; conf_all = []
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone()
    for s in range(16):
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for _ in range(8):
                h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
            a = inner.attn(h, AB, kc).float(); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh).float(); lg = inner.w_cls(h).float()
        h = h.float()
    vc = vv - vv.mean(1, keepdim=True); vcn = vc / (vc.norm(dim=-1, keepdim=True) + eps); cosc = torch.einsum('bthc,bnhc->bhtn', vcn, vcn)
    p = torch.softmax(lg[:, :, 2:11], -1); pp = torch.einsum('btd,bnd->btn', p, p)
    P = lg[:, :, 2:11].argmax(-1); same = P[:, :, None] == P[:, None, :]
    fin = torch.where(bl[b:b+n], P, x - 2); conf = (fin[:, :, None] == fin[:, None, :]) & pm[None] & offd
    asum = a.sum(1); a24 = a[:, 2] + a[:, 4]; a024 = a[:, 0] + a[:, 2] + a[:, 4]
    cands = {"S1 Γ_sum=-Σ a·cos_c": -(a * cosc).sum(1), "S2 pp·relu(-Σa)": pp * (-asum).clamp_min(0), "S3 pp·relu(-(a2+a4))": pp * (-a24).clamp_min(0),
             "S4 pp·relu(-(a0+a2+a4))": pp * (-a024).clamp_min(0), "S5 same·relu(-Σa)": same.float() * (-asum).clamp_min(0), "S6 pp·relu(-a2)·relu(-a4)^.5": pp * ((-a[:, 2]).clamp_min(0) * (-a[:, 4]).clamp_min(0)).sqrt(),
             "S7 -Σ a·pp (헤드합)": -(a * pp.unsqueeze(1)).sum(1)}
    for k, v in cands.items(): sig.setdefault(k, []).append(v[offd.expand(n, 81, 81)].reshape(n, -1).cpu())
    conf_all.append(conf[offd.expand(n, 81, 81)].reshape(n, -1).cpu())
C = torch.cat(conf_all); print(f"실제 충돌 쌍/퍼즐 {C.sum().item()/N/2:.2f} (양방향 포함 {C.sum().item()/N:.1f})")
for k, L in sig.items():
    V = torch.cat(L); pos = V[C]; neg = V[~C]
    print(f"\n{k}: 충돌 중앙값 {pos.median():.3f} 10% {pos.quantile(.1):.3f} | 비충돌 99% {neg.quantile(.99):.3f} 99.9% {neg.quantile(.999):.3f}")
    for thr in [float(pos.quantile(q)) for q in (.5, .25, .1)]:
        fl = V > thr; tp = (fl & C).sum().item(); print(f"   thr {thr:.3f}: 표시/퍼즐 {fl.sum().item()/N/2:6.1f}  정밀도 {tp/max(fl.sum().item(),1):.3f}  재현율 {tp/C.sum().item():.3f}")
