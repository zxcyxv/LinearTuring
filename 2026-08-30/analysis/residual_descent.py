"""잔차 하강: min_h ½‖F(h) − h‖² (F = 세그먼트 사상, float32, w 초기화). 퍼즐 57 의 seg40 상태(자연 탈출까지 75 세그먼트)에서 시작.
  (a) 경사 하강 (역방향 자동미분, 정규화된 스텝)  (b) 뉴턴-GMRES: (J − I) δ = −(F(h) − h), J v 는 forward-mode 자동미분.
  각 반복 뒤 잔차·오답 수·세그먼트 환산 비용. 라벨은 채점에만.
사용: python analysis/residual_descent.py [퍼즐=57] [시작 세그=40]"""
import sys, os, importlib.util, numpy as np, torch, scipy.sparse.linalg as sla
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
PZ = int(sys.argv[1]) if len(sys.argv) > 1 else 57; SEG0 = int(sys.argv[2]) if len(sys.argv) > 2 else 40; K = 8
inp, lab, _ = load_test(512); G = (lab - 2).long(); bl = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=1, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=False)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
for p in m.parameters(): p.requires_grad_(False)
x = inp[PZ:PZ + 1]
with torch.no_grad(): AB = tuple(t.detach() for t in inner.W_C()); kc = tuple(t.detach() for t in inner.kernel()); kcb = tuple(t.detach() for t in inner.kernel(inner.beta)); inj = inner.injection(make_batch(x, x)).float().detach()
def F(h):
    w = None
    for _ in range(K): h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, kcb)
    return h
Wc = inner.w_cls.weight[2:11].float()
H = np.load(os.path.join(ROOT, "results", "json", f"dyn_macro_{PZ}_H.npy")); h0 = torch.tensor(H[SEG0 * K - 1].reshape(1, 81, 832), device="cuda", dtype=torch.float32)
def status(h):
    with torch.no_grad(): r = float((F(h) - h).norm() / h.norm()); nw = int((((h[0] @ Wc.T).argmax(-1) != G[PZ]) & bl[PZ]).sum())
    return r, nw
def settle(h, n=6):
    """하강 뒤 평소 사상으로 n 세그먼트: 정답 고정점에 안착했는가"""
    with torch.no_grad():
        for _ in range(n): h = F(h)
    return status(h)
print(f"시작 (seg{SEG0}): 잔차 {status(h0)[0]:.4f}, 오답 {status(h0)[1]}")
# ---------- (a) 경사 하강
for alpha in (0.02, 0.05, 0.1):
    h = h0.clone(); cost = 0; hist = []; solved = None
    for it in range(60):
        hh = h.detach().requires_grad_(True); res = F(hh) - hh; L = 0.5 * (res ** 2).sum(); g, = torch.autograd.grad(L, hh); cost += 3       # forward+backward ≈ 3 세그먼트
        with torch.no_grad(): h = h - alpha * h.norm() * g / (g.norm() + 1e-9)
        r, nw = status(h); hist.append((it + 1, r, nw))
        if nw == 0 and solved is None: solved = (it + 1, cost)
    s = settle(h); print(f"(a) 경사 α={alpha}: 60회 뒤 잔차 {hist[-1][1]:.4f} 오답 {hist[-1][2]} | 처음 오답 0: {solved} (반복, 세그 환산) | 이후 6 세그 안착: 잔차 {s[0]:.4f} 오답 {s[1]} | 궤적(반복,잔차,오답) " + " ".join(f"{a}:{b:.3f}/{c}" for a, b, c in hist[::10]))
# ---------- (b) 뉴턴-GMRES (감쇠)
D = 81 * 832
for damp in (0.5, 1.0):
    h = h0.clone(); cost = 0; log = []
    for it in range(12):
        with torch.no_grad(): res = (F(h) - h).reshape(-1)
        hc = h.clone()
        def mv(v):
            vt = torch.tensor(np.asarray(v, dtype=np.float32).reshape(1, 81, 832), device="cuda"); _, jv = torch.func.jvp(F, (hc,), (vt,)); return (jv - vt).detach().reshape(-1).cpu().numpy().astype(np.float64)
        op = sla.LinearOperator((D, D), matvec=mv, dtype=np.float64); nmv = [0]
        def cb(rk): nmv[0] += 1
        delta, info = sla.gmres(op, -res.cpu().numpy().astype(np.float64), rtol=1e-2, restart=20, maxiter=2, callback=cb, callback_type="pr_norm"); cost += 1 + nmv[0]
        with torch.no_grad(): h = h + damp * torch.tensor(delta.astype(np.float32).reshape(1, 81, 832), device="cuda")
        r, nw = status(h); log.append((it + 1, r, nw, cost))
        if nw == 0 and r < 1e-3: break
    s = settle(h); print(f"(b) 뉴턴-GMRES 감쇠 {damp}: " + " ".join(f"[{a}: 잔차 {b:.3f} 오답 {c} 비용 {d}]" for a, b, c, d in log) + f" | 안착: 잔차 {s[0]:.4f} 오답 {s[1]}")
