"""2번: 값 블록 경사 동역학 (E = −½Σ a⟨v,v⟩ − Σ logcosh(W_g h) + (γ/4)Σ‖h‖⁴) 이 고정점에 수렴하는가 — 스텝 크기별.
3번: EqProp 두-위상 갱신 (1/β)[∂E/∂θ(h^β) − ∂E/∂θ(h⁰)] 이 autograd 로 펼친 dC(h*(θ))/dθ 와 일치하는가 — β별 코사인."""
import os, importlib.util, math, numpy as np, torch
ROOT = "/workspace/LinearTuring"; PID = 155; SEG = 8; d = 832; H = 8; dh = d // H
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1, amp=False)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
with torch.no_grad():
    m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner; L = I.layers[1]
    z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
    x = torch.from_numpy(X0[None].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y0[None].astype(np.int32) + 1).cuda().long()
    batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(1, dtype=torch.int32, device="cuda"))
    cap = {}; st = {"b": 0}; orig = I.step
    def hooked(Lx, h, AB, kc, *a_, **k_):
        b = st["b"]; st["b"] += 1
        if b == (SEG - 1) * 16 + 1: cap.update(h=h.clone(), AB=AB)
        return orig(Lx, h, AB, kc, *a_, **k_)
    I.step = hooked
    with torch.device("cuda"): carry = m.initial_carry(batch)
    for si in range(SEG): carry, _ = m(carry, batch)
    h_real = cap["h"] + I.embed_scale * I.injection(batch)
    a = I.attn_xy(I.addr(h_real, cap["AB"]), I.kernel(L, torch.zeros_like(L.psi))).detach()[0]     # [H,T,T], ψ=0 → 대칭
    gamma = float(I.gamma); Wcls = I.w_cls.weight.detach().clone(); bcls = I.w_cls.bias.detach().clone()
g = torch.Generator(device="cuda").manual_seed(3)
def params(kind):
    if kind == "ckpt": Wsh = L.w_sh.detach().clone(); Wg = L.b_gate_up.weight.detach()[:832].clone()
    else: Wsh = torch.randn(H, dh, d, device="cuda", generator=g) / math.sqrt(d); Wg = torch.randn(832, d, device="cuda", generator=g) / math.sqrt(d)
    return Wsh.requires_grad_(True), Wg.requires_grad_(True)
def E(h, Wsh, Wg):
    v = torch.einsum('td,hcd->thc', h, Wsh)
    tr = -0.5 * torch.einsum('htn,thc,nhc->', a, v, v)
    u = h @ Wg.T; pot = -(torch.nn.functional.softplus(2 * u) - u - math.log(2.0)).sum()       # −Σ logcosh(u): 경사 = +W_gᵀ tanh(u)
    return tr + pot                                                                  # 가둠은 퍼텐셜 대신 공 사영으로
def C(h): return torch.nn.functional.cross_entropy(h @ Wcls.T + bcls, y[0])
def relax(h0, Wsh, Wg, beta=0.0, eta=0.1, steps=600, tol=1e-6, track=False, create_graph=False):
    h = h0.detach().requires_grad_(True) if create_graph else h0.detach(); hist = []; k = 0
    for k in range(steps):
        h_ = h if create_graph else h.detach().requires_grad_(True)
        F = E(h_, Wsh, Wg) + (beta * C(h_) if beta else 0.0)
        grad, = torch.autograd.grad(F, h_, create_graph=create_graph)
        h_new = h_ - eta * grad
        nrm = h_new.norm(dim=-1, keepdim=True)
        h_new = h_new * (R / (nrm + 1e-9)) if PROJ == "sphere" else h_new * torch.clamp(R / (nrm + 1e-9), max=1.0)   # sphere = RMSNorm, ball = 노름 자름
        step = ((h_new - h_).norm() / (h_.norm() + 1e-9)).item()
        if track: hist.append((float(F.item()), step))
        h = h_new if create_graph else h_new.detach()
        if not create_graph and step < tol: break
    return h, k + 1, hist
h0 = h_real[0].detach(); R = math.sqrt(d); PROJ = os.environ.get("PROJ", "sphere"); REF_STEPS = int(os.environ.get("REF_STEPS", 1200)); NUDGE_STEPS = int(os.environ.get("NUDGE_STEPS", 1500))
print(f"사영 = {PROJ}, 기준 펼침 = {REF_STEPS}스텝")
print(f"사영 반경 R = √d = {R:.2f}; 실제 상태의 칸 노름 중앙값 {h0.norm(dim=-1).median().item():.2f}")
for kind in (os.environ.get("KINDS", "ckpt,random").split(",")):
    Wsh, Wg = params(kind)
    print(f"\n===== 가중치 {kind} =====")
    print("[2번] 고정점 수렴 (‖Δh‖/‖h‖ < 1e-6, 600스텝 상한)")
    best_eta = None
    for eta in (0.2, 0.1, 0.05, 0.02, 0.01):
        h, nstep, hist = relax(h0, Wsh, Wg, eta=eta, track=True)
        Es = np.array([q[0] for q in hist]); ok = np.isfinite(Es).all(); last = hist[-1][1]
        mono = bool(ok and np.all(np.diff(Es) <= 1e-7 * np.abs(Es[:-1]) + 1e-5))
        status = ("수렴 %d스텝" % nstep) if (ok and last < 1e-6) else ("발산" if not ok else f"미수렴(마지막 변화 {last:.1e})")
        print(f"  η={eta:5.2f}: {status:>22}   E {Es[0]:.1f} → {Es[-1] if ok else float('nan'):.1f}   단조감소 {mono}", flush=True)
        if ok and last < 1e-6 and best_eta is None: best_eta = eta
    eta = best_eta or 0.05
    print(f"[3번] EqProp 경사 검증 (η={eta}). 기준 = 자유 이완을 autograd 로 펼친 dC/dθ (θ = W_sh, W_g)")
    Wsh, Wg = params(kind)
    hT, nT, _ = relax(h0, Wsh, Wg, eta=eta, steps=REF_STEPS, create_graph=True)
    Cval = C(hT); gW, gG = torch.autograd.grad(Cval, [Wsh, Wg]); ref = torch.cat([gW.flatten(), gG.flatten()])
    h_free, nf, _ = relax(h0, Wsh, Wg, eta=eta, steps=3000, tol=1e-8)
    def dE_dtheta(h):
        W2, G2 = Wsh.detach().requires_grad_(True), Wg.detach().requires_grad_(True)
        e = E(h.detach(), W2, G2); gw, gg = torch.autograd.grad(e, [W2, G2]); return torch.cat([gw.flatten(), gg.flatten()])
    g_free = dE_dtheta(h_free)
    print(f"  펼침 기준: {REF_STEPS}스텝, C(h_T) = {Cval.item():.4f}   자유 고정점: {nf}스텝, C(h⁰) = {C(h_free).item():.4f}, ‖ref‖ = {ref.norm().item():.3e}")
    for beta in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003):
        h_nudge, nn_, _ = relax(h_free, Wsh, Wg, beta=beta, eta=eta, steps=NUDGE_STEPS, tol=0.0)      # 조기 정지 없음
        est = (dE_dtheta(h_nudge) - g_free) / beta
        cos = torch.nn.functional.cosine_similarity(est, ref, dim=0).item(); rel = ((est - ref).norm() / ref.norm()).item()
        print(f"  β={beta:6.3f}: 밀림 위상 {nn_:4d}스텝   코사인 {cos:+.4f}   상대오차 {rel:.3f}", flush=True)
