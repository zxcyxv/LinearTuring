"""Scellier 2018 정리 1 검증 — 원래 캐글 체크포인트(비대칭, 있는 그대로).
동역학: s ← F(s) = [L0 블록 → L1 블록] (post 순서, w 는 carry 값으로 고정). μ = F(s) − s.
진짜 경사: dJ/dθ = (∂C/∂s)(I − DF)⁻¹(∂F/∂θ)  ← 수반 λ 를 Neumann 급수(λ ← ∂C/∂s + DFᵀλ)로 풀어 정확히 계산.
프록시(pre×Δpost 두-위상): 밀림 동역학 s ← F(s) − β ∂C/∂s 로 s^β 에 도달, ν = (∂F/∂θ)ᵀ(s⁰)·(s^β − s⁰)/β  (끝점형)
                          및 경로 적분형 Σ_t (∂F/∂θ)ᵀ(s_t)·(s_{t+1}−s_t)/β.  코사인(ν, −dJ/dθ) 을 β 별·파라미터군별로."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; PID = int(os.environ.get("PID", 0)); SEG = 8
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1, amp=False)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
for p in m.parameters(): p.requires_grad_(False)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
x = torch.from_numpy(X0[None].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y0[None].astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(1, dtype=torch.int32, device="cuda"))
with torch.no_grad():
    with torch.device("cuda"): carry = m.initial_carry(batch)
    for si in range(SEG): carry, _ = m(carry, batch)
    s0_init = carry.current_hidden.clone(); w_fixed = carry.coupling.clone()
    inj = I.injection(batch) * I.embed_scale
w_fixed = w_fixed.detach().clone().requires_grad_(True)
theta = {n: p for n, p in I.named_parameters() if n.endswith("w_sh") or n.endswith("b_down.weight")}; theta["w (결합 기억)"] = w_fixed
for p in theta.values(): p.requires_grad_(True)
names = list(theta); print("θ =", names, flush=True)
def F(s):
    h = s
    for L in I.layers:
        AB = I.W_C(L); kc = I.kernel(L)
        h = h + inj
        hout, _ = I.step(L, h, AB, kc, w_fixed, None, None, apply_phi=False)
        h = I.phi(I.boundary(L, hout))
    return h
gen = torch.Generator(device="cuda").manual_seed(0)
T_LOGIT = 5.0 * torch.nn.functional.one_hot(torch.randint(1, 10, (81,), device="cuda", generator=gen), 11).float()   # 무작위 목표 로짓 (경사 0 방지)
def C(s): return 0.5 * (I.w_cls(s)[0] - T_LOGIT).pow(2).sum() / 81
def flat(gs): return torch.cat([g.reshape(-1) for g in gs])
def vjp_theta(s, cot):
    s = s.detach(); out = F(s); gs = torch.autograd.grad(out, list(theta.values()), grad_outputs=cot, allow_unused=True)
    return flat([g if g is not None else torch.zeros_like(p) for g, p in zip(gs, theta.values())])
def vjp_s(s, cot):
    s = s.detach().requires_grad_(True); out = F(s); g, = torch.autograd.grad(out, s, grad_outputs=cot); return g
s = s0_init; t0 = time.time()
for k in range(400):
    with torch.no_grad(): s_new = F(s); res = ((s_new - s).norm() / s.norm()).item(); s = s_new
    if res < 1e-7: break
print(f"[자유 이완] {k+1}주기, 상대 잔차 {res:.2e}, C(s⁰) = {C(s).item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
s0 = s.detach()
sg = s0.clone().requires_grad_(True); g_s, = torch.autograd.grad(C(sg), sg)
lam = g_s.clone(); term = g_s.clone()
for k in range(600):
    term = vjp_s(s0, term); lam = lam + term
    if term.norm().item() < 1e-9 * lam.norm().item(): break
print(f"[수반 급수] {k+1}항, 마지막 항/합 = {(term.norm()/lam.norm()).item():.2e}", flush=True)
grad_true = vjp_theta(s0, lam)
print(f"‖dJ/dθ‖ = {grad_true.norm().item():.3e}", flush=True)
idx = {}; off = 0
for n, p in theta.items(): idx[n] = slice(off, off + p.numel()); off += p.numel()
def cos(a, b): return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
print(f"\n{'β':>7} | {'밀림 주기':>7} | {'끝점형 ν 코사인':>13} | {'경로적분형 ν 코사인':>15} | " + " ".join(f"{(n.split('.')[-2][:6]+'L'+n.split('.')[1]) if '.' in n else n:>10}" for n in names), flush=True)
if res > 1e-4: print(f"→ 고정점 아님 (잔차 {res:.2e}) — 정리 1 전제 불성립, 코사인 생략"); raise SystemExit
for beta in (0.1, 0.03):
    sb = s0.clone(); nu_path = torch.zeros_like(grad_true)
    for k in range(400):
        sbg = sb.detach().requires_grad_(True); gC, = torch.autograd.grad(C(sbg), sbg)
        with torch.no_grad(): sb_new = F(sb) - beta * gC
        ds = (sb_new - sb).detach()
        nu_path = nu_path + vjp_theta(sb, ds)
        res = (ds.norm() / sb.norm()).item(); sb = sb_new.detach()
        if res < 1e-8: break
    nu_end = vjp_theta(s0, (sb - s0)) / beta; nu_path = nu_path / beta
    per = " ".join(f"{cos(nu_end[idx[n]], -grad_true[idx[n]]):10.3f}" for n in names)
    print(f"{beta:7.3f} | {k+1:7d} | {cos(nu_end, -grad_true):13.4f} | {cos(nu_path, -grad_true):15.4f} | {per}", flush=True)
