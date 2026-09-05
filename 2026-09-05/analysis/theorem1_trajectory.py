"""정리 1 의 궤적 판 (논문 각주): J̄ = (1/K) Σ_t C(s_t) (자유 궤적 창 K주기), 진짜 경사 = 창을 autograd 로 펼침.
프록시 = (1/β) Σ_t [ (∂F/∂θ)ᵀ(s^β_t) Δs^β_t − (∂F/∂θ)ᵀ(s⁰_t) Δs⁰_t ]  (밀림 경로적분 − 자유 경로적분, 같은 시작점·같은 창)."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; PID = int(os.environ.get("PID", 155)); SEG = 8; K = int(os.environ.get("K", 32)); BURN = 100
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
    s_start = carry.current_hidden.clone(); w_fixed = carry.coupling.clone(); inj = I.injection(batch) * I.embed_scale
w_fixed = w_fixed.detach().clone().requires_grad_(True)
theta = {n: p for n, p in I.named_parameters() if n.endswith("w_sh") or n.endswith("b_down.weight")}; theta["w"] = w_fixed
for p in theta.values(): p.requires_grad_(True)
names = list(theta)
def F(s):
    h = s
    for L in I.layers:
        AB = I.W_C(L); kc = I.kernel(L); h = h + inj
        hout, _ = I.step(L, h, AB, kc, w_fixed, None, None, apply_phi=False); h = I.phi(I.boundary(L, hout))
    return h
gen = torch.Generator(device="cuda").manual_seed(0)
T_LOGIT = 5.0 * torch.nn.functional.one_hot(torch.randint(1, 10, (81,), device="cuda", generator=gen), 11).float()
def C(s): return 0.5 * (I.w_cls(s)[0] - T_LOGIT).pow(2).sum() / 81
def flat(gs): return torch.cat([g.reshape(-1) for g in gs])
def vjp_theta(s, cot):
    s = s.detach(); out = F(s); gs = torch.autograd.grad(out, list(theta.values()), grad_outputs=cot, allow_unused=True)
    return flat([g if g is not None else torch.zeros_like(p) for g, p in zip(gs, theta.values())])
# 번인
s = s_start
with torch.no_grad():
    for _ in range(BURN): s = F(s)
    res = ((F(s) - s).norm() / s.norm()).item()
s0 = s.detach(); print(f"퍼즐 #{PID}: 번인 {BURN}주기 뒤 잔차 {res:.2e}  ({'고정점' if res < 1e-4 else '비정상 궤적'})", flush=True)
# 진짜 경사: 창 K 주기 펼침
for p in theta.values(): p.grad = None
sK = s0; Jbar = 0.0
for t in range(K):
    sK = F(sK); Jbar = Jbar + C(sK) / K
Jbar.backward(); grad_true = flat([p.grad for p in theta.values()]).clone()
print(f"J̄ (창 {K}주기) = {Jbar.item():.2f}, ‖dJ̄/dθ‖ = {grad_true.norm().item():.3e}", flush=True)
idx = {}; off = 0
for n, p in theta.items(): idx[n] = slice(off, off + p.numel()); off += p.numel()
def cos(a, b): return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
SEP = {}
def path_integral(beta, ref=None):
    s = s0.clone(); acc = torch.zeros_like(grad_true); traj = []
    for t in range(K):
        if beta:
            sg = s.detach().requires_grad_(True); gC, = torch.autograd.grad(C(sg), sg)
            with torch.no_grad(): s_new = F(s) - beta * gC
        else:
            with torch.no_grad(): s_new = F(s)
        acc = acc + vjp_theta(s, (s_new - s).detach()); s = s_new.detach(); traj.append(s)
        if ref is not None and beta: SEP.setdefault(beta, []).append(((s - ref[t]).norm() / (beta * ref[t].norm())).item())
    return (acc, traj) if ref is None else acc
free, free_traj = path_integral(0.0)
print(f"\n{'β':>6} | {'대조 경로적분 ν 코사인':>18} | " + " ".join(f"{(n.split('.')[-2][:6]+'L'+n.split('.')[1]) if '.' in n else n:>10}" for n in names), flush=True)
for beta in (0.1, 0.03, 0.01):
    nu = (path_integral(beta, ref=free_traj) - free) / beta
    sp = SEP[beta]; print(f"{beta:6.2f} | {cos(nu, -grad_true):18.4f} | " + " ".join(f"{cos(nu[idx[n]], -grad_true[idx[n]]):10.3f}" for n in names) + f"   | ‖s^β−s⁰‖/(β‖s⁰‖): t=1 {sp[0]:.3f}, t=K/2 {sp[len(sp)//2]:.3f}, t=K {sp[-1]:.3f}", flush=True)
