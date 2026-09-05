"""추론 중 샘플별 빠른 시냅스 u 의 절단 경사하강 — 이론 정합 판.
  결합: a_eff = (1−λ)a + λw + u.   w 는 모델의 EMA 상태(그대로), u 는 샘플별 파라미터(초기 0, 창 동안 고정).
  목적: J_K(u) = C(s_{t+K}(u)),  C = E[동료 충돌] + τ·평균 엔트로피 (확신 낮추기 봉쇄; 최솟값 = 확신 있는 유효 격자).
  갱신: 창(K 블록)마다 u ← u − η ∇_u J_K  (표준 경사하강). u=0 이면 모델 그대로.
  진단: 완답, 소프트 비용, argmax 위반 쌍 수, 평균 최대확률(확신도), 기준선 대비 얻음/잃음."""
import os, math, importlib.util, time, numpy as np, torch, torch.nn.functional as F
ROOT = "/workspace/LinearTuring"; N = int(os.environ.get("N", 512)); SEGS = int(os.environ.get("SEGS", 128)); BS = 256
K = int(os.environ.get("K", 16)); TAU = float(os.environ.get("TAU", 0.3)); TAU_END = float(os.environ.get("TAU_END", TAU))
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=BS, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
for p in m.parameters(): p.requires_grad_(False)
assert I.config.stdp and I.config.stdp_target == "faithful" and I.config.stdp_window == "psi" and not I.gate_on and I.config.stdp_diag == "keep"
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:N].reshape(N, 81).astype(int); Y = z["test_labels"][:N].reshape(N, 81).astype(int)
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = torch.from_numpy(((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)).cuda()
peer_f = peer.float(); NPAIR = peer_f.sum() / 2
def step_u(L, h, AB, kc, w, fresh, u):
    """train_kaggle.LT_Inner.step 의 faithful/psi 경로를 그대로 + 결합에 u 를 더함. u=None 이면 원본과 동일."""
    xy = I.addr(h, AB); a = I.attn_xy(xy, kc)
    v = torch.einsum('btd,hcd->bthc', h, L.w_sh)
    eta_ = torch.sigmoid(L.eta_raw); lam = torch.sigmoid(L.lam_raw) if I.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(I.config.stdp_lam_fixed))
    vv = v / (v.norm(dim=-1, keepdim=True) + I.config.eps); agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
    G = a * agree; gain = F.softplus(L.gain_raw) if I.config.stdp_gain_fixed < 0 else float(I.config.stdp_gain_fixed); tgt = gain * G
    if w is None: w = tgt
    else:
        w = torch.where(fresh.view(-1, 1, 1, 1), tgt, w) if fresh is not None else w
        w = (1 - eta_) * w + eta_ * tgt
    a_eff = (1 - lam) * a + lam * w
    if u is not None: a_eff = a_eff + u
    o = torch.einsum('bhtn,bnhc->bthc', a_eff, v); f = torch.einsum('bthc,hcd->btd', o, L.w_sh)
    return h + f, w
def probs(h): return torch.softmax(I.w_cls(h)[..., 2:11].float(), -1)                       # 숫자 1..9
def cost(h, tau):
    p = probs(h); viol = torch.einsum('btk,bnk,tn->b', p, p, peer_f) / (2 * NPAIR)              # 쌍당 기대 충돌
    ent = -(p * torch.log(p + 1e-12)).sum(-1).mean(-1) / math.log(9)                               # 칸 평균 정규화 엔트로피
    return viol + tau * ent, viol
def hard_viol(h):
    pred = I.w_cls(h).argmax(-1); return ((pred[:, :, None] == pred[:, None, :]) & peer[None]).sum((1, 2)).float() / 2
layers = list(I.layers); nl = len(layers); BLK_SEG = I.config.blocks_per_seg * nl
def run(eta, tau, calib=False):
    ex = np.zeros(SEGS); soft = np.zeros(SEGS); hard = np.zeros(SEGS); conf = np.zeros(SEGS); t0 = time.time(); solved_end = np.zeros(N, bool); grms = []
    for b in range(0, N, BS):
        x = torch.from_numpy(X[b:b+BS].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y[b:b+BS].astype(np.int32) + 1).cuda().long(); nb = len(x)
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(nb, dtype=torch.int32, device="cuda"))
        inj = I.injection(batch) * I.embed_scale; ABk = [(I.W_C(L), I.kernel(L)) for L in layers]
        h = I.init_hidden.expand(nb, 81, -1).clone(); w = None; fresh = torch.ones(nb, dtype=torch.bool, device="cuda")
        u = torch.zeros(nb, I.H, 81, 81, device="cuda") if eta > 0 else None
        total = SEGS * BLK_SEG; blk = 0
        while blk < total:
            do_grad = eta > 0
            if do_grad: u = u.detach().requires_grad_(True)
            hh, ww = h, w
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(do_grad):
                for j in range(K):
                    li = (blk + j) % nl; L = layers[li]; AB, kc = ABk[li]
                    hout, ww = step_u(L, hh + inj, AB, kc, ww, fresh, u); hh = I.phi(I.boundary(L, hout)); fresh = None
                    if (blk + j + 1) % BLK_SEG == 0:
                        s = (blk + j + 1) // BLK_SEG - 1
                        with torch.no_grad():
                            pred = I.w_cls(hh).argmax(-1); ok = (pred == y).all(-1); ex[s] += ok.sum().item()
                            cs, cv = cost(hh, tau); soft[s] += cv.sum().item(); hard[s] += hard_viol(hh).sum().item(); conf[s] += probs(hh).max(-1).values.mean(-1).sum().item()
                if do_grad:
                    frac = min(1.0, blk / max(total - K, 1)); tau_now = tau + (TAU_END - tau) * frac        # 선형 어닐링
                    J = cost(hh, tau_now)[0].sum(); g, = torch.autograd.grad(J, u)
            with torch.no_grad():
                if do_grad:
                    if calib: grms.append(g.float().pow(2).mean().sqrt().item())
                    u = u - eta * g.to(u.dtype)
                h = hh.detach(); w = ww.detach()
            blk += K
        solved_end[b:b+nb] = ok.cpu().numpy()
        print(f"  K={K} τ={tau} η={eta:<8g} batch {b//BS+1}/{-(-N//BS)}  {time.time()-t0:.0f}s  seg16 {int(ex[15])}  seg{SEGS} {int(ex[-1])}" + (f"  |∇u| rms {np.mean(grms):.2e}" if calib else ""), flush=True)
    return dict(ex=ex, soft=soft / N, hard=hard / N, conf=conf / N, solved_end=solved_end)
if __name__ == "__main__":
    BASE_F = os.path.join(ROOT, f"2026-09-05/results/json/winfer_baseline_N{N}_S{SEGS}.npz")
    if os.path.exists(BASE_F): bz = np.load(BASE_F); base = {k: bz[k] for k in bz.files}; print(f"기준선 로드 (seg{SEGS} {int(base['ex'][-1])})", flush=True)
    else: base = run(0.0, TAU); np.savez(BASE_F, **base); print("기준선 저장", flush=True)
    etas = [float(v) for v in os.environ.get("ETAS", "").split(",") if v]
    res = {0.0: base}
    for eta in etas: res[eta] = run(eta, TAU, calib=(eta == etas[0]))
    print(f"\nK={K} τ={TAU}→{TAU_END}   {'η':>8} | " + " ".join(f"seg{s:>3}" for s in (16, 32, 64, SEGS)) + " | 소프트위반 끝 | argmax 위반쌍/퍼즐 끝 | 확신도 끝 | 얻음/잃음")
    for eta, R in res.items():
        print(f"           {eta:8g} | " + " ".join(f"{int(R['ex'][s-1]):6d}" for s in (16, 32, 64, SEGS)) + f" | {R['soft'][-1]:.5f} | {R['hard'][-1]:8.2f} | {R['conf'][-1]:.3f} | +{int((R['solved_end'] & ~base['solved_end']).sum())} / −{int((~R['solved_end'] & base['solved_end']).sum())}", flush=True)
