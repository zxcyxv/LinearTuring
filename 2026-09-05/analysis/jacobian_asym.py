"""현재 체크포인트에서 블록 사상의 야코비안 비대칭 비율 r = ‖J − Jᵀ‖_F / ‖J + Jᵀ‖_F 를 성분별로 측정.
추정: u,v ~ N(0,I) 에 대해 E[(uᵀMv)²] = ‖M‖_F². uᵀJv 와 vᵀJu 를 JVP 로 계산 → d = uᵀ(J−Jᵀ)v, e = uᵀ(J+Jᵀ)v.
성분: (1) a 행렬 자체  (2) 수송 Df, A 얼림  (3) 수송 Df, A 재계산(전체)  (4) 경계 Db  (5) Φ  (6) 블록 전체 J−I."""
import os, importlib.util, numpy as np, torch
from torch.func import jvp
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = int(os.environ.get("PID", 155)); SEG = 8; K = 48
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1, amp=False)   # fp32
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
x = torch.from_numpy(X0[None].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y0[None].astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(1, dtype=torch.int32, device="cuda"))
cap = {}; st = {"b": 0}; orig = I.step
def hooked(L, h, AB, kc, w=None, fresh=None, kcb=None, vprev=None, apply_phi=True):
    b = st["b"]; st["b"] += 1
    if b in ((SEG - 1) * 16, (SEG - 1) * 16 + 1): cap[b] = dict(L=L, h=h.clone(), AB=AB, kc=kc, w=(w.clone() if w is not None else None))
    return orig(L, h, AB, kc, w, fresh, kcb, vprev, apply_phi)
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(SEG): carry, _ = m(carry, batch)
inj = I.embed_scale * I.injection(batch)
def ratio(fn, h0, K=K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed); d2 = e2 = 0.0
    for _ in range(K):
        u = torch.randn(h0.shape, device=h0.device, dtype=h0.dtype, generator=g); v = torch.randn(h0.shape, device=h0.device, dtype=h0.dtype, generator=g)
        _, Jv = jvp(fn, (h0,), (v,)); _, Ju = jvp(fn, (h0,), (u,))
        s1 = (u * Jv).sum().item(); s2 = (v * Ju).sum().item()
        d2 += (s1 - s2) ** 2; e2 += (s1 + s2) ** 2
    return (d2 / e2) ** 0.5
print(f"퍼즐 #{PID}, 세그 {SEG} 첫 두 블록 (L0, L1), fp32, 확률 추정 K={K}쌍.   r = ‖J−Jᵀ‖/‖J+Jᵀ‖ (0 = 대칭, 1 = 완전 반대칭)")
print(f"{'성분':>28} | {'L0':>7} {'L1':>7}")
rows = {}
for b, c in sorted(cap.items()):
    L, h_in, AB, kc, w = c["L"], c["h"], c["AB"], c["kc"], c["w"]; li = 0 if L is I.layers[0] else 1
    h1 = h_in + inj                                                             # step 입력 (post: 주입 후)
    lam = torch.sigmoid(L.lam_raw) if I.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(I.config.stdp_lam_fixed))
    def a_of(hh): return I.attn_xy(I.addr(hh, AB), kc)
    def transport(hh, a_eff):
        v = torch.einsum('btd,hcd->bthc', hh, L.w_sh); o = torch.einsum('bhtn,bnhc->bthc', a_eff, v); return torch.einsum('bthc,hcd->btd', o, L.w_sh)
    a0 = a_of(h1); aeff0 = (1 - lam) * a0 + lam * w
    r_a = ((a0 - a0.transpose(-1, -2)).norm() / (a0 + a0.transpose(-1, -2)).norm()).item()
    r_aeff = ((aeff0 - aeff0.transpose(-1, -2)).norm() / (aeff0 + aeff0.transpose(-1, -2)).norm()).item()
    f_frozen = lambda hh: transport(hh, aeff0)
    f_full = lambda hh: transport(hh, (1 - lam) * a_of(hh) + lam * w)           # w 는 carry 값 고정 (갱신 항 제외)
    f_full_w = lambda hh: transport(hh, (1 - lam) * a_of(hh) + lam * ((1 - torch.sigmoid(L.eta_raw)) * w + torch.sigmoid(L.eta_raw) * a_of(hh)))  # w 갱신 포함 (addr 형)
    hb = h1 + f_full(h1)
    b_fn = lambda hh: I.boundary(L, hh) - hh
    phi_fn = lambda hh: I.phi(hh)
    block = lambda hh: I.phi(I.boundary(L, hh + f_full(hh))) - hh
    block_frozen = lambda hh: I.phi(I.boundary(L, hh + transport(hh, aeff0))) - hh
    res = {"a (즉석 어텐션 행렬)": r_a, "a_eff = (1−λ)a + λw": r_aeff,
           "수송 Df, A 얼림": ratio(f_frozen, h1), "수송 Df, A 재계산": ratio(f_full, h1), "수송 Df, A·w 재계산": ratio(f_full_w, h1),
           "경계 Db (쌍선형)": ratio(b_fn, hb), "Φ": ratio(phi_fn, I.boundary(L, hb)),
           "블록 전체 J−I, A 얼림": ratio(block_frozen, h1), "블록 전체 J−I, A 재계산": ratio(block, h1)}
    for k_, v_ in res.items(): rows.setdefault(k_, {})[li] = v_
for k_, d in rows.items(): print(f"{k_:>28} | {d.get(0, float('nan')):7.3f} {d.get(1, float('nan')):7.3f}")
