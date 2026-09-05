"""v2.1 계열 체크포인트에서 블록 사상의 야코비안 비대칭 r = ‖J−Jᵀ‖/‖J+Jᵀ‖ 성분별 측정 (곱셈 읽기·sym 경계 지원).
env: CKPT(필수), PSI0=1 이면 ψ 를 0 으로 강제(가중치 그대로), PID, SEG."""
import os, importlib.util, numpy as np, torch
from torch.func import jvp
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = int(os.environ.get("PID", 155)); SEG = int(os.environ.get("SEG", 8)); K = 48
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.environ["CKPT"], map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1, amp=False, compile=False)
sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
if os.environ.get("PSI0") == "1":
    for L in I.layers: L.psi.data.zero_()
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
        s1 = (u * Jv).sum().item(); s2 = (v * Ju).sum().item(); d2 += (s1 - s2) ** 2; e2 += (s1 + s2) ** 2
    return (d2 / e2) ** 0.5
print(f"ckpt={os.path.basename(os.environ['CKPT'])} boundary={I.config.boundary} read={I.config.stdp_read} psi0={os.environ.get('PSI0','0')} dist_decay={I.config.dist_decay}  퍼즐 #{PID} 세그 {SEG}, fp32, K={K}")
print(f"{'성분':>28} | {'L0':>7} {'L1':>7}"); rows = {}
for b, c in sorted(cap.items()):
    L, h_in, AB, kc, w = c["L"], c["h"], c["AB"], c["kc"], c["w"]; li = 0 if L is I.layers[0] else 1
    h1 = h_in + inj
    def a_of(hh): return I.attn_xy(I.addr(hh, AB), kc)
    def a_eff_of(a):
        if I.config.stdp_read == "mul": return a * torch.exp((torch.sign(a) * w).clamp(-4, 4))
        lam = torch.sigmoid(L.lam_raw) if I.config.stdp_lam_fixed < 0 else float(I.config.stdp_lam_fixed); return (1 - lam) * a + lam * w
    def transport(hh, a_eff):
        v = torch.einsum('btd,hcd->bthc', hh, L.w_sh); o = torch.einsum('bhtn,bnhc->bthc', a_eff, v); return torch.einsum('bthc,hcd->btd', o, L.w_sh)
    a0 = a_of(h1); aeff0 = a_eff_of(a0)
    r_a = ((a0 - a0.transpose(-1, -2)).norm() / (a0 + a0.transpose(-1, -2)).norm()).item()
    r_aeff = ((aeff0 - aeff0.transpose(-1, -2)).norm() / (aeff0 + aeff0.transpose(-1, -2)).norm()).item()
    f_frozen = lambda hh: transport(hh, aeff0); f_full = lambda hh: transport(hh, a_eff_of(a_of(hh)))
    hb = h1 + f_full(h1)
    res = {"a (즉석 어텐션 행렬)": r_a, "a_eff (읽기 후)": r_aeff,
           "수송 Df, A 얼림": ratio(f_frozen, h1), "수송 Df, A 재계산": ratio(f_full, h1),
           f"경계 Db ({I.config.boundary})": ratio(lambda hh: I.boundary(L, hh) - hh, hb), "Φ": ratio(lambda hh: I.phi(hh), I.boundary(L, hb)),
           "블록 전체 J−I, A 얼림": ratio(lambda hh: I.phi(I.boundary(L, hh + transport(hh, aeff0))) - hh, h1),
           "블록 전체 J−I, A 재계산": ratio(lambda hh: I.phi(I.boundary(L, hh + f_full(hh))) - hh, h1)}
    for k_, v_ in res.items(): rows.setdefault(k_, {})[li] = v_
for k_, d in rows.items(): print(f"{k_:>28} | {d.get(0, float('nan')):7.3f} {d.get(1, float('nan')):7.3f}", flush=True)
