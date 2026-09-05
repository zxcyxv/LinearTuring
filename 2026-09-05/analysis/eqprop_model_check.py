"""1단계 확인: 설계 변경(주소·값 분리 + ψ=0 + 경계 대칭 묶기 + Φ 처리) 후 값 블록 동역학이 경사 흐름인가.
r = ‖J−Jᵀ‖/‖J+Jᵀ‖ 를 성분별·전체로 측정. 체크포인트 L1 가중치와 무작위 가중치 둘 다."""
import os, importlib.util, math, numpy as np, torch
from torch.func import jvp
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; SEG = 8; K = 48; d = 832; H = 8
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1, amp=False)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
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
h_real = cap["h"] + I.embed_scale * I.injection(batch)                         # 실제 상태 (주입 후)
def ratio(fn, h0, K=K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed); d2 = e2 = 0.0
    for _ in range(K):
        u = torch.randn(h0.shape, device="cuda", generator=g); v = torch.randn(h0.shape, device="cuda", generator=g)
        _, Jv = jvp(fn, (h0,), (v,)); _, Ju = jvp(fn, (h0,), (u,))
        s1 = (u * Jv).sum().item(); s2 = (v * Ju).sum().item(); d2 += (s1 - s2) ** 2; e2 += (s1 + s2) ** 2
    return (d2 / max(e2, 1e-30)) ** 0.5
def build(weights, psi_zero):
    """값 블록 동역학의 성분들. 주소 h_a 는 고정(현재 상태), a 는 h_a 에서 1회 계산 → 값 이완 동안 상수."""
    if weights == "ckpt":
        Wsh = L.w_sh.detach().clone(); Wg = L.b_gate_up.weight.detach()[: L.b_gate_up.weight.shape[0] // 2].clone()   # [inter, d]
        psi = torch.zeros_like(L.psi) if psi_zero else L.psi.detach().clone()
    else:
        g = torch.Generator(device="cuda").manual_seed(1)
        Wsh = torch.randn(H, d // H, d, device="cuda", generator=g) / math.sqrt(d); Wg = torch.randn(3328, d, device="cuda", generator=g) / math.sqrt(d)
        psi = torch.zeros(H, d // H // 2, device="cuda") if psi_zero else (torch.rand(H, d // H // 2, device="cuda", generator=g) * 2 - 1) * math.pi
    # a: 주소 고정에서 1회 계산 (ψ 교체)
    kc = I.kernel(L, psi); a = I.attn_xy(I.addr(h_real, cap["AB"]), kc).detach()      # [1,H,T,T]
    gamma = I.gamma
    def transport(hv): v = torch.einsum('btd,hcd->bthc', hv, Wsh); o = torch.einsum('bhtn,bnhc->bthc', a, v); return torch.einsum('bthc,hcd->btd', o, Wsh)
    def boundary_tied(hv): u = hv @ Wg.T; return 0.5 * (u * u) @ Wg                    # ∇V, V = ⅙ Σ (Wg h)³  (W_d = Wgᵀ 묶기)
    def boundary_bilinear(hv):                                                          # 대조: 체크포인트 원래 경계 (묶지 않음)
        gate, up = L.b_gate_up(hv).chunk(2, -1); return L.b_down(0.5 * gate * up)
    def phi(hv): return hv / torch.sqrt(1 + gamma * hv.pow(2).sum(-1, keepdim=True))
    def confine(hv): return -gamma * hv * hv.pow(2).sum(-1, keepdim=True)              # −∇B, B = (γ/4)‖h‖⁴ 가둠 퍼텐셜
    E_grad = lambda hv: transport(hv) + boundary_tied(hv)                                # −∇E 의 값 블록 부분 (E = −½ Σ a⟨Wh,Wh⟩ − V)
    return dict(a=a, comps={
        "a (ψ 처리 후)": None,
        "수송 Df (A 고정, 값 블록)": transport,
        "경계 Db, 묶음 (∇V)": boundary_tied,
        "경계 Db, 원래 쌍선형 (대조)": boundary_bilinear,
        "Φ 단독": phi,
        "블록 A: Φ(h + 수송 + 묶은 경계) − h   [Φ 사영]": lambda hv: phi(hv + E_grad(hv)) - hv,
        "블록 B: h + 수송 + 묶은 경계 − ∇B   [Φ→가둠 퍼텐셜]": lambda hv: E_grad(hv) + confine(hv),
        "블록 C: Φ(h + 수송 + 쌍선형) − h   [현행 구조, ψ 처리만]": lambda hv: phi(hv + transport(hv) + boundary_bilinear(hv)) - hv,
    })
print(f"퍼즐 #{PID} 세그 {SEG} 실제 상태, 레이어 1, fp32, K={K}.   r = ‖J−Jᵀ‖/‖J+Jᵀ‖")
for weights in ("ckpt", "random"):
    for psi_zero in (False, True):
        B = build(weights, psi_zero); a = B["a"]
        print(f"\n[가중치 {weights}, ψ={'0' if psi_zero else '학습값'}]")
        print(f"  {'a 행렬 비대칭':>46}: {((a - a.transpose(-1,-2)).norm() / (a + a.transpose(-1,-2)).norm()).item():.4f}")
        for name, fn in B["comps"].items():
            if fn is None: continue
            print(f"  {name:>46}: {ratio(fn, h_real):.4f}", flush=True)
