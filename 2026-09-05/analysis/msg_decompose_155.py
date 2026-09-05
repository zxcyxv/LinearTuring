"""r0c2 (세그 8, FAIL) 가 '비동료·정답 9 를 든 칸' 들에게서 받는 기여를 헤드별로 분해: D · 위상정렬 · 값사영."""
import os, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; S = 8; T_CELL = 2
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=S + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
x = torch.from_numpy(X0[None].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y0[None].astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(1, dtype=torch.int32, device="cuda"))
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
dist = np.abs(r[:, None] - r[None]) + np.abs(c[:, None] - c[None]); nm = lambda t: f"r{t//9}c{t%9}"
Wc = I.w_cls.weight.float(); st = {"b": 0}; orig = I.step; out = {}
def hooked(L, h, AB, kc, *a_, **k_):
    b = st["b"]; st["b"] += 1
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    if b == (S - 1) * 16 + 1:
        xy = I.addr(h, AB); a = I.attn_xy(xy, kc)[0]                      # [H,T,T]
        D = kc[0]                                                         # [H,T,T] 감쇠
        phase = a / D.clamp_min(1e-6)                                     # 위상 정렬 부분 Σ r r cos
        lam = torch.sigmoid(L.lam_raw)[:, 0, 0] if I.config.stdp_lam_fixed < 0 else torch.full((8,), float(I.config.stdp_lam_fixed), device="cuda")
        a_eff = (1 - lam.view(-1, 1, 1)) * a + lam.view(-1, 1, 1) * w_new[0]
        v = torch.einsum('btd,hcd->bthc', h, L.w_sh)[0]                   # [T,H,C]
        pred = (I.w_cls(I.phi(I.boundary(L, hout))).argmax(-1)[0] - 1).cpu().numpy()
        d = int(Y0[T_CELL]); wd = Wc[d + 1]
        val = torch.einsum('nhc,hcd,d->nh', v, L.w_sh, wd).float()        # 값 사영 [T,H]: ⟨W_cls[d], W_shᵀ v_n⟩
        out.update(a=a.float().cpu().numpy(), D=D.float().cpu().numpy(), phase=phase.float().cpu().numpy(), a_eff=a_eff.float().cpu().numpy(),
                   val=val.cpu().numpy(), pred=pred, lam=lam.float().cpu().numpy(), alpha=L.alpha.flatten().float().cpu().numpy())
    return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(S): carry, _ = m(carry, batch)
t = T_CELL; d = int(Y0[t]); src = [n for n in range(81) if n != t and not peer[t, n] and out["pred"][n] == d]
tot = {n: float((out["a_eff"][:, t, n] * out["val"][n]).sum()) for n in src}
print(f"t = {nm(t)}, 정답 {d}, 세그 {S} 블록 2 (L1).  비동료·정답칸: " + "  ".join(f"{nm(n)}(d{dist[t,n]}: {tot[n]:+.2f})" for n in sorted(src, key=lambda q: dist[t, q])))
print("헤드별 α:", np.round(out["alpha"], 3), " λ:", np.round(out["lam"], 2))
far = max(src, key=lambda n: dist[t, n]); near = min(src, key=lambda n: (abs(tot[n]), dist[t, n]))
big_near = max([n for n in src if dist[t, n] <= 7], key=lambda n: tot[n])
for n, tag in [(far, "가장 먼 칸"), (big_near, "가까운데 큰 칸"), (near, "기여 ≈ 0 인 칸")]:
    print(f"\n--- {tag}: n = {nm(n)}, 거리 {dist[t,n]}, 총 기여 {tot[n]:+.2f}")
    print(f"{'head':>4} {'D(Δ)':>6} {'위상정렬':>8} {'a':>7} {'w':>7} {'a_eff':>7} {'값사영':>7} {'기여':>7}")
    for hh in range(8):
        a_ = out["a"][hh, t, n]; D_ = out["D"][hh, t, n]; ph = out["phase"][hh, t, n]; ae = out["a_eff"][hh, t, n]; vv = out["val"][n, hh]
        w_ = (ae - (1 - out["lam"][hh]) * a_) / max(out["lam"][hh], 1e-6)
        print(f"{hh:4d} {D_:6.2f} {ph:+8.3f} {a_:+7.3f} {w_:+7.3f} {ae:+7.3f} {vv:+7.2f} {ae*vv:+7.2f}")
