"""#155 FAIL vs SOLVED: 행 위반 칸 4개(r0c2,r0c6,r1c4,r1c8)가 '비동료 중 정답 숫자를 든 칸'에게서 받는
정답 로짓 기여 — 거리별 나열, 거리 상관, 합계. 동료 소거·충돌 상대와 비교."""
import os, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; S = 16
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=2, seq_len=81, num_puzzle_identifiers=1, loops=S + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
dm = np.array([0, 1, 3, 8, 2, 4, 7, 5, 6, 9])
Xb = np.stack([X0, np.where(X0 > 0, dm[X0], 0)]); Yb = np.stack([Y0, dm[Y0]])
x = torch.from_numpy(Xb.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Yb.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(2, dtype=torch.int32, device="cuda"))
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
dist = np.abs(r[:, None] - r[None]) + np.abs(c[:, None] - c[None])
Wc = I.w_cls.weight.float()
CELLS = [2, 6, 13, 17]; nm = lambda t: f"r{t//9}c{t%9}"
rec = {}; st = {"b": 0}; orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    b = st["b"]; st["b"] += 1
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    if b % 16 == 1 and (b // 16 + 1) in (1, 4, 8, 13, 16):                 # 각 세그 두 번째 블록 = L1
        xy = I.addr(h, AB); a = I.attn_xy(xy, kc); v = torch.einsum('btd,hcd->bthc', h, L.w_sh)
        lam = torch.sigmoid(L.lam_raw) if I.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(I.config.stdp_lam_fixed))
        a_eff = (1 - lam) * a + lam * w_new
        per = torch.einsum('bhtn,bnhc,hcd->btnd', a_eff, v, L.w_sh).float()
        pred = (I.w_cls(I.phi(I.boundary(L, hout))).argmax(-1) - 1).cpu().numpy()
        out = {}
        for k in range(2):
            for t in CELLS:
                d = int(Yb[k][t]); contrib = (per[k, t] @ Wc[d + 1]).float().cpu().numpy()      # [81] 각 n → t 의 정답 로짓
                src = [n for n in range(81) if n != t and not peer[t, n] and pred[k][n] == d]
                el = [n for n in range(81) if peer[t, n] and pred[k][n] != pred[k][t]]
                conf = [n for n in range(81) if peer[t, n] and pred[k][n] == pred[k][t]]
                out[(k, t)] = dict(cur=int(pred[k][t]), d=d, src=[(n, int(dist[t, n]), float(contrib[n])) for n in src],
                                   el_sum=float(contrib[el].sum()), conf_sum=float(contrib[conf].sum()), nonpeer_other=float(sum(contrib[n] for n in range(81) if n != t and not peer[t, n] and pred[k][n] != d)))
        rec[b // 16 + 1] = out
    return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(S): carry, _ = m(carry, batch)
RUN = ["FAIL(원본)", "SOLVED(치환)"]
for sg in sorted(rec):
    print(f"\n===== 세그 {sg} =====")
    print(f"{'런':>12} {'칸':>5} {'현재/정답':>8} | {'비동료·정답칸 수':>12} {'합':>7} {'거리별 (거리:기여)':>40} {'거리-기여 상관':>10} | {'동료소거→정답':>10} {'충돌상대→정답':>10} {'비동료·다른숫자→정답':>14}")
    for k in range(2):
        for t in CELLS:
            o = rec[sg][(k, t)]; src = sorted(o["src"], key=lambda q: q[1])
            tot = sum(q[2] for q in src); ds = np.array([q[1] for q in src]); cs = np.array([q[2] for q in src])
            corr = float(np.corrcoef(ds, cs)[0, 1]) if len(src) > 2 and cs.std() > 0 else float("nan")
            byd = " ".join(f"{q[1]}:{q[2]:+.1f}" for q in src)
            print(f"{RUN[k]:>12} {nm(t):>5} {o['cur']:>3}/{o['d']:<4} | {len(src):12d} {tot:+7.1f} {byd:>40} {corr:10.2f} | {o['el_sum']:+10.1f} {o['conf_sum']:+10.1f} {o['nonpeer_other']:+14.1f}")
