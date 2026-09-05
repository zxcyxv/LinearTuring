"""FAIL run 의 위반 칸 4개가 받는 힘을 분해한다. 충돌 상대에게서 오는 밀어냄 vs 나머지 전부.
각 블록에서 f_t = Σ_n a_eff[t,n] · W_shᵀ v_n 을 상대별로 나누고, 그것을 로짓 방향(W_cls)으로 사영해
'현재 숫자의 로짓을 얼마나 깎는가' 를 마진과 비교한다."""
import os, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; S = 16
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=S + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
x = torch.from_numpy(X0[None].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y0[None].astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(1, dtype=torch.int32, device="cuda"))
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
rec = []; orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    xy = I.addr(h, AB); a = I.attn_xy(xy, kc); v = torch.einsum('btd,hcd->bthc', h, L.w_sh)
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    lam = torch.sigmoid(L.lam_raw) if I.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(I.config.stdp_lam_fixed))
    a_eff = (1 - lam) * a + lam * w_new                                      # [1,H,T,T]
    # 상대별 기여: contrib[t,n,:] = Σ_h a_eff[h,t,n] · (W_shᵀ v_n)_h   → [T,T,d]
    per = torch.einsum('bhtn,bnhc,hcd->btnd', a_eff, v, L.w_sh)[0]           # [T,T,d]
    h_end = I.phi(I.boundary(L, hout)); logit = I.w_cls(h_end)[0].float()      # [T,V]
    pred = logit.argmax(-1) - 1
    rec.append(dict(per=per.float().cpu(), logit=logit.cpu(), pred=pred.cpu().numpy(), Wc=I.w_cls.weight.float().cpu(), li=0 if L is I.layers[0] else 1))
    return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(S): carry, _ = m(carry, batch)
nb = len(rec); F = rec[-1]["pred"]
V = [(t, n) for t in range(81) for n in range(t + 1, 81) if peer[t, n] and F[t] == F[n]]
cells = sorted({t for p in V for t in p}); partner = {t: (n if t == p[0] else p[0]) for p in V for t in p for n in [p[1]]}
print("위반 쌍:", [(f"r{t//9}c{t%9}", f"r{n//9}c{n%9}", int(F[t]) ) for t, n in V])
def name(t): return f"r{t//9}c{t%9}"
print(f"\n{'block':>5} {'L':>1} {'칸':>5} {'현재':>3} {'정답':>3} {'마진':>6} | {'|충돌상대 기여|':>13} {'|다른동료 합|':>12} {'|비동료 합|':>10} | {'상대가 현재숫자 로짓에':>16} {'상대가 정답숫자 로짓에':>16} {'다른동료→현재':>12} {'비동료→현재':>10}")
for b in [15, 63, 127, 191, 254, 255]:
    R = rec[b]; per = R["per"]; Wc = R["Wc"]; logit = R["logit"]
    for t in cells:
        n_ = partner[t]; cur = int(R["pred"][t]); tru = int(Y0[t])
        top2 = logit[t].topk(2).values; margin = float(top2[0] - top2[1])
        others = [k for k in range(81) if peer[t, k] and k != n_]; non = [k for k in range(81) if not peer[t, k] and k != t]
        c_p = per[t, n_]; c_o = per[t, others].sum(0); c_n = per[t, non].sum(0)
        proj = lambda vec, dgt: float(vec @ Wc[dgt + 1])
        print(f"{b:5d} {R['li']:>1} {name(t):>5} {cur:3d} {tru:3d} {margin:6.1f} | {c_p.norm():13.3f} {c_o.norm():12.3f} {c_n.norm():10.3f} | {proj(c_p, cur):+16.2f} {proj(c_p, tru):+16.2f} {proj(c_o, cur):+12.2f} {proj(c_n, cur):+10.2f}")
    print()
# 위반 칸 마진 vs 나머지 틀린 칸 / 맞은 칸 마진 (마지막 세그 평균)
last = range(nb - 16, nb); wrong = F != Y0; blank = X0 == 0
mv = np.mean([[float((rec[b]["logit"][t].topk(2).values.diff().abs())) for t in cells] for b in last])
mw = np.mean([[float((rec[b]["logit"][t].topk(2).values.diff().abs())) for t in np.where(wrong & blank)[0] if t not in cells] for b in last])
mo = np.mean([[float((rec[b]["logit"][t].topk(2).values.diff().abs())) for t in np.where(~wrong & blank)[0]] for b in last])
print(f"마지막 세그 평균 마진: 위반 칸 4개 {mv:.1f}  |  다른 틀린 칸 {mw:.1f}  |  맞은 칸 {mo:.1f}")
# 위반 칸의 2등 후보가 무엇인지, 그걸로 바꾸면 위반이 어떻게 되나
print("\n위반 칸의 2등 후보와 그 경우 총 위반 수 (현재 2):")
g = F.copy()
for t in cells:
    top = rec[-1]["logit"][t].topk(3).indices.numpy() - 1; alt = int(top[1]) if top[1] >= 1 else int(top[2])
    g2 = g.copy(); g2[t] = alt; viol2 = sum(1 for i in range(81) for j in range(i + 1, 81) if peer[i, j] and g2[i] == g2[j])
    print(f"  {name(t)}: 현재 {int(F[t])} → 2등 {alt} (정답 {int(Y0[t])})  → 총 위반 {viol2}")
