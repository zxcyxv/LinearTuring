"""비동료 칸의 '같은 숫자' 증거 채널. 세그 16 첫 블록(L1)에서, 빈칸 t 의 정답 숫자 d 로짓에 대한 기여를
보내는 칸 n 의 부류별·거리별로 합산: (a) 비동료이면서 현재 d 를 든 칸, (b) 비동료이면서 다른 숫자, (c) 동료.
거리 = |Δ|₁. 256퍼즐."""
import os, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); N = 256; SEGS = 16
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=N, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:N].reshape(N, 81).astype(int); Y = z["test_labels"][:N].reshape(N, 81).astype(int)
x = torch.from_numpy(X.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device="cuda"))
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = torch.from_numpy(((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)).cuda()
dist = torch.from_numpy(np.abs(r[:, None] - r[None]) + np.abs(c[:, None] - c[None])).cuda()
Wc = I.w_cls.weight.float()
acc = {}; state = {"blk": 0}; orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    state["blk"] += 1; b = state["blk"] - 1
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    if b == 15 * 16 + 1:                                                  # 세그 16, 두 번째 블록 (L1)
        xy = I.addr(h, AB); a = I.attn_xy(xy, kc); v = torch.einsum('btd,hcd->bthc', h, L.w_sh)
        lam = torch.sigmoid(L.lam_raw) if I.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(I.config.stdp_lam_fixed))
        a_eff = (1 - lam) * a + lam * w_new
        per = torch.einsum('bhtn,bnhc,hcd->btnd', a_eff, v, L.w_sh).float()   # [B,T,T,d]
        pred = (I.w_cls(I.phi(I.boundary(L, hout))).argmax(-1) - 1)          # 현재 답 [B,T]
        blank = (x == 1); yt = y - 1                                             # 정답 숫자 [B,T]
        # 각 (b,t): 정답 d 의 로짓 방향 Wc[d+1] 로 사영 → contrib[b,t,n]
        Wd = Wc[yt + 1]                                                          # [B,T,d]
        contrib = torch.einsum('btnd,btd->btn', per, Wd)                         # [B,T,T]
        same = (pred[:, None, :] == yt[:, :, None])                              # n 이 지금 t 의 정답 숫자를 들고 있나 [B,T,N]
        P = peer[None]; Dm = dist[None].expand(N, 81, 81)
        bm = blank[:, :, None].expand(N, 81, 81) & ~torch.eye(81, dtype=torch.bool, device="cuda")[None]
        mk_same = ~P & same & bm; mk_el = P & ~same & bm
        acc["_digit"] = {}
        for dgt in range(1, 10):
            sel = (yt == dgt)[:, :, None].expand(N, 81, 81)
            m1 = mk_same & sel; m2 = mk_el & sel
            acc["_digit"][dgt] = (contrib[m1].sum().item() / max(m1.sum().item(), 1), m1.sum().item() / max((blank & (yt == dgt)).sum().item(), 1),
                                  contrib[m2].sum().item() / max(m2.sum().item(), 1))
        for name, mask in [("비동료·같은숫자", ~P & same), ("비동료·다른숫자", ~P & ~same), ("동료·같은숫자(위반)", P & same), ("동료·다른숫자", P & ~same)]:
            mk = mask & bm
            tot = contrib[mk].sum().item() / blank.sum().item()                  # 빈칸당 총 기여
            row = {"총(빈칸당)": tot}
            for lo, hi in [(1, 2), (3, 4), (5, 8), (9, 16)]:
                mm = mk & (Dm >= lo) & (Dm <= hi); row[f"d{lo}-{hi}: 칸당"] = contrib[mm].sum().item() / max(mm.sum().item(), 1)
                row[f"d{lo}-{hi}: 수"] = mm.sum().item() / blank.sum().item()
            acc[name] = row
    return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(SEGS): carry, _ = m(carry, batch)
print("세그 16 L1 블록, 빈칸 t 의 '정답 숫자' 로짓에 대한 기여 (양수 = 정답을 밀어줌). 256퍼즐 평균.")
print(f"{'보내는 칸 부류':>14} | {'빈칸당 총기여':>10} | " + " | ".join(f"{'d'+str(lo)+'-'+str(hi)+' 칸당기여 (칸수)':>20}" for lo, hi in [(1,2),(3,4),(5,8),(9,16)]))
for name, row in acc.items():
    if name == "_digit": continue
    print(f"{name:>14} | {row['총(빈칸당)']:10.2f} | " + " | ".join(f"{row[f'd{lo}-{hi}: 칸당']:+8.3f} ({row[f'd{lo}-{hi}: 수']:4.1f})" for lo, hi in [(1,2),(3,4),(5,8),(9,16)]))

print("\n정답 숫자 d 별: 비동료·같은숫자 칸당 지지 / 그런 칸 수(빈칸당) / 동료 소거 칸당 지지")
for dgt in range(1, 10):
    a_, n_, e_ = acc["_digit"][dgt]; print(f"  d={dgt}: 같은숫자 지지 {a_:+.3f}  (칸 수 {n_:.1f})   동료 소거 {e_:+.3f}")
vals = [acc["_digit"][d][0] for d in range(1, 10)]; print(f"  숫자 간 편차: 최소 {min(vals):+.3f} 최대 {max(vals):+.3f} (비 {max(vals)/min(vals):.2f}배)")
