"""FAIL run: 충돌 상대의 밀어냄(현재 숫자 로짓 기여)이 세그먼트에 따라 커지는가, 진동하는가."""
import os, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; S = 32
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
Wc = I.w_cls.weight.float()                                                  # [V,d]
rec = []; orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    xy = I.addr(h, AB); a = I.attn_xy(xy, kc); v = torch.einsum('btd,hcd->bthc', h, L.w_sh)
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    lam = torch.sigmoid(L.lam_raw) if I.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(I.config.stdp_lam_fixed))
    a_eff = (1 - lam) * a + lam * w_new
    per = torch.einsum('bhtn,bnhc,hcd->btnd', a_eff, v, L.w_sh)[0].float()  # [T,T,d]
    plog = (per @ Wc.T).float()                                                        # [T,T,V]  상대 n 이 t 의 각 숫자 로짓에 주는 기여
    h_end = I.phi(I.boundary(L, hout)); logit = I.w_cls(h_end)[0].float(); pred = (logit.argmax(-1) - 1).cpu().numpy()
    rec.append(dict(plog=plog.cpu().numpy(), pred=pred, li=0 if L is I.layers[0] else 1, margin=(logit.topk(2, -1).values.diff(dim=-1).abs()[:, 0]).cpu().numpy()))
    return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(S): carry, _ = m(carry, batch)
L1 = [b for b in range(len(rec)) if rec[b]["li"] == 1]
def pairs_at(b): p = rec[b]["pred"]; return [(t, n) for t in range(81) for n in range(t + 1, 81) if peer[t, n] and p[t] == p[n]]
V16 = pairs_at(16 * 16 - 1); V32 = pairs_at(len(rec) - 1)
finalV = sorted(set(V16) | set(V32))
print("세그16 끝 위반 쌍:", [(f"r{t//9}c{t%9}", f"r{n//9}c{n%9}") for t, n in V16], "  세그32 끝 위반 쌍:", [(f"r{t//9}c{t%9}", f"r{n//9}c{n%9}") for t, n in V32])
print("세그별 끝 블록 틀린 칸 수:", [int((rec[16*sg+15]["pred"] != Y0).sum()) for sg in range(S)])
print("A) 마지막 위반 쌍", [(f"r{t//9}c{t%9}", f"r{n//9}c{n%9}") for t, n in finalV], "— 상대가 t 의 '현재 숫자' 로짓에 주는 기여 (레이어1, 세그별 8블록 평균). 음수 = 밀어냄")
print(f"{'seg':>3} | " + " ".join(f"{'r'+str(t//9)+'c'+str(t%9)+'←'+'r'+str(n//9)+'c'+str(n%9):>14}" for t, n in finalV) + " | 같은 값?  t 마진")
for sg in range(S):
    bl = [b for b in L1 if b // 16 == sg]; row = f"{sg+1:3d} | "
    for t, n in finalV:
        val = np.mean([rec[b]["plog"][t, n, rec[b]["pred"][t] + 1] for b in bl]); row += f"{val:14.2f}"
    same = " ".join("=" if rec[bl[-1]]["pred"][t] == rec[bl[-1]]["pred"][n] else "≠" for t, n in finalV)
    mg = " ".join(f"{rec[bl[-1]]['margin'][t]:4.1f}" for t, n in finalV)
    print(row + f" |   {same}    {mg}")
print("\nB) 각 블록 시점의 실제 위반 쌍 전체 — 밀어냄 평균 (상대→현재 숫자 로짓), 위반 수, 위반 칸 마진 (레이어1, 세그별 평균)")
print(f"{'seg':>3} | {'위반 수':>6} {'밀어냄 평균':>10} {'밀어냄 최소':>10} {'위반칸 마진':>10} {'비위반 틀린칸 마진':>16} {'맞은칸 마진':>10}")
blank = X0 == 0
for sg in range(S):
    bl = [b for b in L1 if b // 16 == sg]; pushes = []; nv = []; mv = []; mw = []; mo = []
    for b in bl:
        p = rec[b]["pred"]; V = [(t, n) for t in range(81) for n in range(81) if t != n and peer[t, n] and p[t] == p[n]]
        nv.append(len(V) // 2)
        if V: pushes += [rec[b]["plog"][t, n, p[t] + 1] for t, n in V]; vc = sorted({t for t, _ in V}); mv += list(rec[b]["margin"][vc])
        wr = (p != Y0) & blank; vset = {t for t, _ in V}
        mw += [rec[b]["margin"][t] for t in np.where(wr)[0] if t not in vset]; mo += list(rec[b]["margin"][(~(p != Y0)) & blank])
    print(f"{sg+1:3d} | {np.mean(nv):6.1f} {np.mean(pushes):10.2f} {np.min(pushes):10.2f} {np.mean(mv):10.1f} {np.mean(mw) if mw else float('nan'):16.1f} {np.mean(mo):10.1f}")
