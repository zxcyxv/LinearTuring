"""MI vs STDP-w: 완화 퍼즐(주어진 칸 k개 제거)의 유효 해 집합을 열거해 진짜 상관 I(x_t;x_n|givens) 를 구하고,
캐글 체크포인트의 결합 기억 w / 즉석 결합 a_eff 가 그것을 담는지 판정."""
import os, sys, time, json, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False)
NPZ = 64; CAP = 5000; LO, HI = 200, 5000; TLIM = 60.0; SEGS = 16; REC_SEGS = (1, 4, 16)
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer_np = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
PEERS = [np.where(peer_np[i])[0].tolist() for i in range(81)]
# ---------------- 열거기 (비트마스크 후보 + MRV + naked single 전파)
class Timeout(Exception): pass
def enumerate_solutions(grid, cap, tlim):
    t0 = time.time(); sols = []
    cand = [0] * 81
    for i in range(81):
        if grid[i]: cand[i] = 1 << grid[i]
        else:
            m = 0x3FE
            for j in PEERS[i]:
                if grid[j]: m &= ~(1 << grid[j])
            cand[i] = m
    def propagate(cand, g):
        changed = True
        while changed:
            changed = False
            for i in range(81):
                if g[i] == 0:
                    m = cand[i]
                    if m == 0: return False
                    if m & (m - 1) == 0:                       # single
                        v = m.bit_length() - 1; g[i] = v
                        for j in PEERS[i]:
                            if g[j] == 0 and cand[j] & m:
                                cand[j] &= ~m; changed = True
                                if cand[j] == 0: return False
        return True
    def rec(cand, g):
        if time.time() - t0 > tlim: raise Timeout
        if not propagate(cand, g): return
        best, bm = -1, 99
        for i in range(81):
            if g[i] == 0:
                k = bin(cand[i]).count("1")
                if k < bm: bm, best = k, i
        if best < 0:
            sols.append(g.copy()); return
        m = cand[best]
        while m:
            v = (m & -m).bit_length() - 1; m &= m - 1
            g2 = g.copy(); c2 = cand.copy(); g2[best] = 1 << 0  # placeholder
            g2[best] = v; c2[best] = 1 << v
            ok = True
            for j in PEERS[best]:
                if g2[j] == 0:
                    c2[j] &= ~(1 << v)
                    if c2[j] == 0: ok = False; break
            if ok: rec(c2, g2)
            if len(sols) > cap: return
    g0 = grid.copy()
    rec(cand, g0)
    return sols
# ---------------- 완화 퍼즐 만들기
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:NPZ].reshape(NPZ, 81).astype(int); Y = z["test_labels"][:NPZ].reshape(NPZ, 81).astype(int)
rng = np.random.default_rng(0)
relaxed = []   # (pid, Xr, sols)
t_all = time.time()
for pid in range(NPZ):
    giv = np.where(X[pid] > 0)[0]; order = rng.permutation(giv); accepted = None; last_small = None; note = ""
    for k in range(1, len(giv)):
        Xr = X[pid].copy(); Xr[order[:k]] = 0
        try: sols = enumerate_solutions(Xr.tolist(), CAP, TLIM)
        except Timeout: note = f"timeout at k={k}"; break
        n = len(sols)
        if LO <= n <= HI: accepted = (k, Xr, sols); break
        if n > HI: note = f"k={k} 에서 {n}+ (직전 k 는 {last_small})"; break
        last_small = n
    if accepted:
        k, Xr, sols = accepted; relaxed.append((pid, Xr, np.array(sols)))
        print(f"  퍼즐 {pid:3d}: k={k} 제거, 해 {len(sols)}개  ({time.time()-t_all:.0f}s)", flush=True)
    else:
        print(f"  퍼즐 {pid:3d}: 건너뜀 ({note})  ({time.time()-t_all:.0f}s)", flush=True)
print(f"완화 퍼즐 {len(relaxed)}개 확보", flush=True)
if len(relaxed) == 0: sys.exit("no puzzles")
# ---------------- MI 행렬
def mi_matrix(S):
    """S [n_sol, 81] (1..9). I[t,n], H[t]"""
    n = len(S); H = np.zeros(81); I = np.zeros((81, 81))
    oh = np.stack([(S == v) for v in range(1, 10)], -1).astype(np.float64)        # [n,81,9]
    p = oh.mean(0)                                                                  # [81,9]
    H = -(p * np.log(p + 1e-12)).sum(-1)
    J = np.einsum('sti,snj->tnij', oh, oh) / n                                      # [81,81,9,9]
    HJ = -(J * np.log(J + 1e-12)).sum((-1, -2))
    I = H[:, None] + H[None] - HJ
    np.fill_diagonal(I, 0.0); return I, H
MIs = []; Hs = []
for pid, Xr, S in relaxed:
    I, H = mi_matrix(S); MIs.append(I); Hs.append(H)
print("MI 계산 완료", flush=True)
# ---------------- 모델
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
B = len(relaxed); cfg = dict(ck["cfg"]); cfg.update(batch_size=B, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I_ = m.inner
Xb = np.stack([q[1] for q in relaxed]); Yb = np.stack([Y[q[0]] for q in relaxed])
x = torch.from_numpy(Xb.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Yb.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(B, dtype=torch.int32, device="cuda"))
rec = {}; preds = []; st = {"b": 0}; orig = I_.step
def hooked(L, h, AB, kc, w=None, fresh=None, kcb=None, vprev=None, apply_phi=True):
    b = st["b"]; st["b"] += 1
    hout, w_new = orig(L, h, AB, kc, w, fresh, kcb, vprev, apply_phi)
    preds.append((I_.w_cls(I_.phi(I_.boundary(L, hout))).argmax(-1) - 1).cpu().numpy())
    seg = b // 16 + 1
    if b % 16 == 15 and seg in REC_SEGS and L is I_.layers[1]:
        a = I_.attn_xy(I_.addr(h, AB), kc)
        lam = torch.sigmoid(L.lam_raw) if I_.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(I_.config.stdp_lam_fixed))
        a_eff = (1 - lam) * a + lam * w_new
        rec[seg] = dict(w=w_new.float().cpu().numpy(), a=a.float().cpu().numpy(), a_eff=a_eff.float().cpu().numpy())
        print(f"  기록: seg {seg}", flush=True)
    return hout, w_new
I_.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
t0 = time.time()
for si in range(SEGS):
    carry, _ = m(carry, batch)
    if (si + 1) % 4 == 0: print(f"  forward seg {si+1}/{SEGS} {time.time()-t0:.0f}s", flush=True)
P = np.stack(preds)                                                              # [blk,B,81]
# ---------------- 판정 지표
def rank(v): o = np.argsort(v); rk = np.empty(len(v)); rk[o] = np.arange(len(v)); return rk
def spearman(u, v):
    if len(u) < 3 or np.std(u) == 0 or np.std(v) == 0: return np.nan
    return np.corrcoef(rank(u), rank(v))[0, 1]
prng = np.random.default_rng(1)
res = {"n_puzzles": B, "per_puzzle": [], "seg": {}}
summ = {s: {} for s in REC_SEGS}
for s in REC_SEGS:
    W = rec[s]["w"]; A = rec[s]["a_eff"]; Ai = rec[s]["a"]
    acc = {k: [] for k in ["w_mean_all", "w_mean_peer", "w_mean_non", "w_max_all", "w_max_peer", "w_max_non",
                           "a_mean_all", "a_mean_peer", "a_mean_non", "ctrl_w_mean_all", "ctrl_w_mean_non", "ctrl_a_mean_non",
                           "w_low25", "w_high25", "a_low25", "a_high25", "asym_w_vs_H", "asym_a_vs_H", "ctrl_asym", "wsigned_peer", "wsigned_non"]}
    for i, (pid, Xr, S) in enumerate(relaxed):
        I = MIs[i]; H = Hs[i]; bl = Xr == 0
        pm = bl[:, None] & bl[None] & ~np.eye(81, dtype=bool)
        wm = np.abs(W[i].mean(0)); wx = np.abs(W[i]).max(0); am = np.abs(A[i].mean(0)); wsg = W[i].mean(0)
        # 대조군: 빈칸 인덱스 치환한 MI
        idx = np.where(bl)[0]; perm = prng.permutation(idx); Ic = I.copy(); Ic[np.ix_(idx, idx)] = I[np.ix_(perm, perm)]
        for nm_, msk in [("all", pm), ("peer", pm & peer_np), ("non", pm & ~peer_np)]:
            acc[f"w_mean_{nm_}"].append(spearman(wm[msk], I[msk])); acc[f"w_max_{nm_}"].append(spearman(wx[msk], I[msk])); acc[f"a_mean_{nm_}"].append(spearman(am[msk], I[msk]))
        acc["ctrl_w_mean_all"].append(spearman(wm[pm], Ic[pm])); acc["ctrl_w_mean_non"].append(spearman(wm[pm & ~peer_np], Ic[pm & ~peer_np])); acc["ctrl_a_mean_non"].append(spearman(am[pm & ~peer_np], Ic[pm & ~peer_np]))
        acc["wsigned_peer"].append(spearman(wsg[pm & peer_np], I[pm & peer_np])); acc["wsigned_non"].append(spearman(wsg[pm & ~peer_np], I[pm & ~peer_np]))
        Iv = I[pm]; q25, q75 = np.percentile(Iv, 25), np.percentile(Iv, 75)
        acc["w_low25"].append(wm[pm][Iv <= q25].mean()); acc["w_high25"].append(wm[pm][Iv >= q75].mean())
        acc["a_low25"].append(am[pm][Iv <= q25].mean()); acc["a_high25"].append(am[pm][Iv >= q75].mean())
        asw = (W[i].mean(0) - W[i].mean(0).T)[pm]; asa = (A[i].mean(0) - A[i].mean(0).T)[pm]; dH = (H[:, None] - H[None])[pm]
        acc["asym_w_vs_H"].append(spearman(asw, dH)); acc["asym_a_vs_H"].append(spearman(asa, dH))
        dHc = (H[perm.argsort()][:, None] if False else (H[:, None] - H[None]))[pm]; acc["ctrl_asym"].append(spearman(asw, prng.permutation(dH)))
    summ[s] = {k: float(np.nanmean(v)) for k, v in acc.items()}
    summ[s]["_std_w_mean_non"] = float(np.nanstd(acc["w_mean_non"])); summ[s]["_std_a_mean_non"] = float(np.nanstd(acc["a_mean_non"]))
    res["seg"][str(s)] = summ[s]
# ---- (f) 위치 통제: 각 쌍 위치의 퍼즐 간 MI 평균(위치 사전분포) vs 퍼즐 특이 잔차
Mstack = np.zeros((81, 81)); Cnt = np.zeros((81, 81))
for i, (pid, Xr, S) in enumerate(relaxed):
    bl = Xr == 0; pm = bl[:, None] & bl[None]; Mstack[pm] += MIs[i][pm]; Cnt[pm] += 1
Mpos = np.where(Cnt > 0, Mstack / np.maximum(Cnt, 1), 0.0)
for s in REC_SEGS:
    W = rec[s]["w"]; A = rec[s]["a_eff"]; f_pos_w = []; f_res_w = []; f_pos_a = []; f_res_a = []; f_res_w_peer = []
    for i, (pid, Xr, S) in enumerate(relaxed):
        I = MIs[i]; bl = Xr == 0; pm = bl[:, None] & bl[None] & ~np.eye(81, dtype=bool) & (Cnt >= 5)
        wm = np.abs(W[i].mean(0)); am = np.abs(A[i].mean(0)); resid = I - Mpos
        mk = pm & ~peer_np
        f_pos_w.append(spearman(wm[mk], Mpos[mk])); f_res_w.append(spearman(wm[mk], resid[mk]))
        f_pos_a.append(spearman(am[mk], Mpos[mk])); f_res_a.append(spearman(am[mk], resid[mk]))
        f_res_w_peer.append(spearman(wm[pm & peer_np], resid[pm & peer_np]))
    res["seg"][str(s)].update(pos_w_non=float(np.nanmean(f_pos_w)), resid_w_non=float(np.nanmean(f_res_w)), pos_a_non=float(np.nanmean(f_pos_a)), resid_a_non=float(np.nanmean(f_res_a)), resid_w_peer=float(np.nanmean(f_res_w_peer)))
print("\n(f) 위치 통제 (비동료 쌍): |w|/|a_eff| vs 쌍-위치별 MI 평균(위치 사전분포)  vs 퍼즐 특이 잔차 (I − 위치평균)")
print(f"{'seg':>3} | {'|w| vs 위치평균':>13} {'|w| vs 잔차':>10} | {'|a| vs 위치평균':>13} {'|a| vs 잔차':>10} | {'동료: |w| vs 잔차':>15}")
for s in REC_SEGS:
    q = res["seg"][str(s)]; print(f"{s:3d} | {q['pos_w_non']:13.3f} {q['resid_w_non']:10.3f} | {q['pos_a_non']:13.3f} {q['resid_a_non']:10.3f} | {q['resid_w_peer']:15.3f}")
# 모델이 해 하나에 앉는가
final = P[-1]; sit = []
for i, (pid, Xr, S) in enumerate(relaxed):
    f = final[i]; valid = all(len(set(f[np.where(r == u)[0]])) == 9 for u in range(9)) and all(len(set(f[np.where(c == u)[0]])) == 9 for u in range(9)) and all(len(set(f[np.where(bx == u)[0]])) == 9 for u in range(9)) and (f[Xr > 0] == Xr[Xr > 0]).all()
    in_set = bool(valid and (S == f[None]).all(-1).any())
    churn = float((P[-16:] [:, i] != P[-17:-1][:, i]).mean())
    viol = int(((f[:, None] == f[None]) & peer_np).sum() // 2)
    sit.append(dict(pid=int(pid), n_sol=int(len(S)), valid=bool(valid), in_solution_set=in_set, churn_last_seg=churn, violations=viol, wrong_vs_orig=int((f != Y[pid]).sum())))
res["sit"] = sit
res["sit_summary"] = dict(valid=float(np.mean([q["valid"] for q in sit])), in_set=float(np.mean([q["in_solution_set"] for q in sit])), churn=float(np.mean([q["churn_last_seg"] for q in sit])), mean_viol=float(np.mean([q["violations"] for q in sit])))
os.makedirs(os.path.join(ROOT, "2026-09-05/results/json"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "2026-09-05/results/json/mi_vs_w.json"), "w"), indent=1)
print("\n=== 결과 (퍼즐 평균 스피어만) ===")
print(f"{'seg':>3} | {'|w| vs I 전체':>12} {'동료':>7} {'비동료':>7} | {'|w|max 비동료':>12} | {'|a_eff| 전체':>11} {'동료':>7} {'비동료':>7} | {'대조 w전체':>9} {'대조 w비동료':>10} {'대조 a비동료':>10}")
for s in REC_SEGS:
    q = summ[s]; print(f"{s:3d} | {q['w_mean_all']:12.3f} {q['w_mean_peer']:7.3f} {q['w_mean_non']:7.3f} | {q['w_max_non']:12.3f} | {q['a_mean_all']:11.3f} {q['a_mean_peer']:7.3f} {q['a_mean_non']:7.3f} | {q['ctrl_w_mean_all']:9.3f} {q['ctrl_w_mean_non']:10.3f} {q['ctrl_a_mean_non']:10.3f}")
print(f"\n{'seg':>3} | {'|w| I하위25%':>11} {'I상위25%':>9} | {'|a| I하위25%':>11} {'I상위25%':>9} | {'비대칭 w vs ΔH':>13} {'a vs ΔH':>8} {'대조':>6} | {'부호w 동료':>9} {'부호w 비동료':>10}")
for s in REC_SEGS:
    q = summ[s]; print(f"{s:3d} | {q['w_low25']:11.4f} {q['w_high25']:9.4f} | {q['a_low25']:11.4f} {q['a_high25']:9.4f} | {q['asym_w_vs_H']:13.3f} {q['asym_a_vs_H']:8.3f} {q['ctrl_asym']:6.3f} | {q['wsigned_peer']:9.3f} {q['wsigned_non']:10.3f}")
ss = res["sit_summary"]; print(f"\n모델 끝 상태: 유효 격자 {100*ss['valid']:.0f}%, 열거된 해 집합 안 {100*ss['in_set']:.0f}%, 마지막 세그 churn {ss['churn']:.4f}, 평균 위반 {ss['mean_viol']:.1f}")
