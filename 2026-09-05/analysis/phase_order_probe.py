"""굳은 상태의 상태 위상 φ = arg(W_C h) (위치 항 제외) 가 칸의 확정 순서를 담는가.
확정 시각 c_t = 마지막으로 답이 바뀐 블록 + 1 (주어진 칸은 0). 마지막 블록 상태에서 φ 를 헤드·채널별로 꺼내
쌍 (t,n) 의 sin(φ_t − φ_n) 과 (c_t − c_n) 의 상관을 계산. 대조군 = c 를 퍼즐 안에서 섞음."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); N, SEGS = 256, 16
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=N, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:N].reshape(N, 81).astype(int); Y = z["test_labels"][:N].reshape(N, 81).astype(int)
x = torch.from_numpy(X.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device="cuda"))
preds = []; phases = {}; st = {"b": 0}; orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    b = st["b"]; st["b"] += 1
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    preds.append((I.w_cls(I.phi(I.boundary(L, hout))).argmax(-1) - 1).cpu().numpy())
    if b >= SEGS * 16 - 2:                                                     # 마지막 두 블록 (L0, L1)
        xx, yy = I.addr(h, AB); phases[0 if L is I.layers[0] else 1] = torch.atan2(yy, xx).float().cpu().numpy()   # [N,T,H,p]
    return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
t0 = time.time()
for si in range(SEGS): carry, _ = m(carry, batch)
print(f"forward {time.time()-t0:.0f}s", flush=True)
P = np.stack(preds); nb = len(P)
changed = (P[1:] != P[:-1])                                                   # [nb-1, N, 81]
last_change = np.where(changed.any(0), (np.arange(1, nb)[:, None, None] * changed).max(0), 0)   # 마지막 변경 블록
commit = last_change + 1; commit[X > 0] = 0                                      # 주어진 칸 0
solved = (P[-1] == Y).all(-1)
print(f"풀린 퍼즐 {solved.sum()}/{N}.  확정 시각 분포(풀린 퍼즐, 빈칸): 중앙값 {np.median(commit[solved][X[solved]==0]):.0f} 블록, 90% {np.percentile(commit[solved][X[solved]==0], 90):.0f}")
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
rng = np.random.default_rng(0)
def pair_corr(ph, com, sel_puz, pair_mask, shuffle=False):
    """ph [N,T,H,p], com [N,T]. 쌍별 sin(φ_t−φ_n) vs (c_t−c_n) 피어슨 상관, 헤드·채널별 → [H,p]"""
    out = []
    for k in np.where(sel_puz)[0]:
        cm = com[k].copy()
        if shuffle: bl = X[k] == 0; cm[bl] = rng.permutation(cm[bl])
        bb = (X[k] == 0)[:, None] & (X[k] == 0)[None]                                   # 빈칸끼리만
        pm2 = pair_mask & bb
        dc = (cm[:, None] - cm[None])[pm2]                                             # [pairs]
        if dc.std() == 0: continue
        dphi = np.sin(ph[k][:, None] - ph[k][None])[pm2]                             # [pairs,H,p]
        dphi = dphi - dphi.mean(0); dcz = (dc - dc.mean()) / dc.std()
        out.append((dphi * dcz[:, None, None]).mean(0) / (dphi.std(0) + 1e-9))
    return np.mean(out, 0)                                                            # [H,p]
blank_pairs = np.ones((81, 81), bool) & ~np.eye(81, dtype=bool)
for li in (0, 1):
    ph = phases[li]
    for name, pm in [("모든 쌍", blank_pairs), ("동료 쌍만", peer)]:
        cr = pair_corr(ph, commit, solved, pm); cn = pair_corr(ph, commit, solved, pm, shuffle=True)
        best = np.unravel_index(np.abs(cr).argmax(), cr.shape)
        print(f"L{li} {name}: |상관| 최대 {abs(cr[best]):.3f} (head {best[0]}, ch {best[1]}), 헤드별 최대 {np.round(np.abs(cr).max(1), 3).tolist()}   대조군(섞음) 최대 {np.abs(cn).max():.3f}")
print("\n칸 단위: 빈칸의 확정 시각 c_t 와 (cos φ, sin φ) 의 상관 — 헤드·채널 최대 (풀린 퍼즐 평균) / 대조군")
for li in (0, 1):
    ph = phases[li]; real = []; ctrl = []
    for k in np.where(solved)[0]:
        bl = X[k] == 0; cm = commit[k][bl].astype(float)
        if cm.std() == 0: continue
        cz = (cm - cm.mean()) / cm.std(); F = np.concatenate([np.cos(ph[k][bl]), np.sin(ph[k][bl])], -1)   # [nb, H, 2p]
        Fz = (F - F.mean(0)) / (F.std(0) + 1e-9); real.append((Fz * cz[:, None, None]).mean(0))
        cs = rng.permutation(cz); ctrl.append((Fz * cs[:, None, None]).mean(0))
    real = np.mean(real, 0); ctrl = np.mean(ctrl, 0)
    print(f"  L{li}: 실제 |상관| 최대 {np.abs(real).max():.3f}  헤드별 {np.round(np.abs(real).max(1), 3).tolist()}   대조군 최대 {np.abs(ctrl).max():.3f}")
# 순서와 위치의 상관 (혹시 확정 순서 자체가 위치를 따르나)
sel = solved; cc = []
for k in np.where(sel)[0]:
    bl = X[k] == 0; cc.append(np.corrcoef(commit[k][bl], (r + c)[bl])[0, 1])
print(f"참고: 확정 시각 vs 위치(행+열) 상관 평균 {np.nanmean(cc):+.3f}")
# 가장 좋은 채널로 '먼저 확정된 칸이 위상이 앞서는가' 방향 확인
li, (hh, jj) = 1, best
lead = []
for k in np.where(solved)[0]:
    bl = X[k] == 0; ph = phases[li][k][:, hh, jj]; cm = commit[k]
    pairs = [(t, n) for t in np.where(bl)[0] for n in np.where(bl)[0] if cm[t] < cm[n]]
    if pairs: lead.append(np.mean([np.sin(ph[t] - ph[n]) for t, n in pairs]))
print(f"L1 head {hh} ch {jj}: 먼저 확정된 칸 t 와 나중 칸 n 의 sin(φ_t−φ_n) 평균 {np.mean(lead):+.4f} (부호 일관성 {np.mean(np.sign(lead)==np.sign(np.mean(lead))):.2f})")
