"""퍼즐 #155 FAIL vs SOLVED: 블록마다 상태 속도 ‖Δh‖, 로짓 마진(틀린 칸/맞은 칸), 규칙 위반 수, 조기 확정 여부."""
import os, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; S = 16
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=2, seq_len=81, num_puzzle_identifiers=1, loops=S + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
dm = np.array([0, 1, 3, 8, 2, 4, 7, 5, 6, 9])          # 앞서 고른 치환 #3
Xb = np.stack([X0, np.where(X0 > 0, dm[X0], 0)]); Yb = np.stack([Y0, dm[Y0]])
x = torch.from_numpy(Xb.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Yb.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(2, dtype=torch.int32, device="cuda"))
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
rec = []; orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    h_end = I.phi(I.boundary(L, hout))                                   # 블록 끝 상태 (post 순서)
    logit = I.w_cls(h_end).float(); top2 = logit.topk(2, -1).values; margin = (top2[..., 0] - top2[..., 1]).cpu().numpy()
    pred = logit.argmax(-1).cpu().numpy() - 1
    vel = (h_end - h).norm(dim=-1).mean(-1).cpu().numpy() / np.sqrt(832)  # 칸당 상태 변화, √d 로 정규화
    rec.append(dict(pred=pred, margin=margin, vel=vel))
    return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(S): carry, _ = m(carry, batch)
nb = len(rec); P = np.stack([q["pred"] for q in rec]); M = np.stack([q["margin"] for q in rec]); V = np.stack([q["vel"] for q in rec])
blank = Xb == 0; wrong = P != Yb[None]
viol = np.array([[((P[b, k][:, None] == P[b, k][None]) & peer).sum() // 2 for k in range(2)] for b in range(nb)])
print(f"{'block':>5} {'seg':>3} | {'FAIL 틀림':>8} {'위반':>4} {'속도':>6} {'마진(틀린)':>9} {'마진(맞은)':>9} | {'SOLVED 틀림':>10} {'위반':>4} {'속도':>6} {'마진(틀린)':>9} {'마진(맞은)':>9}")
for b in [0, 1, 2, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 159, 191, 207, 215, 223, 239, 255]:
    row = f"{b:5d} {b//16+1:3d} |"
    for k in range(2):
        w = wrong[b, k] & blank[k]; ok = ~wrong[b, k] & blank[k]
        row += f" {int(wrong[b,k].sum()):8d} {viol[b,k]:4d} {V[b,k]:6.3f} {M[b,k][w].mean() if w.any() else float('nan'):9.2f} {M[b,k][ok].mean() if ok.any() else float('nan'):9.2f} |"
    print(row)
# 조기 확정: FAIL 의 끝 틀린 칸 집합이 언제 형성됐나
final_wrong = wrong[-1, 0]
for b in [3, 7, 15, 31, 63]:
    print(f"FAIL: block {b:3d} 시점 틀린 칸 중 끝까지 틀린 것 {int((wrong[b,0] & final_wrong).sum())}/{int(wrong[b,0].sum())},  끝 틀린 칸 37개 중 이미 그 값이었던 것 {int(((P[b,0]==P[-1,0]) & final_wrong).sum())}")
# SOLVED: 언제부터 정답 칸이 늘기 시작했나 — 세그별 맞은 칸 수
print("SOLVED 세그별 끝 블록 맞은 칸 수:", [int((~wrong[16*s_+15, 1]).sum()) for s_ in range(S)])
print("FAIL   세그별 끝 블록 맞은 칸 수:", [int((~wrong[16*s_+15, 0]).sum()) for s_ in range(S)])
