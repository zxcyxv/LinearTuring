"""거리 감쇠 D(Δ) 를 거리 무관 상수로 교체. (a) D≡1  (b) D≡헤드별 비대각 평균  (c) D≡헤드별 동료쌍 평균.
#155 원본 + 앞 256퍼즐 완답(맥락)."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; SEGS = 64; N = 256
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:N].reshape(N, 81).astype(int); Y = z["test_labels"][:N].reshape(N, 81).astype(int)
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
peer_t = torch.from_numpy(peer).cuda(); offd = ~torch.eye(81, dtype=torch.bool, device="cuda")
def make(mode):
    cfg = dict(ck["cfg"]); cfg.update(batch_size=N, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
    m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner; orig_kernel = I.kernel
    def kernel(L, psi=None):
        decay, cA, sA, cB, sB = orig_kernel(L, psi)                       # decay [H,T,T]
        if mode == "원본": return decay, cA, sA, cB, sB
        if mode == "D≡1": d2 = torch.ones_like(decay)
        elif mode == "D≡비대각평균": d2 = decay[:, offd].mean(-1).view(-1, 1, 1).expand_as(decay).clone()
        elif mode == "D≡동료평균": d2 = decay[:, peer_t].mean(-1).view(-1, 1, 1).expand_as(decay).clone()
        return d2, cA, sA, cB, sB
    I.kernel = kernel; return m
x = torch.from_numpy(X.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device="cuda"))
print(f"{'모드':>10} | {'#155 풀림':>9} {'#155 끝 틀림/위반':>14} | {'256퍼즐 완답 seg16':>16} {'seg64':>6} | #155 seg1끝 원본과 다른 칸")
ref1 = None
for mode in ["원본", "D≡1", "D≡비대각평균", "D≡동료평균"]:
    m = make(mode); t0 = time.time()
    with torch.device("cuda"): carry = m.initial_carry(batch)
    first = -1; ex16 = ex64 = 0; p1 = None
    for si in range(SEGS):
        carry, o = m(carry, batch); p = o["logits"].argmax(-1).cpu().numpy() - 1
        if si == 0: p1 = p[PID].copy()
        if first < 0 and (p[PID] == Y[PID]).all(): first = si + 1
        if si + 1 == 16: ex16 = int((p == Y).all(-1).sum())
    ex64 = int((p == Y).all(-1).sum()); pend = p[PID]; viol = int(((pend[:, None] == pend[None]) & peer).sum() // 2)
    if mode == "원본": ref1 = p1
    print(f"{mode:>10} | {str(first) if first > 0 else '–':>9} {int((pend != Y[PID]).sum()):6d} / {viol:5d} | {ex16:16d} {ex64:6d} | {int((p1 != ref1).sum())}   ({time.time()-t0:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache()
