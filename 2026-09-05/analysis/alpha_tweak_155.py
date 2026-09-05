"""#155 원본에서 거리 감쇠 α 를 조금 줄여(D→1 방향) 추론. α ← s·α, s 를 1 근처에서 스윕. 두 레이어 전 헤드 동일 배율."""
import os, importlib.util, math, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; SEGS = 64
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
x = torch.from_numpy(X0[None].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y0[None].astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(1, dtype=torch.int32, device="cuda"))
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
inv_softplus = lambda v: math.log(math.expm1(v))
print(f"{'배율 s':>6} | {'풀림 seg':>8} {'끝 틀림':>7} {'끝 위반':>7} | 세그 4/8/16/32/64 끝 틀린 칸 | 원본과 다른 칸 (seg1 끝)")
ref1 = None
for s in [1.0, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8, 0.7, 1.02, 1.05, 1.1]:
    m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
    for L in I.layers:
        alpha = torch.nn.functional.softplus(L.alpha_raw) * s
        L.alpha_raw.data = torch.log(torch.expm1(alpha.clamp_min(1e-6)))
    with torch.device("cuda"): carry = m.initial_carry(batch)
    first = -1; hist = {}
    for si in range(SEGS):
        carry, o = m(carry, batch); p = o["logits"].argmax(-1)[0].cpu().numpy() - 1
        if first < 0 and (p == Y0).all(): first = si + 1
        if si + 1 in (1, 4, 8, 16, 32, 64): hist[si + 1] = p.copy()
    if s == 1.0: ref1 = hist[1]
    pend = hist[64]; viol = int(((pend[:, None] == pend[None]) & peer).sum() // 2)
    print(f"{s:6.2f} | {str(first) if first > 0 else '–':>8} {int((pend != Y0).sum()):7d} {viol:7d} | " + "/".join(f"{int((hist[k] != Y0).sum()):2d}" for k in (4, 8, 16, 32, 64)) + f"   {int((hist[1] != ref1).sum())}", flush=True)
    del m; torch.cuda.empty_cache()
