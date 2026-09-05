"""초기 상태 민감도: 원본 #155 의 init_hidden 에 작은 잡음을 넣은 복제본 64개 × 잡음 크기 4단계. 성패와 궤적 분기 시점."""
import os, importlib.util, time, numpy as np, torch
from dataclasses import replace
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); PID = 155; SEGS = 64; R = 64
EPS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]; K = R * len(EPS)
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=K, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X0 = z["test_inputs"].reshape(-1, 81).astype(int)[PID]; Y0 = z["test_labels"].reshape(-1, 81).astype(int)[PID]
x = torch.from_numpy(np.repeat(X0[None], K, 0).astype(np.int32) + 1).cuda(); y = torch.from_numpy(np.repeat(Y0[None], K, 0).astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(K, dtype=torch.int32, device="cuda"))
# 첫 세그먼트에서 reset 되는 init_hidden 에 잡음을 얹기 위해 reset_carry 를 감싼다
torch.manual_seed(0); eps = torch.tensor(np.repeat(EPS, R), device="cuda", dtype=torch.float32)  # [K]
noise = torch.randn(K, 81, 832, device="cuda") * eps.view(-1, 1, 1) * np.sqrt(832) / np.sqrt(832)   # 성분당 표준편차 = eps (init_hidden 성분 std 1 대비)
orig_reset = I.reset_carry
def reset_noisy(flag, carry):
    c2 = orig_reset(flag, carry)
    h = c2.current_hidden + torch.where(flag.view(-1, 1, 1), noise.to(c2.current_hidden.dtype), torch.zeros_like(c2.current_hidden))
    return replace(c2, current_hidden=h)
I.reset_carry = reset_noisy
preds = []; orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    hout, w_new = orig(L, h, AB, kc, *a_, **k_)
    preds.append((I.w_cls(I.phi(I.boundary(L, hout))).argmax(-1) - 1).cpu().numpy()); return hout, w_new
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
t0 = time.time()
for si in range(SEGS):
    carry, _ = m(carry, batch)
    if (si + 1) % 16 == 0: print(f"  seg {si+1}/{SEGS} {time.time()-t0:.0f}s", flush=True)
P = np.stack(preds); solved = (P[-1] == Y0[None]).all(-1); ref = P[:, 0]        # eps=0 의 첫 복제 = 기준 궤적
print(f"\n{'eps':>6} | {'풀림':>6} | {'기준 궤적과 argmax 가 처음 달라지는 블록 (중앙값/최소)':>40} | {'블록 15 다른 칸 수':>14} {'블록 63 다른 칸':>12}")
for i, e in enumerate(EPS):
    sl = slice(i * R, (i + 1) * R); div = []
    for k in range(i * R, (i + 1) * R):
        d = np.where((P[:, k] != ref).any(-1))[0]; div.append(d[0] if len(d) else 9999)
    div = np.array(div); d15 = (P[15, sl] != ref[15][None]).sum(-1).mean(); d63 = (P[63, sl] != ref[63][None]).sum(-1).mean()
    print(f"{e:6.0e} | {solved[sl].sum():3d}/{R} | {np.median(div):8.0f} / {div.min():4d}{'':28s} | {d15:14.1f} {d63:12.1f}")
