"""비판 검증: 곱셈 읽기 a·exp(w) 에서 반발(a<0) 쌍의 배율이 <1 로 약해지는가. 로컬 1000스텝 체크포인트, 실제 퍼즐 64개, 16세그먼트."""
import importlib.util, torch, numpy as np, os
ROOT="/workspace/LinearTuring"; spec = importlib.util.spec_from_file_location("tk", f"{ROOT}/kaggle/train_kaggle.py"); tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)
ck = torch.load("/tmp/claude-0/-workspace/08bc710b-d194-40cc-8cfb-cb09bdc9e744/scratchpad/v2smoke/step_1000.pt", map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=64, seq_len=81, num_puzzle_identifiers=1, loops=17, compile=False)
m = tk.LT(cfg).cuda().eval(); sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}; m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(f"{ROOT}/kaggle/upload/sudoku_lt_1k.npz"); X = torch.from_numpy(z["test_inputs"][:64].reshape(64,81).astype(np.int32)+1).cuda(); Y = torch.from_numpy(z["test_labels"][:64].reshape(64,81).astype(np.int64)+1).cuda()
batch = dict(inputs=X, labels=Y, puzzle_identifiers=torch.zeros(64, dtype=torch.int32, device="cuda"))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
    carry = m.initial_carry(batch)
    for s in range(16): carry, out = m(carry, batch)
    h = carry.current_hidden; w = carry.coupling.float()
    L = I.layers[1]; a = I.attn(h, I.W_C(L), I.kernel(L)).float()          # 마지막 상태의 순간 커널 (L1)
eye = torch.eye(81, device="cuda").bool()
g = torch.exp((torch.sign(a) * w).clamp(-4, 4)); neg = (a < 0) & ~eye; pos = (a > 0) & ~eye
print(f"L1 기준, 16세그 뒤: a<0 쌍 비율 {neg.float().mean():.2f}")
print(f"  a>0 쌍 배율 exp(w): 평균 {g[pos].mean():.2f}  (>1 인 비율 {(g[pos]>1).float().mean():.2f})")
print(f"  a<0 쌍 배율 exp(w): 평균 {g[neg].mean():.2f}  (<1 인 비율 {(g[neg]<1).float().mean():.2f})")
ae = a * g; print(f"  |a_eff| 합: 인력 {ae[pos].sum():.1f}  반발 {ae[neg].abs().sum():.1f}   (원래 a: 인력 {a[pos].sum():.1f}  반발 {a[neg].abs().sum():.1f})")
