"""256퍼즐: 원래 모델의 2블록 주기 사상 F 를 400회 반복했을 때 상태 잔차 ‖F(s)−s‖/‖s‖ — 해결/미해결별 고정점 존재 여부."""
import os, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); N, SEG = 256, 8
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=N, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1, amp=False)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:N].reshape(N, 81).astype(int); Y = z["test_labels"][:N].reshape(N, 81).astype(int)
x = torch.from_numpy(X.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device="cuda"))
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(SEG): carry, _ = m(carry, batch)
s = carry.current_hidden.clone(); w = carry.coupling.clone(); inj = I.injection(batch) * I.embed_scale
def F(s):
    h = s
    for L in I.layers:
        AB = I.W_C(L); kc = I.kernel(L); h = h + inj
        hout, _ = I.step(L, h, AB, kc, w, None, None, apply_phi=False); h = I.phi(I.boundary(L, hout))
    return h
for k in range(400):
    s_new = F(s); res = (s_new - s).flatten(1).norm(dim=1) / s.flatten(1).norm(dim=1); s = s_new
pred = I.w_cls(s).argmax(-1) - 1; solved = (pred.cpu().numpy() == Y).all(-1); res = res.cpu().numpy()
print(f"400주기 뒤 상태 잔차 (w 고정, 2블록 주기 사상)")
for name, sel in (("해결", solved), ("미해결", ~solved)):
    r = res[sel]; print(f"  {name} {sel.sum():3d}개: 잔차 중앙값 {np.median(r):.1e}  <1e-4 인 비율 {100*(r<1e-4).mean():.0f}%  <1e-2 {100*(r<1e-2).mean():.0f}%  최대 {r.max():.1e}")
