"""w 가 인스턴스 정보를 담는가: 추론에서 w 를 (a) 배치 평균으로 공유, (b) 다른 퍼즐의 w 로 교체, (c) 원본. 256퍼즐 64세그."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); N, SEGS = 256, 64
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=N, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:N].reshape(N, 81).astype(int); Y = z["test_labels"][:N].reshape(N, 81).astype(int)
x = torch.from_numpy(X.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device="cuda"))
def run(mode):
    m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner; orig = I.step
    def hooked(L, h, AB, kc, w=None, fresh=None, kcb=None, vprev=None, apply_phi=True):
        if w is not None:
            if mode == "배치평균 공유": w = w.mean(0, keepdim=True).expand_as(w).contiguous()
            elif mode == "다른 퍼즐 w": w = torch.roll(w, 1, 0)
        return orig(L, h, AB, kc, w, fresh, kcb, vprev, apply_phi)
    I.step = hooked
    with torch.device("cuda"): carry = m.initial_carry(batch)
    ex = {}; t0 = time.time()
    for si in range(SEGS):
        carry, o = m(carry, batch); p = o["logits"].argmax(-1).cpu().numpy() - 1
        if si + 1 in (16, 32, 64): ex[si + 1] = int((p == Y).all(-1).sum())
    acc16 = None
    print(f"  {mode:>10}: 완답 seg16 {ex[16]:3d}  seg32 {ex[32]:3d}  seg64 {ex[64]:3d}  ({time.time()-t0:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache(); return ex
print("256퍼즐, w 처리 방식별 완답 수")
res = {mode: run(mode) for mode in ["원본", "배치평균 공유", "다른 퍼즐 w"]}
