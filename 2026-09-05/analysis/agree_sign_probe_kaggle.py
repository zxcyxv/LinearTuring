"""캐글 체크포인트(step_182450)에서 agree = ⟨v̂_t,v̂_n⟩ 부호. 레이어별로 따로 기록 (블록 k → layers[k%2])."""
import os, importlib.util, numpy as np, torch
ROOT = "/workspace/LinearTuring"
spec = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)
torch.set_grad_enabled(False)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=256, seq_len=81, num_puzzle_identifiers=1, loops=17)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
x = torch.from_numpy(z["test_inputs"][:256].reshape(-1, 81).astype(np.int32) + 1).cuda()
y = torch.from_numpy(z["test_labels"][:256].reshape(-1, 81).astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(256, dtype=torch.int32, device="cuda"))
r = torch.arange(81, device="cuda") // 9; c = torch.arange(81, device="cuda") % 9; b = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (b[:, None] == b[None])) & ~torch.eye(81, dtype=torch.bool, device="cuda")
offd = ~torch.eye(81, dtype=torch.bool, device="cuda"); same_digit = (y[:, :, None] == y[:, None, :]) & offd
rec = {0: [], 1: []}
orig = I.step
def hooked(L, h, AB, kc, *a_, **k_):
    li = 0 if L is I.layers[0] else 1
    xy = I.addr(h, AB); a = I.attn_xy(xy, kc)
    v = torch.einsum('btd,hcd->bthc', h, L.w_sh); vv = v / (v.norm(dim=-1, keepdim=True) + I.config.eps)
    ag = torch.einsum('bthc,bnhc->bhtn', vv, vv); G = a * ag
    P = peer[None, None].expand_as(ag); O = offd[None, None].expand_as(ag); S = same_digit[:, None].expand_as(ag)
    rec[li].append(dict(neg_all=(ag[O] < 0).float().mean().item(), neg_peer=(ag[P] < 0).float().mean().item(),
        neg_same=(ag[S] < 0).float().mean().item(), mean_peer=ag[P].mean().item(), mean_same=ag[S].mean().item(),
        flip_mass=(G.abs() * (ag < 0)).sum().item() / G.abs().sum().item(),
        a_peer=a[P].mean().item(), G_peer=G[P].mean().item(), a_neg_peer=(a[P] < 0).float().mean().item()))
    return orig(L, h, AB, kc, *a_, **k_)
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for s in range(16): carry, _ = m(carry, batch)
keys = ["neg_all", "neg_peer", "neg_same", "mean_peer", "mean_same", "flip_mass", "a_peer", "G_peer", "a_neg_peer"]
for li in (0, 1):
    R = rec[li]; n = len(R); print(f"\n--- layer{li}  (블록 {n}개)  δ={torch.sigmoid(I.layers[li].eta_raw).mean():.3f} λ={torch.sigmoid(I.layers[li].lam_raw).mean():.3f}")
    print(f"{'블록':>8} | " + " ".join(f"{k:>10}" for k in keys))
    for lab, idx in [("seg1 blk1", [0]), ("seg1 last", [7]), ("seg2 blk1", [8]), ("seg8", range(56, 64)), ("seg16", range(120, 128)), ("전체", range(n))]:
        print(f"{lab:>8} | " + " ".join(f"{np.mean([R[i][k] for i in idx]):10.3f}" for k in keys))
