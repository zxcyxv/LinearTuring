"""두 레이어가 같은 w 에 쓰는 목표(Γ)가 서로 얼마나 다른가. 실제 상태 h 위에서 잰다."""
import os, sys, importlib.util, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
with torch.device("cuda"): carry = m.initial_carry(batch)
L0, L1 = I.layers
AB = [I.W_C(L) for L in I.layers]; kc = [I.kernel(L) for L in I.layers]
def corr(u, v): u = u.flatten() - u.mean(); v = v.flatten() - v.mean(); return (u @ v / (u.norm() * v.norm() + 1e-9)).item()
print("seg | corr(a0,a1) corr(Γ0,Γ1) | |a0| |a1| |Γ0| |Γ1| | |w| corr(w,Γ0) corr(w,Γ1) | corr(a0,w)")
for s in range(16):
    h = carry.current_hidden + I.embed_scale * I.injection(batch)   # 블록 입력 (세그 첫 블록)
    out = []
    for li, L in enumerate(I.layers):
        a = I.attn(h, AB[li], kc[li]); v = torch.einsum('btd,hcd->bthc', h, L.w_sh); vv = v / (v.norm(dim=-1, keepdim=True) + 1e-4)
        G = a * torch.einsum('bthc,bnhc->bhtn', vv, vv); out.append((a, torch.nn.functional.softplus(L.gain_raw) * G))
    (a0, g0), (a1, g1) = out; w = carry.coupling
    row = f"{s+1:3d} | {corr(a0,a1):+.3f} {corr(g0,g1):+.3f} | {a0.abs().mean():.3f} {a1.abs().mean():.3f} {g0.abs().mean():.3f} {g1.abs().mean():.3f} | "
    row += (f"{w.abs().mean():.3f} {corr(w,g0):+.3f} {corr(w,g1):+.3f} | {corr(a0,w):+.3f}" if w is not None else "  (w 없음)")
    print(row)
    carry, _ = m(carry, batch)
for li, L in enumerate(I.layers):
    lam = torch.sigmoid(L.lam_raw).flatten(); G = torch.nn.functional.softplus(L.gain_raw).flatten()
    print(f"layer{li}: λ 헤드별 {[round(v,2) for v in lam.tolist()]}  G {[round(v,2) for v in G.tolist()]}  이득밴드 (1−λ)+λG·agree ∈ [{((1-lam)-lam*G).min():.2f}, {((1-lam)+lam*G).max():.2f}]")
