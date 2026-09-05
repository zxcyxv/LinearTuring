"""sym 경계의 야코비안 대칭성 r=‖J−Jᵀ‖/‖J+Jᵀ‖ (칸 하나, 무작위 가중치·상태) — bilinear 와 비교. b_down 은 무작위로 채워서 검사."""
import importlib.util, torch
spec = importlib.util.spec_from_file_location("tk", "/workspace/LinearTuring/kaggle/train_kaggle.py"); tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)
torch.manual_seed(0)
for kind in ("bilinear", "sym"):
    cfg = dict(tk.CFG); cfg.update(batch_size=1, seq_len=81, num_puzzle_identifiers=1, loops=2, boundary=kind, hidden_size=256, mlp_expansion=2.0)
    m = tk.LT(cfg).cuda().float().eval(); I = m.inner; L = I.layers[0]
    with torch.no_grad(): L.b_down.weight.normal_(0, 0.05)
    h0 = torch.randn(256, device="cuda")
    f = lambda x: (I.boundary(L, x.view(1, 1, -1)) - x.view(1, 1, -1)).flatten()
    J = torch.autograd.functional.jacobian(f, h0)
    r = (J - J.T).norm() / (J + J.T).norm()
    print(f"{kind:>9}: r = {r:.4f}   ‖Δ‖ = {f(h0).norm():.3f}")
