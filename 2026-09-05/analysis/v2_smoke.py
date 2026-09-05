import importlib.util, torch
spec = importlib.util.spec_from_file_location("tk", "/workspace/LinearTuring/kaggle/train_kaggle.py"); tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)
cfg = dict(tk.CFG); cfg.update(batch_size=8, seq_len=81, num_puzzle_identifiers=1, loops=3)
torch.manual_seed(0); m = tk.LT(cfg).cuda().eval(); sd = m.state_dict()
x = torch.randint(1, 11, (8, 81), dtype=torch.int32).cuda(); y = torch.randint(1, 11, (8, 81)).cuda()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(8, dtype=torch.int32).cuda())
def run(model):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
        carry = model.initial_carry(batch); carry, out = model(carry, batch)
    return out["logits"].float(), carry
m0 = tk.LT({**cfg, "stdp": False}).cuda().eval(); m0.load_state_dict({k: v for k, v in sd.items() if k in m0.state_dict()}, strict=False)
m1 = tk.LT(cfg).cuda().eval(); m1.load_state_dict(sd)
for L in m1.inner.layers: L.eta_raw.data.fill_(-30.0)
(l0, _), (l1, _), (lm, carry) = run(m0), run(m1), run(m)
print(f"(1) δ→0 mul vs no-stdp  max|Δlogit| = {(l0-l1).abs().max().item():.3e}   (기본 δ mul vs no-stdp = {(l0-lm).abs().max().item():.3e}, 달라야 정상)")
w = carry.coupling.float(); print(f"(2) w rms {w.pow(2).mean().sqrt():.3f}  exp(w) min {w.exp().min():.2f} max {w.exp().max():.2f}   NaN {torch.isnan(w).any().item()}")
m.train()
with torch.device("cuda"): carry = m.initial_carry(batch)
with torch.autocast("cuda", dtype=torch.bfloat16):
    carry, out = m(carry, batch); loss = torch.nn.functional.cross_entropy(out["logits"].float().reshape(-1, out["logits"].shape[-1]), y.reshape(-1))
loss.backward(); g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
print(f"(3) loss {loss.item():.3f}  grad sum {g:.3e}  eta grad {[round(L.eta_raw.grad.abs().sum().item(),4) for L in m.inner.layers]}")
