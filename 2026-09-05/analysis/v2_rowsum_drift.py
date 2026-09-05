"""비판 (c) 검증: a_eff 행합·메시지 크기가 세그먼트에 따라 커지는가(비정상성), 어느 세그먼트에서 포화하는가. v2.1 로컬 1000스텝 ckpt, 퍼즐 64개, 64세그."""
import importlib.util, torch, numpy as np
ROOT="/workspace/LinearTuring"; spec = importlib.util.spec_from_file_location("tk", f"{ROOT}/kaggle/train_kaggle.py"); tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)
ck = torch.load("/tmp/claude-0/-workspace/08bc710b-d194-40cc-8cfb-cb09bdc9e744/scratchpad/v2smoke/step_1000.pt", map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=64, seq_len=81, num_puzzle_identifiers=1, loops=65, compile=False)
m = tk.LT(cfg).cuda().eval(); sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}; m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(f"{ROOT}/kaggle/upload/sudoku_lt_1k.npz"); X = torch.from_numpy(z["test_inputs"][:64].reshape(64,81).astype(np.int32)+1).cuda(); Y = torch.from_numpy(z["test_labels"][:64].reshape(64,81).astype(np.int64)+1).cuda()
batch = dict(inputs=X, labels=Y, puzzle_identifiers=torch.zeros(64, dtype=torch.int32, device="cuda"))
print("δ(헤드별): L0", [round(v,3) for v in torch.sigmoid(I.layers[0].eta_raw).flatten().tolist()], " L1", [round(v,3) for v in torch.sigmoid(I.layers[1].eta_raw).flatten().tolist()])
eye = torch.eye(81, device="cuda").bool()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
    carry = m.initial_carry(batch)
    print(f"{'seg':>4} | {'Σ_n|a| 행합':>10} {'Σ_n|a_eff| 행합':>14} {'평균 배율':>8} {'‖f‖/‖h‖':>8} {'w rms':>6}")
    for s in range(1, 65):
        carry, out = m(carry, batch)
        h = carry.current_hidden; w = carry.coupling.float(); L = I.layers[1]
        AB, kc = I.W_C(L), I.kernel(L); a = I.attn(h, AB, kc).float()
        g = torch.exp((torch.sign(a) * w).clamp(-4, 4)); ae = a * g
        v = torch.einsum('btd,hcd->bthc', h.float(), L.w_sh.float()); o = torch.einsum('bhtn,bnhc->bthc', ae, v); f = torch.einsum('bthc,hcd->btd', o, L.w_sh.float())
        if s <= 16 or s % 8 == 0:
            print(f"{s:>4} | {a.abs().masked_fill(eye,0).sum(-1).mean():10.3f} {ae.abs().masked_fill(eye,0).sum(-1).mean():14.3f} {g.masked_fill(eye,1).mean():8.3f} {(f.norm(dim=-1)/h.float().norm(dim=-1)).mean():8.3f} {w.pow(2).mean().sqrt():6.3f}", flush=True)
