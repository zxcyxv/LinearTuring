"""추론 중 w 갱신, 창 단위: K 블록 창의 끝 위반비용을 창 시작 w 에 대해 미분(창 안 K블록 절단 BPTT), 창마다 1회 갱신.
K=1 이면 앞 실험과 동일. 갱신: w ← w − η·g·rms(w)/rms(g)."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; N = int(os.environ.get("N", 512)); SEGS = int(os.environ.get("SEGS", 128)); BS = 256
K = int(os.environ.get("K", 16))
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=BS, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
for p in m.parameters(): p.requires_grad_(False)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:N].reshape(N, 81).astype(int); Y = z["test_labels"][:N].reshape(N, 81).astype(int)
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = torch.from_numpy(((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)).cuda().float()
def viol_cost(h):
    p = torch.softmax(I.w_cls(h)[..., 2:11].float(), -1); return torch.einsum('btk,bnk,tn->b', p, p, peer) / peer.sum()
layers = list(I.layers); nl = len(layers)
def run(eta):
    ex = np.zeros(SEGS); vc = np.zeros(SEGS); t0 = time.time(); solved_end = np.zeros(N, bool)
    for b in range(0, N, BS):
        x = torch.from_numpy(X[b:b+BS].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y[b:b+BS].astype(np.int32) + 1).cuda().long(); nb = len(x)
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(nb, dtype=torch.int32, device="cuda"))
        inj = I.injection(batch) * I.embed_scale
        h = I.init_hidden.expand(nb, 81, -1).clone(); w = None; fresh = torch.ones(nb, dtype=torch.bool, device="cuda")
        ABk = [(I.W_C(L), I.kernel(L)) for L in layers]
        total_blocks = SEGS * I.config.blocks_per_seg * nl; blk = 0
        while blk < total_blocks:
            # 창: K 블록. 첫 창(w None)은 갱신 없이 통과
            do_grad = eta > 0 and w is not None
            w_in = w.detach().requires_grad_(True) if do_grad else w
            hh = h; ww = w_in
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(do_grad):
                for j in range(K):
                    L = layers[(blk + j) % nl]; AB, kc = ABk[(blk + j) % nl]
                    hout, ww = I.step(L, hh + inj, AB, kc, ww, fresh, None, apply_phi=False); hh = I.phi(I.boundary(L, hout)); fresh = None
                    if (blk + j + 1) % (I.config.blocks_per_seg * nl) == 0:
                        s = (blk + j + 1) // (I.config.blocks_per_seg * nl) - 1
                        with torch.no_grad():
                            pred = I.w_cls(hh).argmax(-1); ok = (pred == y).all(-1); ex[s] += ok.sum().item(); vc[s] += viol_cost(hh).sum().item()
                if do_grad:
                    cost = viol_cost(hh).sum(); g, = torch.autograd.grad(cost, w_in)
            with torch.no_grad():
                if do_grad:
                    step = eta * g * (ww.float().pow(2).mean().sqrt() / (g.float().pow(2).mean().sqrt() + 1e-12)); w = ww.detach() - step.to(ww.dtype)
                else: w = ww.detach() if ww is not None else None
                h = hh.detach()
            blk += K
        solved_end[b:b+nb] = ok.cpu().numpy()
        print(f"  K={K} η={eta:<5} batch {b//BS+1}/{-(-N//BS)}  {time.time()-t0:.0f}s  seg16 누적 {int(ex[15])}  seg{SEGS} 누적 {int(ex[-1])}", flush=True)
    return ex, vc / N, solved_end
etas = [float(v) for v in os.environ.get("ETAS", "0,0.01,0.03").split(",")]
BASE_F = os.path.join(ROOT, f"2026-09-05/results/json/wdescent_baseline_N{N}_S{SEGS}.npz")
if os.path.exists(BASE_F):                       # 기준선은 결정론적 — 한 번 저장해 두고 재사용
    bz = np.load(BASE_F); base = (bz["ex"], bz["vc"], bz["solved_end"]); print(f"기준선 로드: {BASE_F} (seg{SEGS} {int(base[0][-1])})", flush=True)
else:
    base = run(0.0); np.savez(BASE_F, ex=base[0], vc=base[1], solved_end=base[2]); print(f"기준선 저장: {BASE_F}", flush=True)
res = {0.0: base}
for eta in etas:
    if eta > 0: res[eta] = run(eta)
print(f"\nK={K}  {'η':>6} | " + " ".join(f"seg{s:>3}" for s in (16, 32, 64, SEGS)) + " |  위반비용 seg16 / seg끝  | 기준선 대비 (얻음/잃음)")
for eta, (ex, vc, se) in res.items():
    print(f"      {eta:6.3f} | " + " ".join(f"{int(ex[s-1]):6d}" for s in (16, 32, 64, SEGS)) + f" |  {vc[15]:.4f} / {vc[-1]:.4f}  | +{int((se & ~base[2]).sum())} / −{int((~se & base[2]).sum())}", flush=True)
