"""추론 중 w 갱신: 매 블록, 다음 블록 출력의 위반 비용 C_viol 의 w 에 대한 1스텝 절단 경사로 w 를 내린다 (정답 미사용).
C_viol = 평균_동료쌍 ⟨p_t, p_n⟩, p = softmax(숫자 1..9 로짓). 유효 격자에서는 ≈0 → 갱신 ≈0.
갱신: w ← w − η · g · rms(w)/rms(g)  (w 크기 대비 상대 스텝). η=0 이 기준선. 진행 로그 매 배치."""
import os, importlib.util, time, numpy as np, torch
from dataclasses import replace
ROOT = "/workspace/LinearTuring"; N = int(os.environ.get("N", 512)); SEGS = int(os.environ.get("SEGS", 128)); BS = 256
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
    p = torch.softmax(I.w_cls(h)[..., 2:11].float(), -1)               # [B,T,9]
    return torch.einsum('btk,bnk,tn->b', p, p, peer) / peer.sum()        # [B]
def run(eta):
    ex = np.zeros(SEGS); vc = np.zeros(SEGS); t0 = time.time(); solved_end = np.zeros(N, bool)
    for b in range(0, N, BS):
        x = torch.from_numpy(X[b:b+BS].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y[b:b+BS].astype(np.int32) + 1).cuda().long(); nb = len(x)
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(nb, dtype=torch.int32, device="cuda"))
        inj = I.injection(batch) * I.embed_scale
        h = I.init_hidden.expand(nb, 81, -1).clone(); w = None; fresh = torch.ones(nb, dtype=torch.bool, device="cuda")
        for s in range(SEGS):
          for _blk in range(I.config.blocks_per_seg):
            for L in I.layers:
                AB = I.W_C(L); kc = I.kernel(L)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if eta > 0 and w is not None:
                        w_in = w.detach().requires_grad_(True)
                        hout, w_new = I.step(L, h + inj, AB, kc, w_in, fresh, None, apply_phi=False)
                        h_end = I.phi(I.boundary(L, hout)); cost = viol_cost(h_end).sum()
                        g, = torch.autograd.grad(cost, w_in)
                        with torch.no_grad():
                            step = eta * g * (w_new.float().pow(2).mean().sqrt() / (g.float().pow(2).mean().sqrt() + 1e-12))
                            w = (w_new.detach() - step.to(w_new.dtype)); h = h_end.detach()
                    else:
                        with torch.no_grad():
                            hout, w_new = I.step(L, h + inj, AB, kc, w, fresh, None, apply_phi=False)
                            h = I.phi(I.boundary(L, hout)); w = w_new
                fresh = None
          with torch.no_grad():
              pred = I.w_cls(h).argmax(-1); ok = (pred == y).all(-1)
              ex[s] += ok.sum().item(); vc[s] += viol_cost(h).sum().item()
        solved_end[b:b+nb] = ok.cpu().numpy()
        print(f"  η={eta:<5} batch {b//BS+1}/{-(-N//BS)}  {time.time()-t0:.0f}s  seg16 누적 {int(ex[15])}  seg{SEGS} 누적 {int(ex[-1])}", flush=True)
    return ex, vc / N, solved_end
res = {}
for eta in [float(v) for v in os.environ.get("ETAS", "0,0.03,0.1,0.3").split(",")]:
    res[eta] = run(eta)
print(f"\n{'η':>6} | " + " ".join(f"seg{s:>3}" for s in (16, 32, 64, SEGS)) + " |  위반비용 seg16 / seg끝  | 기준선 대비 (얻음/잃음)")
base = res[0.0]
for eta, (ex, vc, se) in res.items():
    gain = int((se & ~base[2]).sum()); loss = int((~se & base[2]).sum())
    print(f"{eta:6.2f} | " + " ".join(f"{int(ex[s-1]):6d}" for s in (16, 32, 64, SEGS)) + f" |  {vc[15]:.4f} / {vc[-1]:.4f}  | +{gain} / −{loss}", flush=True)
