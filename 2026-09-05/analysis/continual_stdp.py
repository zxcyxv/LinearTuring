"""샘플 간 지속 STDP: 공유 시냅스 S 가 테스트 스트림을 따라 모델의 Γ 로 계속 갱신, 결합에 κ·S 추가. 비용·라벨 없음.
512퍼즐을 64개 청크 8개로 순서대로. 청크별 seg128 완답을 κ=0 과 비교."""
import os, math, importlib.util, time, numpy as np, torch, torch.nn.functional as F
ROOT = "/workspace/LinearTuring"; N = 512; CH = 64; SEGS = int(os.environ.get("SEGS", 128)); DS = float(os.environ.get("DS", 1e-3))
src = open(os.path.join(ROOT, "2026-09-05/analysis/winfer_exact.py")).read().split('if __name__ == "__main__":')[0]
src = src.replace('N = int(os.environ.get("N", 512))', 'N = 512').replace("BS = 256", "BS = 64")
exec(src)
def step_gamma(L, h, AB, kc, w, fresh, u):
    """step_u 와 동일하되 Γ(tgt) 도 반환."""
    xy = I.addr(h, AB); a = I.attn_xy(xy, kc); v = torch.einsum('btd,hcd->bthc', h, L.w_sh)
    eta_ = torch.sigmoid(L.eta_raw); lam = torch.sigmoid(L.lam_raw) if I.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(I.config.stdp_lam_fixed))
    vv = v / (v.norm(dim=-1, keepdim=True) + I.config.eps); agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
    gain = F.softplus(L.gain_raw) if I.config.stdp_gain_fixed < 0 else float(I.config.stdp_gain_fixed); tgt = gain * a * agree
    if w is None: w = tgt
    else:
        w = torch.where(fresh.view(-1, 1, 1, 1), tgt, w) if fresh is not None else w
        w = (1 - eta_) * w + eta_ * tgt
    a_eff = (1 - lam) * a + lam * w
    if u is not None: a_eff = a_eff + u
    o = torch.einsum('bhtn,bnhc->bthc', a_eff, v); f = torch.einsum('bthc,hcd->btd', o, L.w_sh)
    return h + f, w, tgt
def stream(kappa):
    S = [torch.zeros(I.H, 81, 81, device="cuda") for _ in layers]                     # 레이어별 공유 시냅스
    per_chunk = []; t0 = time.time()
    for b in range(0, N, CH):
        x = torch.from_numpy(X[b:b+CH].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y[b:b+CH].astype(np.int32) + 1).cuda().long(); nb = len(x)
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(nb, dtype=torch.int32, device="cuda"))
        inj = I.injection(batch) * I.embed_scale; ABk = [(I.W_C(L), I.kernel(L)) for L in layers]
        h = I.init_hidden.expand(nb, 81, -1).clone(); w = None; fresh = torch.ones(nb, dtype=torch.bool, device="cuda"); ex = np.zeros(SEGS)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(SEGS):
                for _ in range(I.config.blocks_per_seg):
                    for li, L in enumerate(layers):
                        AB, kc = ABk[li]; u = (kappa * S[li]).unsqueeze(0).expand(nb, -1, -1, -1) if kappa > 0 else None
                        hout, w, tgt = step_gamma(L, h + inj, AB, kc, w, fresh, u); h = I.phi(I.boundary(L, hout)); fresh = None
                        S[li] = S[li] + DS * (tgt.float().mean(0) - S[li])                    # 샘플 간 지속 STDP (배치 평균 Γ)
                pred = I.w_cls(h).argmax(-1); ex[s] = (pred == y).all(-1).sum().item()
        per_chunk.append(ex)
        print(f"  κ={kappa:<4} chunk {b//CH+1}/{N//CH}  {time.time()-t0:.0f}s  seg16 {int(ex[15])}/{nb}  seg{SEGS} {int(ex[-1])}/{nb}   |S| rms L1 {S[1].pow(2).mean().sqrt().item():.4f}", flush=True)
    return np.array(per_chunk)
res = {k: stream(k) for k in [float(v) for v in os.environ.get("KAPPAS", "0,0.3,1.0").split(",")]}
print(f"\n청크별 seg{SEGS} 완답 (64개 중), δ_s={DS}")
print(f"{'κ':>5} | " + " ".join(f"c{i+1:>3}" for i in range(N // CH)) + " | 합계 | 전반4 | 후반4")
for k, R in res.items():
    e = R[:, -1]; print(f"{k:5.1f} | " + " ".join(f"{int(v):4d}" for v in e) + f" | {int(e.sum()):4d} | {int(e[:4].sum()):4d} | {int(e[4:].sum()):4d}", flush=True)
