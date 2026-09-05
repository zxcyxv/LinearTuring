"""182k(이전 규칙, 덧셈 읽기) 체크포인트의 세그먼트별 결합 크기 포화: Σ_n|a_eff| 행합, w rms. 퍼즐 64개, 64세그."""
import os, numpy as np, torch
os.environ["SEGS"] = "64"
src = open("/workspace/LinearTuring/2026-09-05/analysis/winfer_exact.py").read().split('if __name__ == "__main__":')[0]
exec(src)
for li, L in enumerate(layers):
    print(f"L{li}: δ={torch.sigmoid(L.eta_raw).mean():.4f}  λ={torch.sigmoid(L.lam_raw).mean():.3f}  시정수 1/δ = {1/torch.sigmoid(L.eta_raw).mean():.0f} 블록 ≈ {1/torch.sigmoid(L.eta_raw).mean()/8:.1f} 세그")
nb = 64; x = torch.from_numpy(X[:nb].astype(np.int32)+1).cuda(); y = torch.from_numpy(Y[:nb].astype(np.int32)+1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(nb, dtype=torch.int32, device="cuda"))
eye = torch.eye(81, device="cuda").bool()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
    carry = m.initial_carry(batch)
    print(f"{'seg':>4} | {'L1 Σ|a| 행합':>12} {'L1 Σ|a_eff| 행합':>16} {'w rms':>6} | {'L0 Σ|a_eff|':>10}")
    for s in range(1, 65):
        carry, out = m(carry, batch); h = carry.current_hidden; w = carry.coupling.float()
        row = []
        for li in (1, 0):
            L = layers[li]; a = I.attn(h, I.W_C(L), I.kernel(L)).float(); lam = torch.sigmoid(L.lam_raw).float()
            ae = (1 - lam) * a + lam * w
            row.append((a.abs().masked_fill(eye, 0).sum(-1).mean().item(), ae.abs().masked_fill(eye, 0).sum(-1).mean().item()))
        if s <= 16 or s % 8 == 0:
            print(f"{s:>4} | {row[0][0]:12.3f} {row[0][1]:16.3f} {w.pow(2).mean().sqrt():6.3f} | {row[1][1]:10.3f}", flush=True)
