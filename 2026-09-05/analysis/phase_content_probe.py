"""위상이 숫자(내용)를 담는가: 푼 격자의 최종 상태에서 주소 위상 φ_tj(위치 회전 제외)의
같은 숫자 칸끼리 집중도(진폭 가중 원형 평균 길이)를 무작위 라벨 섞기 기준과 비교. 행/열/상자 집중도도 같이 (위치 정보 대조)."""
import os, numpy as np, torch
os.environ.setdefault("SEGS", "16")
src = open("/workspace/LinearTuring/2026-09-05/analysis/winfer_exact.py").read().split('if __name__ == "__main__":')[0]
exec(src)
N_ = 256; SEG = 16; torch.manual_seed(0)
x = torch.from_numpy(X[:N_].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y[:N_].astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(N_, dtype=torch.int32, device="cuda"))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
    carry = m.initial_carry(batch)
    for s in range(SEG): carry, out = m(carry, batch)
h = carry.current_hidden.float(); pred = out["logits"].argmax(-1); solved = (pred == y).all(-1)
print(f"seg{SEG} 완답 {solved.sum().item()}/{N_} — 푼 샘플만 사용", flush=True)
hs, ys = h[solved], y[solved]; B = hs.shape[0]
pos = torch.arange(81, device="cuda"); groups = {"숫자": ys, "행": (pos // 9).expand(B, -1), "열": (pos % 9).expand(B, -1), "상자": ((pos // 27) * 3 + (pos % 9) // 3).expand(B, -1)}
def conc(z, g):
    """z [B,T,H,p] 복소 (진폭 포함), g [B,T] 정수 그룹 → 그룹 내 진폭가중 원형평균길이의 그룹 평균 [H,p]"""
    num = torch.zeros(z.shape[2], z.shape[3], dtype=torch.float32, device="cuda"); den = torch.zeros_like(num)
    for k in g.unique():
        mk = (g == k).float()[..., None, None]
        num += (z * mk).sum(1).abs().sum(0); den += (z.abs() * mk).sum(1).sum(0)
    return num / den
for li, L in enumerate(layers):
    AB = I.W_C(L)
    with torch.no_grad():
        xr, yr = I.addr(hs, AB)                                   # [B,T,H,p] 정규화 주소 (Σ_j r² = 1), 위치 회전 없음
    z = torch.complex(xr.float(), yr.float()); amp = z.abs().pow(2).mean((0, 1))   # [H,p] 성분별 에너지
    R_all = z.sum(1).abs().sum(0) / z.abs().sum(1).sum(0)          # 그룹 무시 집중도
    rows = []
    for name, g in groups.items():
        R = conc(z, g)
        gs = torch.stack([g[i][torch.randperm(81, device="cuda")] for i in range(B)])   # 라벨 섞기 기준
        R0 = conc(z, gs)
        gain = ((R - R0) * amp).sum() / amp.sum()
        frac = amp[(R - R0) > 0.1].sum() / amp.sum()
        rows.append((name, gain.item(), frac.item(), ((R0) * amp).sum().item() / amp.sum().item()))
    print(f"L{li}: 그룹무시 집중도(진폭가중) {(R_all*amp).sum()/amp.sum():.3f}")
    for name, gain, frac, base in rows:
        print(f"   {name:>3} 그룹 내 집중도 − 섞기 기준: {gain:+.3f}   (기준 {base:.3f})   초과>0.1 인 성분의 에너지 비율 {frac:.2f}", flush=True)
