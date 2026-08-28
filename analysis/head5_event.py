"""특정 헤드의 '해결 상태 게이트' 가설 검증 — 피어→빈칸 유입 |a| 을 송신 칸의 확정 여부로 나눠 loop 별로 비교.
+ 샘플 0 초점 칸에 대한 그 헤드 유입 시계열 그림.
사용: python head5_event.py [--ckpt PATH --bilinear 0|1] [--head 5 --ref 0] [--n 128] [--out PNG]
원 결과: results/figs/head5_timecourse_S1.png (SwiGLU B16 판: head5 미확정/확정 피어 유입 3.4배, 전 loop 일관)."""
import argparse, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--head", type=int, default=5); ap.add_argument("--ref", type=int, default=0)
ap.add_argument("--n", type=int, default=128); ap.add_argument("--out", default=f"{ROOT}/results/figs/head_timecourse.png")
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=128)
inp, lab, _ = load_test(); L, K = m.config.loops, m.config.blocks_per_seg
S = []
for i in range(0, len(inp), 128):
    b = slice(i, i + 128); h = rollout(m, make_batch(inp[b], lab[b]))
    S += (i + torch.where((logits(m, h).argmax(-1).to(torch.int32) == lab[b]).all(1))[0].cpu()).tolist()
    if len(S) >= args.n: break
S = S[:args.n]; B = len(S); idx = torch.tensor(S, device="cuda")
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=B)
AH = np.zeros((L, B, 81, 81), np.float16); AR = np.zeros_like(AH); am = np.zeros((B, L, 81), np.int8)
def hook(loop, blk, stage, h, a):
    if stage == "post_step" and blk == K - 1:
        AH[loop] = a[:, args.head].cpu().numpy(); AR[loop] = a[:, args.ref].cpu().numpy()
        am[:, loop] = logits(m, h).argmax(-1).to(torch.int8).cpu().numpy()
rollout(m, make_batch(inp[idx], lab[idx]), hook=hook)
lab_n, inp_n = lab[idx].cpu().numpy(), inp[idx].cpu().numpy()
commit = L - np.cumprod((am == lab_n[:, None, :])[:, ::-1, :], 1).astype(bool).sum(1)
pm = peer_mask()

# 모집단: 피어→빈칸 유입, 송신자 미확정 vs 확정
print(f"피어→빈칸 유입 (완답 {B}): head{args.head} vs head{args.ref}, 송신자 미확정/확정")
print(f"{'loop':>4} | h{args.head} |a| 미확정/확정 | h{args.head} 부호평균 | h{args.ref} |a| 미확정/확정")
for loop in [0, 3, 7, 11, 15]:
    unres = commit[:, None, :] > loop; rx = (inp_n == 1)[:, :, None]
    mu, mr = rx & pm[None] & unres, rx & pm[None] & ~unres
    print(f"{loop:>4} |  {np.abs(AH[loop])[mu].mean():.4f} / {np.abs(AH[loop])[mr].mean():.4f}  |  "
          f"{AH[loop][mu].mean():+.4f} / {AH[loop][mr].mean():+.4f}  |  {np.abs(AR[loop])[mu].mean():.4f} / {np.abs(AR[loop])[mr].mean():.4f}")

# 샘플 0 초점 칸 시계열
p = 0; focus = int(np.argmax(commit[p] * (inp_n[p] == 1)))
top = np.argsort(-(np.abs(AH[min(7, L - 1), p, focus]) - np.abs(AH[L - 1, p, focus])))[:3]
fig, ax = plt.subplots(figsize=(7, 4))
for n in top:
    ax.plot(range(L), AH[:, p, focus, n], marker='o', ms=3, label=f"from r{n // 9}c{n % 9} (commit L{commit[p, n]})")
ax.plot(range(L), AH[:, p, focus, focus], 'k--', lw=1, label="self")
ax.axvline(commit[p, focus], color='r', ls=':', label=f"focus commits (L{commit[p, focus]})"); ax.axhline(0, color='gray', lw=.5)
ax.set_xlabel("loop"); ax.set_ylabel(f"a_head{args.head}[focus <- n]"); ax.legend(fontsize=8)
ax.set_title(f"sample {S[p]}: head-{args.head} inflow to focus r{focus // 9}c{focus % 9}")
fig.tight_layout(); fig.savefig(args.out, dpi=130); print("→", args.out)
