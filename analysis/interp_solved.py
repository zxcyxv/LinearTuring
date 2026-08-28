"""완답 퍼즐에 대한 해석 — 칸별 커밋 시점(전파 깊이 클래스별) · 소거(엔트로피) 곡선 · 헤드별 피어 정렬(억제/흥분).
사용: python interp_solved.py [--ckpt PATH --bilinear 0|1] [--n 128] [--out NPZ]
원 결과: results/json/interp_solved_B16_19530.npz (SwiGLU B16 판, 완답 128 표본)."""
import argparse, numpy as np, torch
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--n", type=int, default=128); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
inp, lab, depth = load_test(); N = len(inp)
L, K = m.config.loops, m.config.blocks_per_seg; TS = L * K
peer = torch.tensor(peer_mask(), device="cuda")

# 1패스: 완답 퍼즐 식별
solved = []
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N))
    h = rollout(m, make_batch(inp[b], lab[b]))
    solved += (i + torch.where((logits(m, h).argmax(-1).to(torch.int32) == lab[b]).all(1))[0].cpu()).tolist()
print(f"완답 {len(solved)}/{N}")
S = np.array(solved[:args.n]); P = len(S)

# 2패스: 완답 표본 수동 전개 — 스텝별 argmax·엔트로피·헤드 피어질량
am = np.zeros((P, TS, 81), np.int8); en = np.zeros((P, TS, 81), np.float16)
head_peer = torch.zeros(TS, m.config.num_heads, device="cuda"); head_neg = torch.zeros_like(head_peer)
for i in range(0, P, args.bs):
    idx = torch.tensor(S[i:i + args.bs], device="cuda"); n = len(idx)
    def hook(loop, blk, stage, h, a, i=i, n=n):
        if stage != "post_step": return
        t = loop * K + blk
        aa = a.abs(); head_peer[t] += aa[:, :, peer].sum((0, 2)) / (aa.sum((0, 2, 3)) + 1e-9) * n / P
        head_neg[t] += a.clamp(max=0).abs()[:, :, peer].sum((0, 2)) / (aa[:, :, peer].sum((0, 2)) + 1e-9) * n / P
        lg = logits(m, h)[..., 1:]; pr = lg.softmax(-1)
        am[i:i + n, t] = (lg.argmax(-1) + 1).to(torch.int8).cpu().numpy()
        en[i:i + n, t] = (-(pr * (pr + 1e-9).log()).sum(-1)).to(torch.float16).cpu().numpy()
    rollout(m, make_batch(inp[idx], lab[idx]), hook=hook)

fin = lab.cpu().numpy()[S]; giv = inp.cpu().numpy()[S] > 1; dep = depth[S]
correct = am == fin[:, None, :]
commit = TS - np.cumprod(correct[:, ::-1, :], 1).astype(bool).sum(1)     # 이 스텝부터 끝까지 정답 유지
classes = [("주어짐", giv), ("전파 1-2", (dep >= 1) & (dep < 3)), ("전파 3-5", (dep >= 3) & (dep < 6)),
           ("전파 6+", dep >= 6), ("탐색", dep == -1)]
print(f"\n── A. 커밋 시점 ({TS} 스텝 중) ──\n{'클래스':10s} {'중앙값':>6s} {'평균':>7s} {'90%ile':>7s}")
for nm, msk in classes:
    v = commit[msk]; print(f"{nm:10s} {np.median(v):6.0f} {v.mean():7.1f} {np.percentile(v, 90):7.0f}")
print(f"전파확정 칸 corr(전파깊이, 커밋시점) = {np.corrcoef(dep[dep > 0].astype(float), commit[dep > 0])[0, 1]:.3f}")
loop_end = en[:, K - 1::K, :]
print("\n── B. 엔트로피 (loop 말, nats) ──\nloop:      " + " ".join(f"{i:5d}" for i in [0, 3, 7, 11, 15]))
for nm, msk in classes:
    e = (loop_end * msk[:, None, :]).sum((0, 2)) / (msk.sum() + 1e-9)
    print(f"{nm:10s} " + " ".join(f"{e[i]:5.2f}" for i in [0, 3, 7, 11, 15]))
hp, hn = head_peer.mean(0).cpu().numpy(), head_neg.mean(0).cpu().numpy()
print("\n── C. 헤드별 |a| 질량의 피어 비중 [무작위 기대 0.25] / 피어 중 음수 비중 ──")
print("피어비중:   " + " ".join(f"{v:6.3f}" for v in hp)); print("피어중음수: " + " ".join(f"{v:6.3f}" for v in hn))
if args.out:
    np.savez(args.out, commit=commit, dep=dep, giv=giv, head_peer=head_peer.cpu().numpy(),
             head_negpeer=head_neg.cpu().numpy(), solved_idx=S)
