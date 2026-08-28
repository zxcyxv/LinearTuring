"""칸 하나의 출처별 로짓 분해 — C_tn = w_cls·(a_tn W_mᵀW_m h_n) (헤드 합). 조건부 선형성 덕에 근사 없음.
세 모드:
  --mode single   : 정답 격자에서 --k 칸만 비운 뒤 첫 빈칸의 분해를 한 퍼즐 그대로 출력 (실험 1)
  --mode hidden   : 실제 퍼즐의 깊이-1 칸을 naked-single / hidden-only 로 나눠, 블록별 정답률과
                    출처 4군(피어·주어짐 / 피어·빈칸 / 비피어·주어짐 / 비피어·빈칸)의 정답 자리 여유를 추적
  --mode sample   : --puzzle --cell 의 최종 상태 분해를 20 피어 한 줄씩 출력 (오답 칸 진단용)
사용: python cell_decompose.py --mode {single,hidden,sample} [--n 256] [--k 1] [--puzzle 1 --cell 67]"""
import argparse, numpy as np, torch
from common import load_lt, load_test, make_batch, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--mode", default="hidden", choices=["single", "hidden", "sample"])
ap.add_argument("--n", type=int, default=256); ap.add_argument("--k", type=int, default=1); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--puzzle", type=int, default=0); ap.add_argument("--cell", type=int, default=-1); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.set_grad_enabled(False); rng = np.random.default_rng(args.seed)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs); inner = m.inner; core = inner.core
K = m.config.blocks_per_seg; L = m.config.loops; TS = L * K
inp, lab, depth = load_test(args.n); I = inp.cpu().numpy(); LB = lab.cpu().numpy(); N = len(LB); g = LB - 2   # 숫자 0..8
pm = peer_mask(); r_ = np.arange(81) // 9; c_ = np.arange(81) % 9; b_ = (r_ // 3) * 3 + c_ // 3
fc = core.kernel_fast(); AB = core.W_C(); Wsh = core.w_sh.float(); Wc = core.w_cls.weight.float()[2:11]; bc = core.w_cls.bias.float()[2:11]
DIG = "  ".join(f"{d:>5d}" for d in range(1, 10))

def step(h, inj):
    h = inner._boundary(h) + inner.inj_gate * inj; hh = core.phi(h, .5)
    f, a, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc); return core.phi(hh + f, .5), a, hh
def decomp(hh, a, i, t):                                                   # [81,9]
    v = torch.einsum('nd,hcd->hnc', hh[i], Wsh); return (torch.einsum('hn,hnc,hcd->nd', a[i, :, t], v, Wsh) @ Wc.T).cpu().numpy()
def naked(x, i, t): return set(range(9)) - set(g[i, [n for n in np.where(pm[t])[0] if x[i, n] > 1]])
def hidden_only(x, i, t):
    for u in (r_ == r_[t], c_ == c_[t], b_ == b_[t]):
        if all(g[i, t] not in naked(x, i, n) for n in np.where(u)[0] if x[i, n] == 1 and n != t): return True
    return False
def print_table(C, i, t, x, board=None):
    peers = pm[t]; non = ~peers; non[t] = False
    print(f"  {'피어':8s}{'값':>4s}{'정답':>4s}   " + DIG)
    for n in np.where(peers)[0]:
        val = (board[n] if board is not None else g[i, n]) + 1; mark = " *" if board is not None and board[n] != g[i, n] else "  "
        print(f"  r{n//9}c{n%9}   {val:>3d} {g[i,n]+1:>3d}{mark} " + "  ".join(f"{C[n,d]:5.1f}" for d in range(9)))
    Sp, Sn = C[peers].sum(0), C[non].sum(0)
    print("  피어 합      " + "  ".join(f"{v:5.1f}" for v in Sp)); print("  비피어 합    " + "  ".join(f"{v:5.1f}" for v in Sn))
    print("  전체         " + "  ".join(f"{v:5.1f}" for v in Sp + Sn))
    return Sp, Sn

if args.mode == "single":
    X = LB.copy(); T = rng.choice(81, args.k, replace=False); X[:, T] = 1; x = torch.tensor(X, dtype=torch.int32, device="cuda"); t = int(T[0])
    ok = 0
    for i0 in range(0, N, args.bs):
        b = slice(i0, min(i0 + args.bs, N)); n = b.stop - b.start; inj = inner._injection(make_batch(x[b], x[b])); h = inner.init_hidden.expand(n, 81, -1).clone()
        for s in range(TS): h, a, hh = step(h, inj)
        P = logits(m, h)[:, :, 2:11].argmax(-1).cpu().numpy(); ok += (P[:, T] == g[b][:, T]).all(1).sum()
        if i0 == 0:
            i = args.puzzle; C = decomp(hh, a, i, t)
            print(f"퍼즐 {i}, 빈칸 {[f'r{u//9}c{u%9}' for u in T]}, 표시 칸 t=r{t//9}c{t%9} 정답 {g[i,t]+1}, 모델 {P[i,t]+1}")
            for rr in range(9): print("  " + " ".join('.' if rr * 9 + cc in T else str(g[i, rr*9+cc] + 1) for cc in range(9)))
            print(f"  최종 로짓: {np.round((Wc @ hh[i, t] + bc).cpu().numpy(), 1)}"); print_table(C, i, t, X)
    print(f"\n{N} 퍼즐 빈칸 {args.k}개 전부 정답인 비율: {ok/N:.4f}")

elif args.mode == "hidden":
    TG = {"hidden-only (naked 후보 ≥2)": [(i, t) for i in range(N) for t in range(81) if depth[i, t] == 1 and len(naked(I, i, t)) >= 2 and hidden_only(I, i, t)],
          "naked-single": [(i, t) for i in range(N) for t in range(81) if depth[i, t] == 1 and len(naked(I, i, t)) == 1]}
    STEPS = list(range(8)) + [15, 31, TS - 1]
    for label, targets in TG.items():
        need = {}; [need.setdefault(i, []).append(t) for i, t in targets]; out = {s: [] for s in STEPS}
        for i0 in range(0, N, args.bs):
            b = slice(i0, min(i0 + args.bs, N)); n = b.stop - b.start; inj = inner._injection(make_batch(inp[b], inp[b])); h = inner.init_hidden.expand(n, 81, -1).clone()
            for s in range(TS):
                h, a, hh = step(h, inj)
                if s not in STEPS: continue
                P = logits(m, h)[:, :, 2:11].argmax(-1).cpu().numpy()
                for i in range(i0, b.stop):
                    for t in need.get(i, []):
                        C = decomp(hh, a, i - i0, t); cand = naked(I, i, t); ans = g[i, t]; alt = [c for c in cand if c != ans] or [c for c in range(9) if c != ans]
                        peers = pm[t]; giv = I[i] > 1
                        grp = [peers & giv, peers & ~giv, ~peers & giv, ~peers & ~giv]; grp[3][t] = False
                        out[s].append((P[i - i0, t] == ans,) + tuple(C[msk].sum(0)[ans] - max(C[msk].sum(0)[c] for c in alt) for msk in grp))
        print(f"\n[{label}] {len(targets)}칸 — 블록별 정답률 | 정답자리−최대타후보 여유: 평균 (양수비율)")
        print(f"{'블록':>4s} {'정답':>5s} | {'피어·주어짐':>16s} {'피어·빈칸':>16s} {'비피어·주어짐':>16s} {'비피어·빈칸':>16s}")
        for s in STEPS:
            r = np.array(out[s], float); print(f"{s:>4d} {r[:,0].mean():5.3f} | " + " ".join(f"{r[:,k].mean():+7.2f} ({(r[:,k]>0).mean():.2f})" for k in range(1, 5)))

else:  # sample: 실제 퍼즐, 최종 상태에서 칸 하나 분해
    i = args.puzzle; b = slice(i, i + 1); inj = inner._injection(make_batch(inp[b], inp[b])); h = inner.init_hidden.expand(1, 81, -1).clone()
    for s in range(TS): h, a, hh = step(h, inj)
    board = logits(m, h)[0, :, 2:11].argmax(-1).cpu().numpy(); wrong = np.where((board != g[i]) & (I[i] == 1))[0]
    t = args.cell if args.cell >= 0 else (int(wrong[0]) if len(wrong) else 0)
    print(f"퍼즐 {i}: 틀린 빈칸 {len(wrong)}개. 칸 t=r{t//9}c{t%9}: 모델 {board[t]+1}, 정답 {g[i,t]+1}   (* = 오답 피어)")
    for rr in range(9): print("  " + " ".join((str(board[u]+1) if I[i,u] > 1 or board[u] == g[i,u] else f"({board[u]+1}/{g[i,u]+1})") for u in range(rr*9, rr*9+9)))
    print(f"  최종 로짓: {np.round((Wc @ hh[0, t] + bc).cpu().numpy(), 1)}"); print_table(decomp(hh, a, 0, t), i, t, I, board)
