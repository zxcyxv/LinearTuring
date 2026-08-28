"""은닉상태 이식: 어려운 퍼즐(원 입력)에서 최종 오답인 빈칸 t 를 고르고, 같은 퍼즐을 '80칸 주어짐·t 만 빈칸' 으로 돌려 얻은 h_t* 를
원 실행의 최종 상태에 꽂아 넣은 뒤 --extra loop 더 실행. 측정: t 가 정답 유지되는 비율, 판 전체 정답 칸 수 변화, 완답 전환.
대조군: init_hidden 으로 교체 / 다른 퍼즐의 h* 로 교체 / 교체 없음(연장만).
사용: python transplant.py [--n 512 --extra 8] [--out JSON]"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--n", type=int, default=512); ap.add_argument("--extra", type=int, default=8); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.set_grad_enabled(False); rng = np.random.default_rng(args.seed)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
inner = m.inner; core = inner.core; K = m.config.blocks_per_seg; L = m.config.loops
inp, lab, _ = load_test(args.n); N = len(inp); I = inp.cpu().numpy(); LB = lab.cpu().numpy()
fc = core.kernel_fast(); AB = core.W_C(); dt = 1.0 / core.R

def run(h, x, loops):
    inj = inner._injection(make_batch(x, x))
    for _ in range(loops):
        for _ in range(K):
            h = inner._boundary(h) + inner.inj_gate * inj
            for _ in range(core.R):
                hh = core.phi(h, dt / 2); f, *_ = core.field(hh, None, None, None, AB, fast_ctx=fc); h = core.phi(hh + dt * f, dt / 2)
    return h
def board(h): return (logits(m, h)[:, :, 2:11].argmax(-1) + 2).cpu().numpy()

# 1) 원 실행
Hf = torch.zeros(N, 81, core.d, device="cuda")
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N)); Hf[b] = run(inner.init_hidden.expand(b.stop - b.start, 81, -1).clone(), inp[b], L)
B0 = board(Hf); ok0 = B0 == LB; blank = I == 1
sel = np.where((~ok0 & blank).any(1))[0]                                    # 오답 빈칸이 있는 퍼즐
T = np.array([rng.choice(np.where(~ok0[i] & blank[i])[0]) for i in sel]); n = len(sel)
print(f"오답 있는 퍼즐 {n}/{N}, 기본 완답 {ok0.all(1).sum()}")
# 2) 80칸 주어짐 실행 → h_t*
X1 = LB[sel].copy(); X1[np.arange(n), T] = 1; x1 = torch.tensor(X1, dtype=torch.int32, device="cuda")
Hstar = torch.zeros(n, core.d, device="cuda")
for i in range(0, n, args.bs):
    b = slice(i, min(i + args.bs, n)); h = run(inner.init_hidden.expand(b.stop - b.start, 81, -1).clone(), x1[b], L)
    Hstar[b] = h[torch.arange(b.stop - b.start), torch.tensor(T[b], device="cuda")]
# 3) 이식 변형
idx = torch.tensor(sel, device="cuda"); tt = torch.tensor(T, device="cuda"); ar = torch.arange(n, device="cuda")
perm = torch.tensor(rng.permutation(n), device="cuda")
variants = {"none(연장만)": None, "h*_own": Hstar, "h*_other_puzzle": Hstar[perm], "init_hidden": inner.init_hidden.expand(n, -1).clone(),
            "h*_own_x10norm_match": None}
res = {"n_puzzles": n, "extra_loops": args.extra, "base_cells_ok_mean": float(ok0[sel][blank[sel]].mean()), "base_exact": int(ok0[sel].all(1).sum())}
for nm, rep in variants.items():
    if nm.startswith("h*_own_x10"): continue
    h = Hf[idx].clone()
    if rep is not None: h[ar, tt] = rep
    out = np.zeros((n, 81), np.int64)
    for i in range(0, n, args.bs):
        b = slice(i, min(i + args.bs, n)); out[b] = board(run(h[b], inp[idx[b]], args.extra))
    ok = out == LB[sel]
    res[nm] = {"t_correct_frac": float(ok[np.arange(n), T].mean()), "cells_ok_mean": float(ok[blank[sel]].mean()),
               "exact": int(ok.all(1).sum()), "cells_gained_mean": float((ok.sum(1) - ok0[sel].sum(1)).mean())}
    # 이식 직후(연장 0) t 판독
    if rep is not None:
        hb = Hf[idx].clone(); hb[ar, tt] = rep
        res[nm]["t_correct_at_insert"] = float((board(hb)[np.arange(n), T] == LB[sel][np.arange(n), T]).mean())
    print(nm, json.dumps(res[nm]), flush=True)
print(json.dumps(res, indent=1, ensure_ascii=False))
if args.out: json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
