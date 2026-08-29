"""퍼즐 몇 개를 그대로 본다 — 빈칸을 (확신/비확신)×(정답/오답) 으로 격자 위에 표시하고, 대표 칸의 장(어텐션 행·행합·진폭·커밋 궤적)을 출력.
확신 = 최종 로짓 여유(top1−top2) ≥ τ.  격자 기호:  숫자=주어짐  V=확신·정답  X=확신·오답  v=비확신·정답  x=비확신·오답
사용: python quadrant_view.py [--puzzles 0 1 2] [--tau 8] [--cells r3c4 ...]"""
import argparse, numpy as np, torch
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--puzzles", type=int, nargs="+", default=None)
ap.add_argument("--tau", type=float, default=8.0); ap.add_argument("--n_show", type=int, default=3)
ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--max_cells", type=int, default=4)
args = ap.parse_args(); torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=True, batch_size=64); inner = m.inner; core = inner.core
K = m.config.blocks_per_seg; L = m.config.loops; TS = L * K; H = core.H
inp, lab, depth = load_test(); I = inp.cpu().numpy(); G = lab.cpu().numpy() - 2; pm = peer_mask()
r_ = np.arange(81) // 9; c_ = np.arange(81) % 9; b_ = (r_ // 3) * 3 + c_ // 3
Wsh = core.w_sh.float(); Wc = core.w_cls.weight.float()[2:11]

def cname(t): return f"r{t//9}c{t%9}"
def parse(s): return int(s[1]) * 9 + int(s[3])

def trace(idx):
    """퍼즐 idx 들을 전개하며 스텝별 argmax·여유, 마지막 스텝의 a·h·d 를 기록"""
    x = inp[idx]; n = len(idx); rec = {"arg": np.zeros((n, TS, 81), np.int8), "mar": np.zeros((n, TS, 81), np.float32),
                                        "hn": np.zeros((n, TS, 81), np.float32), "da": np.zeros((n, TS, 81), np.float32)}
    last = {}
    def hook(loop, blk, stage, h, a):
        if stage != "post_step": return
        s = loop * K + blk; lg = logits(m, h)[:, :, 2:11].float(); sv = lg.sort(-1).values
        rec["arg"][:, s] = lg.argmax(-1).cpu().numpy(); rec["mar"][:, s] = (sv[..., -1] - sv[..., -2]).cpu().numpy()
        rec["hn"][:, s] = h.norm(dim=-1).cpu().numpy()
        if "a_prev" in last: rec["da"][:, s] = (a - last["a_prev"]).abs().mean(dim=(1, 3)).cpu().numpy()   # 행 t 의 간선 변화 (헤드 평균)
        last["a_prev"] = a.clone(); last["a"] = a; last["h"] = h; last["hh"] = core.phi(h, 0.5)
    rollout(m, make_batch(x, x), hook=hook)
    return rec, last

def quadrant(P, M, i):
    q = np.full(81, " ", dtype=object)
    for t in range(81):
        if I[i, t] != 1: q[t] = str(I[i, t] - 1); continue
        ok = P[t] == G[i, t]; conf = M[t] >= args.tau
        q[t] = ("V" if ok else "X") if conf else ("v" if ok else "x")
    return q

def show_cell(i, j, t, rec, last, P):
    a = last["a"][j, :, t].cpu().numpy(); hh = last["hh"][j]; d = a.sum(-1)
    arg = rec["arg"][j, :, t]; final = arg[-1]; first_commit = TS - 1
    while first_commit > 0 and arg[first_commit - 1] == final: first_commit -= 1
    flips = int((np.diff(arg) != 0).sum())
    C = torch.einsum('hn,hnc,hcd->nd', last["a"][j, :, t], torch.einsum('nd,hcd->hnc', hh, Wsh), Wsh) @ Wc.T; C = C.cpu().numpy()   # [81,9] 이웃별 로짓 기여
    print(f"\n  ── 칸 {cname(t)}: 모델 {P[t]+1} / 정답 {G[i,t]+1}  여유 {rec['mar'][j,-1,t]:.1f}  ‖h‖ {rec['hn'][j,-1,t]:.2f}  "
          f"커밋 스텝 {first_commit}/{TS}  뒤집힘 {flips}회  마지막8스텝 간선변화 {rec['da'][j,-8:,t].mean():.4f}")
    print(f"     argmax 궤적(8스텝 간격): {' '.join(str(arg[s]+1) for s in range(0, TS, 8))} | 여유: {' '.join(f'{rec['mar'][j,s,t]:.0f}' for s in range(0, TS, 8))}")
    print(f"     d_t 헤드별: {np.round(d, 2)}")
    peers = np.where(pm[t])[0]; same_non = [n for n in range(81) if not pm[t, n] and n != t and P[n] == P[t]]
    print(f"     피어 20칸 (칸:값[정답] | 헤드별 a | 로짓기여 모델답/정답):")
    for n in peers:
        val = (I[i, n] - 1) if I[i, n] != 1 else P[n] + 1; tag = "주" if I[i, n] != 1 else ("✓" if P[n] == G[i, n] else "✗")
        print(f"       {cname(n)} {val}{tag}[{G[i,n]+1}] | {' '.join(f'{v:+.2f}' for v in a[:, n])} | {C[n, P[t]]:+5.1f} {C[n, G[i,t]]:+5.1f}")
    print(f"     같은 숫자({P[t]+1}) 비피어 {len(same_non)}칸: " + "  ".join(f"{cname(n)}{'주' if I[i,n]!=1 else ''}(Σa {a[:, n].sum():+.2f}, C {C[n, P[t]]:+.1f})" for n in same_non[:8]))
    print(f"     기여 합계 (모델답/정답): 피어 {C[peers, P[t]].sum():+.1f}/{C[peers, G[i,t]].sum():+.1f}   비피어 {C[~pm[t], P[t]].sum()-C[t,P[t]]:+.1f}/{C[~pm[t], G[i,t]].sum()-C[t,G[i,t]]:+.1f}   자기 {C[t,P[t]]:+.1f}/{C[t,G[i,t]]:+.1f}")

if args.puzzles is None:
    # 오답 칸이 있는 퍼즐 중 앞에서부터
    pr = rollout(m, make_batch(inp[:64], inp[:64])); P0 = logits(m, pr)[:, :, 2:11].argmax(-1).cpu().numpy()
    wrong = ((P0 != G[:64]) & (I[:64] == 1)).sum(1); cand = np.where(wrong > 0)[0]
    args.puzzles = [int(c) for c in cand[:args.n_show]]
idx = torch.tensor(args.puzzles, device="cuda"); rec, last = trace(idx)
for j, i in enumerate(args.puzzles):
    P = rec["arg"][j, -1]; M = rec["mar"][j, -1]; q = quadrant(P, M, i); bl = I[i] == 1
    cnt = {k: int(sum(1 for t in range(81) if q[t] == k)) for k in "VXvx"}
    print(f"\n=== 퍼즐 {i}  빈칸 {bl.sum()}  V(확신정답) {cnt['V']}  X(확신오답) {cnt['X']}  v(비확신정답) {cnt['v']}  x(비확신오답) {cnt['x']}   τ={args.tau}")
    for rr in range(9):
        print("   " + " ".join(q[rr * 9 + cc] for cc in range(9)) + "     " + " ".join(f"{M[rr*9+cc]:4.0f}" if bl[rr*9+cc] else "   ." for cc in range(9)))
    cells = [parse(s) for s in args.cells] if args.cells else []
    if not cells:
        for k in "XxvV":
            ts = [t for t in range(81) if q[t] == k][:1]; cells += ts
    for t in cells[:args.max_cells]: show_cell(i, j, t, rec, last, P)
