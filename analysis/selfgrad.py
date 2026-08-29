"""내재 교정 신호 — 모델 자신의 정착 목적함수 J 의 상태 gradient.
  J(h) = Σ_{세그 s} Σ_t ‖p_t^{(s)} − p_t^{(s−1)}‖²  +  β Σ_t H(p_t^{(마지막)})     (h 에서 --J_loops 세그 롤아웃, 빈칸만)
  최종 carry h 에서 h ← h − η ∇_h J / ‖∇_h J‖·‖h‖·--rel 로 --steps 회. 매 스텝 16 loop 재실행 → 오답 수·유효성 (라벨은 판정에만).
  대조: 같은 상대 노름의 무작위 섭동.  사용: python selfgrad.py --puzzles 1 3 [--steps 6 --rel 0.1 --J_loops 4 --beta 1.0]"""
import argparse, numpy as np, torch, torch.nn.functional as F
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--puzzles", type=int, nargs="+", default=None); ap.add_argument("--n_pick", type=int, default=4)
ap.add_argument("--steps", type=int, default=6); ap.add_argument("--rel", type=float, default=0.1); ap.add_argument("--J_loops", type=int, default=4); ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args(); torch.manual_seed(args.seed)
m = load_lt(args.ckpt, bilinear=True, batch_size=1); inner = m.inner; core = inner.core; K = m.config.blocks_per_seg; L = m.config.loops
inp, lab, _ = load_test(); I = inp.cpu().numpy(); G = lab.cpu().numpy() - 2; pm = peer_mask()
for p in m.parameters(): p.requires_grad_(False)
with torch.no_grad():
    fc = tuple(t.detach() for t in core.kernel_fast()); AB = tuple(t.detach() for t in core.W_C())

def seg(h, inj):
    for _ in range(K):
        h = inner._boundary(h) + inner.inj_gate * inj; hp = core.phi(h, .5); f, a, *_ = core.field(hp, None, None, None, AB, fast_ctx=fc); h = core.phi(hp + f, .5)
    return h
def probs(h): return F.softmax(logits(m, h)[:, :, 2:11].float(), -1)
def J_of(h, inj, blank):
    ps = []; hh = h
    for _ in range(args.J_loops): hh = seg(hh, inj); ps.append(probs(hh))
    settle = sum(((ps[s] - ps[s - 1]) ** 2).sum(-1)[blank].sum() for s in range(1, len(ps)))
    ent = -(ps[-1] * torch.log(ps[-1] + 1e-9)).sum(-1)[blank].sum()
    return settle + args.beta * ent, settle.item(), ent.item()
def evaluate(h, inj, i):
    with torch.no_grad():
        hh = h
        for _ in range(L): hh = seg(hh, inj)
        P = probs(hh)[0].argmax(-1).cpu().numpy(); bl = I[i] == 1; fin = P.copy(); fin[~bl] = I[i][~bl] - 2
        valid = not any((fin[pm[t]] == fin[t]).any() for t in range(81))
        return int(((P != G[i]) & bl).sum()), valid, P

if args.puzzles is None:
    with torch.no_grad():
        P0 = logits(m, rollout(m, make_batch(inp[:64], inp[:64])))[:, :, 2:11].argmax(-1).cpu().numpy()
    args.puzzles = [int(i) for i in np.where(((P0 != G[:64]) & (I[:64] == 1)).any(1))[0][:args.n_pick]]
for i in args.puzzles:
    x = inp[i:i + 1]; inj = inner._injection(make_batch(x, x)).detach(); blank = torch.tensor(I[i] == 1, device="cuda")[None]
    with torch.no_grad():
        h0 = inner.init_hidden.expand(1, 81, -1).clone()
        for _ in range(L): h0 = seg(h0, inj)
    w0, v0, Pb = evaluate(h0, inj, i)
    print(f"\n=== 퍼즐 {i}: 원판 오답 {w0} {'유효' if v0 else '무효'}   ‖h‖ {h0.norm():.1f}")
    for mode in ["gradient", "random"]:
        h = h0.clone(); line = []
        for s in range(args.steps + 1):
            h = h.detach().requires_grad_(True); J, se, en = J_of(h, inj, blank)
            if s == 0 or mode == "gradient": J.backward(); g = h.grad.detach()
            wr, va, P = evaluate(h.detach(), inj, i)
            changed = int(((P != Pb) & (I[i] == 1)).sum()); fixed = int(((P == G[i]) & (Pb != G[i]) & (I[i] == 1)).sum()); broke = int(((P != G[i]) & (Pb == G[i]) & (I[i] == 1)).sum())
            line.append(f"s{s}: J {J.item():7.2f} (정착 {se:6.2f} 엔트 {en:6.2f}) 오답 {wr:2d} {'유효' if va else '무효'} 바뀜 {changed:2d} (고침 {fixed} 망침 {broke})")
            if s == args.steps: break
            with torch.no_grad():
                d = g if mode == "gradient" else torch.randn_like(h)
                d = d[blank.unsqueeze(-1).expand_as(d)].reshape(-1) if False else d * blank.unsqueeze(-1)   # 빈칸만 움직임
                h = h - d / (d.norm() + 1e-9) * h.norm() * args.rel
        print(f"  [{mode}]"); [print("    " + l) for l in line]
