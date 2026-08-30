"""블록 1 추측 되돌리기 (라벨 없음): 미해결(정지 안 함) 퍼즐마다 가장 먼저 결정된 빈칸들(결정 블록 ≤ B0, 지지 약한 순, 최대 M개)을 하나씩 잡아
  그 칸의 현재 숫자를 결합에서 억제(주입에 −δ·Wc⁺[x])하고 재전개. 채택 = 정지(한 세그먼트 더 돌린 상대 변위 < τ). 라벨은 채점과 참고(진짜 뿌리 억제 상한)에만.
사용: python analysis/guess_retract.py [--n 512] [--M 8] [--B0 1] [--delta 10] [--tau 0.03] [--oracle 1]"""
import argparse, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--M", type=int, default=8); ap.add_argument("--B0", type=int, default=1); ap.add_argument("--delta", type=float, default=10.0)
ap.add_argument("--tau", type=float, default=0.03); ap.add_argument("--segs", type=int, default=16); ap.add_argument("--oracle", type=int, default=1); args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda"); OFFD = ~torch.eye(81, dtype=torch.bool, device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; Wc = inner.w_cls.weight[2:11].float(); Wp = Wc.T @ torch.linalg.inv(Wc @ Wc.T); K = 8; T = args.segs * K
def segment(h, x, bias):
    w = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x)) + bias
        for _ in range(K): h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
    return h.float()
def run(x, bias, track=False):
    n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); ARG = torch.zeros(n, T, 81, dtype=torch.long, device="cuda") if track else None; EXT = torch.zeros(n, T, 81, device="cuda") if track else None; k = 0
    for s in range(args.segs):
        if not track: h = segment(h, x, bias); continue
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x)) + bias
            for _ in range(K):
                h = inner.boundary(h); h = h + inner.inj_gate * inj
                a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps); wm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn) if w is None else w
                h, w = inner.step(h, AB, kc, w, None, None)
                l = inner.w_cls(h).float()[:, :, 2:11]; P = l.argmax(-1)
                wf = wm.float() * OFFD[None, None]; o = torch.einsum('bhtn,bnhc->bthc', wf, vv.float()); dl = torch.einsum('bthc,hcd->btd', o, inner.w_sh.float()) @ Wc.T
                ARG[:, k] = P; EXT[:, k] = dl.gather(-1, P.unsqueeze(-1)).squeeze(-1) - dl.scatter(-1, P.unsqueeze(-1), -1e9).max(-1).values; k += 1
        h = h.float()
    P = (inner.w_cls(h).float()[:, :, 2:11]).argmax(-1); h2 = segment(h, x, bias); still = (h2 - h).flatten(1).norm(dim=1) / h.flatten(1).norm(dim=1)
    if not track: return P, still, None, None
    final = ARG[:, -1]; stable = (ARG == final[:, None, :]); commit = torch.clamp(T - torch.flip(stable, [1]).cummin(1).values.sum(1), max=T - 1); ext = EXT.gather(1, commit[:, None, :]).squeeze(1)
    return P, still, commit, ext
tot = dict(base=0, acc0=0, acc0_wrong=0, final=0, acc=0, acc_wrong=0, tried=0, rounds_used=[], oracle=0, root_in_cand=0, uns=0)
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); g = G[b:b + n]; mb = bl[b:b + n]; zero = torch.zeros(n, 81, inner.d, device="cuda")
    P, still, commit, ext = run(x, zero, track=True); solved = ((P == g) | ~mb).all(1); accepted = still < args.tau
    tot["base"] += int(solved.sum()); tot["acc0"] += int(accepted.sum()); tot["acc0_wrong"] += int((accepted & ~solved).sum())
    # 후보: 결정 블록 ≤ B0 인 빈칸, 지지 약한 순 (라벨 없음)
    cm = commit.clone().float(); cm[~mb] = 1e9; cand_mask = (cm <= args.B0) & mb
    score = torch.where(cand_mask, ext, torch.full_like(ext, 1e9)); order = torch.argsort(score, dim=1)          # 앞에서부터 후보
    ncand = cand_mask.sum(1)
    first_wrong = torch.where((P != g) & mb, commit.float(), torch.full_like(cm, 1e9)).min(1).values
    is_root = ((P != g) & mb) & (commit.float() <= first_wrong[:, None] + 0.5)                                    # 진짜 뿌리(첫 오답) 칸들
    active = ~accepted; final_solved = solved.clone(); acc_any = accepted.clone(); acc_wrong = (accepted & ~solved); used = torch.full((n,), -1, device="cuda")
    tot["uns"] += int(active.sum()); tot["root_in_cand"] += int((active & (is_root & cand_mask).any(1)).sum())
    for j in range(args.M):
        has = active & (ncand > j)
        if not has.any(): break
        cell = order[:, j]; val = P.gather(1, cell[:, None]).squeeze(1)
        bias = torch.zeros(n, 81, inner.d, device="cuda"); bias[torch.arange(n, device="cuda"), cell] = -args.delta * Wp[:, val].T * has[:, None].float()
        P2, still2, _, _ = run(x, bias); s2 = ((P2 == g) | ~mb).all(1); a2 = still2 < args.tau
        newacc = has & a2; final_solved |= has & s2 & a2; acc_wrong |= newacc & ~s2; acc_any |= newacc; used[newacc & (used < 0)] = j + 1; active = active & ~a2
        tot["tried"] += int(has.sum())
    tot["final"] += int(final_solved.sum()); tot["acc"] += int(acc_any.sum()); tot["acc_wrong"] += int(acc_wrong.sum()); tot["rounds_used"] += used[used > 0].tolist()
    if args.oracle:   # 참고 상한: 진짜 뿌리 하나를 억제
        rootcell = torch.where(is_root.any(1), is_root.float().argmax(1), torch.zeros(n, dtype=torch.long, device="cuda")); val = P.gather(1, rootcell[:, None]).squeeze(1); has = (~accepted) & is_root.any(1)
        bias = torch.zeros(n, 81, inner.d, device="cuda"); bias[torch.arange(n, device="cuda"), rootcell] = -args.delta * Wp[:, val].T * has[:, None].float()
        P3, still3, _, _ = run(x, bias); tot["oracle"] += int((has & ((P3 == g) | ~mb).all(1)).sum())
    print(f"배치 {b//128}: 기준 {int(solved.sum())} → 되돌리기 뒤 {int(final_solved.sum())} | 채택 {int(acc_any.sum())} (오답 채택 {int(acc_wrong.sum())})", flush=True)
r = np.array(tot["rounds_used"]) if tot["rounds_used"] else np.array([0])
print(f"\n합계 {N}: 기준 완답 {tot['base']} (정지 채택 {tot['acc0']}, 그중 오답 {tot['acc0_wrong']})")
print(f"미해결 {tot['uns']} 중 후보(결정≤{args.B0}) 안에 진짜 뿌리가 있는 퍼즐: {tot['root_in_cand']} ({tot['root_in_cand']/max(tot['uns'],1):.2f})")
print(f"되돌리기 최대 {args.M}회 뒤: 완답 {tot['final']} (+{tot['final']-tot['base']}) | 정지 채택 {tot['acc']} 중 오답 채택 {tot['acc_wrong']} | 재전개 총 {tot['tried']}회, 채택된 퍼즐의 사용 회수 중앙값 {np.median(r):.0f}")
if args.oracle: print(f"참고 상한(진짜 뿌리 하나를 억제, 라벨 사용): 미해결 {tot['uns']} 중 완답 {tot['oracle']}")
