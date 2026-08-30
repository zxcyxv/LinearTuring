"""블록 1 추측 되돌리기, 깊이 2 (탐욕). 라벨은 채점에만.
  L0: 평소 풀이 → 정지 안 하면 미해결. 후보 = 결정 블록 ≤ B0 빈칸, 지지 약한 순 최대 M.
  L1: 후보마다 (칸, 처음 숫자) 억제 후 재전개(추적). 채택 = 억제 아래 정지 **그리고** 억제를 뺀 뒤 2세그 굴려 스스로 정지(clean).
  L2: L1 에서 채택 안 된 퍼즐: 잔차 최소인 L1 재전개 P개를 부모로, 부모 풀이의 첫 추측 칸 M개를 억제 누적(부모 억제 + 새 억제)해 재전개. 채택 동일.
사용: python analysis/guess_retract2.py [--n 512] [--M 8] [--P 2] [--B0 1] [--delta 10] [--tau 0.03]"""
import argparse, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--M", type=int, default=8); ap.add_argument("--P", type=int, default=2); ap.add_argument("--B0", type=int, default=1)
ap.add_argument("--delta", type=float, default=10.0); ap.add_argument("--tau", type=float, default=0.03); ap.add_argument("--segs", type=int, default=16); args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; OFFD = ~torch.eye(81, dtype=torch.bool, device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; Wc = inner.w_cls.weight[2:11].float(); Wp = Wc.T @ torch.linalg.inv(Wc @ Wc.T); K = 8; T = args.segs * K; ar = lambda n: torch.arange(n, device="cuda")
def segment(h, x, bias):
    w = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x)) + bias
        for _ in range(K): h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
    return h.float()
def run(x, bias):
    """추적 재전개: 예측, 억제 아래 정지, clean 정지(억제 제거 후 2세그), clean 예측, 결정 블록, 지지"""
    n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); ARG = torch.zeros(n, T, 81, dtype=torch.long, device="cuda"); EXT = torch.zeros(n, T, 81, device="cuda"); k = 0
    for s in range(args.segs):
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
    h2 = segment(h, x, bias); still = (h2 - h).flatten(1).norm(dim=1) / h.flatten(1).norm(dim=1)
    zero = torch.zeros_like(bias); hc = segment(segment(h, x, zero), x, zero); hc2 = segment(hc, x, zero); still_c = (hc2 - hc).flatten(1).norm(dim=1) / hc.flatten(1).norm(dim=1)
    Pc = (inner.w_cls(hc).float()[:, :, 2:11]).argmax(-1); final = ARG[:, -1]
    stable = (ARG == final[:, None, :]); commit = torch.clamp(T - torch.flip(stable, [1]).cummin(1).values.sum(1), max=T - 1); ext = EXT.gather(1, commit[:, None, :]).squeeze(1)
    return final, still, still_c, Pc, commit, ext
def candidates(commit, ext, mb):
    cm = commit.clone().float(); cm[~mb] = 1e9; mask = (cm <= args.B0) & mb; score = torch.where(mask, ext, torch.full_like(ext, 1e9)); return torch.argsort(score, dim=1), mask.sum(1)
tot = dict(base=0, L1=0, L2=0, acc1=0, acc1w=0, acc2=0, acc2w=0, uns=0, runs=0, biased_still_wrong=0)
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); g = G[b:b + n]; mb = bl[b:b + n]; zero = torch.zeros(n, 81, inner.d, device="cuda")
    P0, still0, _, _, commit0, ext0 = run(x, zero); solved0 = ((P0 == g) | ~mb).all(1); acc0 = still0 < args.tau; tot["base"] += int(solved0.sum())
    order0, nc0 = candidates(commit0, ext0, mb); active = ~acc0; tot["uns"] += int(active.sum())
    solved = solved0.clone(); accepted = acc0.clone(); accw = torch.zeros(n, dtype=torch.bool, device="cuda")
    L1 = []   # (bias, still, commit, ext) per candidate
    for j in range(args.M):
        has = active & (nc0 > j); cell = order0[:, j]; val = P0.gather(1, cell[:, None]).squeeze(1)
        bias = torch.zeros(n, 81, inner.d, device="cuda"); bias[ar(n), cell] = -args.delta * Wp[:, val].T * has[:, None].float()
        P1, still1, stillc, Pc, commit1, ext1 = run(x, bias); tot["runs"] += int(has.sum())
        ok = has & (still1 < args.tau) & (stillc < args.tau); sc = ((Pc == g) | ~mb).all(1)
        tot["biased_still_wrong"] += int((has & (still1 < args.tau) & ~sc).sum())
        newacc = ok & ~accepted; solved |= newacc & sc; accw |= newacc & ~sc; accepted |= newacc; active = active & ~ok
        L1.append((bias, torch.where(has, still1, torch.full_like(still1, 1e9)), commit1, ext1, P1))
    tot["L1"] += int(solved.sum()); tot["acc1"] += int(accepted.sum()); tot["acc1w"] += int(accw.sum())
    # L2: 잔차 최소 부모 P개
    stills = torch.stack([t[1] for t in L1], 1); parents = torch.argsort(stills, dim=1)[:, :args.P]
    for p in range(args.P):
        pj = parents[:, p]
        pbias = torch.stack([L1[j][0] for j in range(len(L1))], 1)[ar(n), pj]; pcommit = torch.stack([L1[j][2] for j in range(len(L1))], 1)[ar(n), pj]; pext = torch.stack([L1[j][3] for j in range(len(L1))], 1)[ar(n), pj]; pP = torch.stack([L1[j][4] for j in range(len(L1))], 1)[ar(n), pj]
        order1, nc1 = candidates(pcommit, pext, mb); valid_parent = stills[ar(n), pj] < 1e8
        for j in range(args.M):
            has = active & valid_parent & (nc1 > j); cell = order1[:, j]; val = pP.gather(1, cell[:, None]).squeeze(1)
            bias = pbias.clone(); bias[ar(n), cell] += -args.delta * Wp[:, val].T * has[:, None].float()
            P2, still2, stillc, Pc, _, _ = run(x, bias); tot["runs"] += int(has.sum())
            ok = has & (still2 < args.tau) & (stillc < args.tau); sc = ((Pc == g) | ~mb).all(1)
            tot["biased_still_wrong"] += int((has & (still2 < args.tau) & ~sc).sum())
            newacc = ok & ~accepted; solved |= newacc & sc; accw |= newacc & ~sc; accepted |= newacc; active = active & ~ok
    tot["L2"] += int(solved.sum()); tot["acc2"] += int(accepted.sum()); tot["acc2w"] += int(accw.sum())
    print(f"배치 {b//128}: 기준 {int(solved0.sum())} → L1 뒤 {tot['L1'] - sum(0 for _ in [0])} (누적) → L2 뒤 {int(solved.sum())} | 채택 {int(accepted.sum())} 오답채택 {int(accw.sum())}", flush=True)
print(f"\n합계 {N}: 기준 {tot['base']} → 깊이1 {tot['L1']} → 깊이2 {tot['L2']} | 채택(clean) 깊이1 {tot['acc1']}(오답 {tot['acc1w']}) 깊이2 {tot['acc2']}(오답 {tot['acc2w']}) | 억제 아래 정지했지만 clean 판정에서 걸러진 오답 {tot['biased_still_wrong']} | 미해결 {tot['uns']}, 재전개 총 {tot['runs']}회")
