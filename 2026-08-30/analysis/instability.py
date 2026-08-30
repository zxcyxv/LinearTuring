"""칸별 '응답 진폭'(회전에 대한 흔들림) 이 오답 칸을 가르는가 — 라벨은 채점에만.
  inst_t = EMA_β‖ℓ_t(k) − ℓ_t(k−1)‖ / (EMA_β gap_t + 1),  ℓ = 9 숫자 로짓(블록마다), gap = 1등−2등 로짓.
사용: python analysis/instability.py [--n 512] [--segs 16 32] [--beta 0.0625] [--puzzle 57]"""
import argparse, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--segs", type=int, nargs="+", default=[16, 32]); ap.add_argument("--beta", type=float, default=1 / 16); ap.add_argument("--puzzle", type=int, default=57)
args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
S = max(args.segs); INST = {s: torch.zeros(N, 81, device="cuda") for s in args.segs}; PRED = {s: torch.zeros(N, 81, dtype=torch.long, device="cuda") for s in args.segs}
def auc(pos, neg):
    pos = pos.cpu().numpy(); neg = neg.cpu().numpy(); allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); prev = None; dch = torch.zeros(n, 81, device="cuda"); gap = torch.zeros(n, 81, device="cuda")
    for s in range(S):
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for _ in range(8):
                h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
                l = inner.w_cls(h).float()[:, :, 2:11]
                if prev is not None: dch = dch + args.beta * ((l - prev).norm(dim=-1) - dch)
                t2 = l.topk(2, -1).values; gap = gap + args.beta * ((t2[..., 0] - t2[..., 1]) - gap); prev = l
        h = h.float()
        if s + 1 in INST: INST[s + 1][b:b + n] = dch / (gap + 1); PRED[s + 1][b:b + n] = l.argmax(-1)
for s in args.segs:
    inst = INST[s]; P = PRED[s]; wrong = (P != G) & bl; right = (P == G) & bl; giv = ~bl
    unsolved = wrong.any(1); right_in_unsolved = right & unsolved[:, None]; right_in_solved = right & ~unsolved[:, None]
    print(f"\n[세그먼트 {s}] 칸별 응답 진폭 inst (중앙값 / 90%):  오답 칸 {inst[wrong].median():.3f} / {inst[wrong].quantile(.9):.3f}   "
          f"미해결 퍼즐의 정답 빈칸 {inst[right_in_unsolved].median():.3f} / {inst[right_in_unsolved].quantile(.9):.3f}   "
          f"해결 퍼즐의 빈칸 {inst[right_in_solved].median():.3f} / {inst[right_in_solved].quantile(.9):.3f}   주어진 칸 {inst[giv].median():.3f}")
    print(f"   오답 vs 정답빈칸 AUC (전체) {auc(inst[wrong], inst[right]):.3f}    오답 vs 같은(미해결) 퍼즐의 정답 빈칸 AUC {auc(inst[wrong], inst[right_in_unsolved]):.3f}")
    # 퍼즐 안 순위: 오답 칸이 그 퍼즐의 빈칸 중 상위 k 에 드는 비율
    for k in (5, 10):
        hits = 0; tot = 0
        for i in torch.where(unsolved)[0].tolist():
            v = inst[i].clone(); v[~bl[i]] = -1; top = set(v.topk(k).indices.tolist()); ws = torch.where(wrong[i])[0].tolist(); hits += len(top & set(ws)); tot += len(ws)
        print(f"   미해결 퍼즐에서 진폭 상위 {k}칸이 오답 칸을 덮는 비율: {hits/max(tot,1):.3f}  (퍼즐당 오답 평균 {tot/max(int(unsolved.sum()),1):.1f})")
    i = args.puzzle
    if i < N:
        v = inst[i]; order = torch.argsort(v, descending=True)[:12].tolist()
        print(f"   퍼즐 {i} 진폭 상위 12칸: " + "  ".join(f"칸{c}({'오답' if wrong[i,c] else ('정답' if bl[i,c] else '주어짐')}, {v[c]:.2f})" for c in order))
        print(f"   퍼즐 {i} 추적 칸: " + "  ".join(f"칸{c}={v[c]:.2f}" for c in (2, 4, 20, 21, 76, 12, 26)) + f"   정답 빈칸 중앙값 {v[right[i]].median():.2f}")
