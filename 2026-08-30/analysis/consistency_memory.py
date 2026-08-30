"""STDP 누적 축을 '제시' 위에 놓기: 초기 상태를 다르게 깬 재전개(제시) 마다 결합 기억 C_tn ← C_tn + η(Γ_tn − C_tn) 을 쌓고,
전달에 w + λ·C 를 쓴다 (Γ = 그 제시의 최종 a·⟨v̂,v̂⟩). 제시 간 일관된 쌍의 결합만 남는다. 라벨은 채점에만. stdp1, w 세그먼트 초기화.
  변형: lam=0 (단순 재시작, 마지막 제시 채점) / lam>0 (C 결합) / 참고: 제시들의 셀별 다수결
사용: python analysis/consistency_memory.py [--n 512] [--R 8] [--sigma 0.3] [--lam 0 0.5 1.0] [--eta 0.3]"""
import argparse, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--R", type=int, default=8); ap.add_argument("--sigma", type=float, default=0.3)
ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.3, 0.5]); ap.add_argument("--eta", type=float, default=0.3); ap.add_argument("--segs", type=int, default=16)
ap.add_argument("--eta_sched", default="const", help="const | mean (η_r = 1/(r+1): 제시 평균) | inv (η0/(1+r/8))")
ap.add_argument("--lam_max", type=float, default=-1.0, help=">0 이면 λ 를 λ→λ_max 로 제시에 따라 선형 증가")
ap.add_argument("--lock", type=float, default=-1.0, help=">0 이면 퍼즐별 잠금: 직전 제시 대비 바뀐 빈칸 비율이 이 값 아래면 그 퍼즐의 λ 를 +lock_step 씩 lam_max 까지, 아니면 λ 로 복귀")
ap.add_argument("--lock_step", type=float, default=0.1)
ap.add_argument("--aug", default="relabel_transpose", help="제시 = none | relabel | relabel_transpose | full (재라벨+전치+밴드/스택/행/열 순열; 첫 제시는 항등)"); args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; eta_h = torch.sigmoid(inner.eta_raw).float(); K = 8
TR = torch.tensor([(c % 9) * 9 + c // 9 for c in range(81)], device="cuda")            # 전치: 새 위치 j 에 원래 칸 TR[j]
def cell_perm(gen, tr, geo):
    """칸 순열 cp: 새 위치 j 에 원래 칸 cp[j]. geo: 밴드·스택·밴드 안 행·스택 안 열 순열"""
    rows = torch.arange(9, device="cuda"); cols = torch.arange(9, device="cuda")
    if geo:
        band = torch.randperm(3, generator=gen, device="cuda"); rows = torch.cat([band[b] * 3 + torch.randperm(3, generator=gen, device="cuda") for b in range(3)])
        stack = torch.randperm(3, generator=gen, device="cuda"); cols = torch.cat([stack[b] * 3 + torch.randperm(3, generator=gen, device="cuda") for b in range(3)])
    cp = (rows[:, None] * 9 + cols[None, :]).reshape(-1)                                 # 새 (r,c) ← 원래 (rows[r], cols[c])
    if tr: cp = cp[TR]
    return cp
def make_view(x, gen, r):
    """r 번째 제시의 입력 뷰. 반환 x', 숫자 순열 perm (원래 d → perm[d]), 칸 순열 cp"""
    ident = torch.arange(81, device="cuda")
    if r == 0 or args.aug == "none": return x, torch.arange(9, device="cuda"), ident
    perm = torch.randperm(9, generator=gen, device="cuda"); xv = x.clone(); dig = (x >= 2)
    xv[dig] = (perm[(x[dig] - 2).long()] + 2).to(x.dtype)
    tr = (args.aug in ("relabel_transpose", "full")) and bool(torch.rand(1, generator=gen, device="cuda") < 0.5)
    cp = cell_perm(gen, tr, geo=(args.aug == "full")); xv = xv[:, cp]
    return xv, perm, cp
def to_canon(P, Gm, perm, cp):
    inv = torch.argsort(perm); P = inv[P]; icp = torch.argsort(cp)                          # 숫자·칸 되돌리기: 정규[c] = 뷰[icp[c]]
    return P[:, icp], Gm[:, :, icp][:, :, :, icp]
def present(x, C, lam, gen):
    lam_t = lam if not torch.is_tensor(lam) else lam.view(-1, 1, 1, 1)
    """한 제시: 잡음 초기 상태, 16 세그먼트(w 세그먼트 초기화), 전달 = w + lam·C. 반환 예측, 최종 Γ"""
    n = len(x); h0 = inner.init_hidden.expand(n, 81, -1).clone(); h = h0 + args.sigma * h0.norm(dim=-1, keepdim=True) * torch.randn(h0.shape, generator=gen, device="cuda") / (h0.shape[-1] ** 0.5)
    for s in range(args.segs):
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for _ in range(K):
                h = inner.boundary(h); h = h + inner.inj_gate * inj
                a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
                Gm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn); w = Gm if w is None else w + eta_h * (Gm - w)
                a_eff = (1 - lam_t) * w + lam_t * C if C is not None else w                 # 볼록 혼합 (규모 보존), λ 는 스칼라 또는 퍼즐별
                o = torch.einsum('bhtn,bnhc->bthc', a_eff.to(vv.dtype), vv); f = torch.einsum('bthc,hcd->btd', o, inner.w_sh); h = inner.phi(h + f)
            lg = inner.w_cls(h).float()[:, :, 2:11]
        h = h.float()
    return lg.argmax(-1), Gm.float()
for lam in args.lam:
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    solved_hist = []; vote_hist = []; changed = []; ever = torch.zeros(N, dtype=torch.bool, device="cuda"); lost_hist = []; lockin_hist = []; hist_P = []
    P_all = torch.zeros(N, 81, dtype=torch.long, device="cuda"); votes = torch.zeros(N, 81, 9, device="cuda"); prevP = None; LAMI = torch.full((N,), lam, device="cuda")
    for r in range(args.R):
        for b in range(0, N, 128):
            x = inp[b:b + 128]; n = len(x); xv, perm, cp = make_view(x, gen, r)
            if r == 0 and b == 0: Cs = {}
            Cb = Cs.get(b); Cv = (Cb[:, :, cp][:, :, :, cp] if Cb is not None else None)          # 정규 → 뷰 좌표: 뷰[j] = 정규[cp[j]]
            lam_r = lam if args.lam_max <= 0 else lam + (args.lam_max - lam) * r / max(args.R - 1, 1)
            if args.lock > 0: lam_r = LAMI[b:b + n]                                          # 퍼즐별 잠금 λ
            eta_r = args.eta if args.eta_sched == "const" else (1.0 / (r + 1) if args.eta_sched == "mean" else args.eta / (1 + r / 8))
            P, Gm = present(xv, Cv if (r > 0) else None, lam_r, gen); P, Gm = to_canon(P, Gm, perm, cp)
            Cs[b] = Gm if Cb is None else Cb + eta_r * (Gm - Cb)
            P_all[b:b + n] = P
        votes.scatter_add_(-1, P_all.unsqueeze(-1), torch.ones_like(votes[..., :1]))
        sv = ((P_all == G) | ~bl).all(1); solved = int(sv.sum()); V = votes.argmax(-1); vsolved = (((V == G) | ~bl).all(1)).sum().item()
        lost_hist.append(int((ever & ~sv).sum())); ever |= sv; hist_P.append(P_all.clone())
        if len(hist_P) >= 3:                                                              # 최근 3제시에서 값이 같은데 틀린 칸 (오류 고착)
            cons = (hist_P[-1] == hist_P[-2]) & (hist_P[-2] == hist_P[-3]); lock = cons & (P_all != G) & bl
            lockin_hist.append(round(float(lock[~sv].sum(1).float().median()) if (~sv).any() else 0, 1))
        if prevP is not None:
            changed.append(float(((P_all != prevP) & bl).float().sum() / bl.sum()))
            if args.lock > 0:                                                                   # 퍼즐별: 안정하면 λ 올리고, 흔들리면 λ 로 복귀
                ch_i = ((P_all != prevP) & bl).float().sum(1) / bl.sum(1)
                LAMI = torch.where(ch_i < args.lock, torch.clamp(LAMI + args.lock_step, max=args.lam_max), torch.full_like(LAMI, lam))
        prevP = P_all.clone(); solved_hist.append(solved); vote_hist.append(vsolved)
    print(f"λ={lam}→{args.lam_max if args.lam_max>0 else lam} lock={args.lock} aug={args.aug} η={args.eta}/{args.eta_sched}:\n  제시별 완답 {solved_hist}\n  한 번이라도 완답 누적 {int(ever.sum())} | 맞았다가 틀린 퍼즐(제시별) {lost_hist}\n  미해결 퍼즐의 '3제시 연속 같은 값인 오답 칸' 중앙값(제시별) {lockin_hist}\n  다수결 누적 {vote_hist} | 제시 간 바뀐 빈칸 비율 {np.mean(changed) if changed else 0:.3f}", flush=True)
