# 결정 순간의 실제 지지 기여로 의존 그래프를 만들고, 충돌 칸에서 책임을 거슬러 흘려 뿌리를 찾는다 (라벨은 채점에만)
import os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis')); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", "core/minimal.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); inp, lab, _ = load_test(512); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load("checkpoints/R1B8_min_stdp1_step228501.pt", map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; Wc = inner.w_cls.weight[2:11].float(); K = 8; T = 16 * K; OFFD = ~torch.eye(81, dtype=torch.bool, device="cuda"); HEADS = [0, 2, 4]
def rollout(x, commit=None):
    """commit 이 None: 결정 블록 계산. 아니면 각 t 의 결정 블록에서 n→t 지지 기여 dep[t,n] (t 의 결정 숫자 로짓에 n 이 보탠 양) 기록"""
    n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); ARG = torch.zeros(n, T, 81, dtype=torch.long, device="cuda"); dep = torch.zeros(n, 81, 81, device="cuda"); k = 0
    for s in range(16):
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for _ in range(K):
                h = inner.boundary(h); h = h + inner.inj_gate * inj
                a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
                wm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn) if w is None else w
                h, w = inner.step(h, AB, kc, w, None, None)
                l = inner.w_cls(h).float()[:, :, 2:11]; P = l.argmax(-1); ARG[:, k] = P
                if commit is not None:
                    sel = (commit == k)                                                                  # 이 블록에 결정된 칸들
                    if sel.any():
                        wf = wm.float() * OFFD[None, None]
                        o = torch.einsum('bhtn,bnhc->bnthc', wf, vv.float())                              # n 별 기여 [n,N,T,H,C]... 메모리: 128*81*81*8*104 너무 큼 → t 선택
                        # 대신 t 마다: f_tn = Σ_h w_h,tn (Wᵀ_h W_h v_n) → 로짓[x_t]: 헤드별 W_h v_n 를 읽기로 미리 사영
                        # Wc·(Wᵀ_h (W_h v_n)) = (Wc Wᵀ_h)(W_h v_n): R_h = Wc @ w_sh[h]ᵀ [9,C]
                        Rh = torch.einsum('vd,hcd->hvc', Wc, inner.w_sh.float())                           # [H,9,C]
                        proj = torch.einsum('hvc,bnhc->bhnv', Rh, vv.float())                              # [n,H,N,9]: 헤드 h 로 n 이 보내는 값의 숫자별 로짓
                        xt = P                                                                             # 결정 숫자 = 그 블록의 argmax
                        pv = proj.gather(-1, xt[:, None, None, :].expand(-1, 8, 81, -1))                    # [n,H,N,T]: n 의 값이 t 의 결정 숫자에 주는 로짓
                        contrib = torch.einsum('bhtn,bhnt->btn', wf, pv)                                   # [n,T,N]
                        dep[sel] = contrib.float()[sel]
                k += 1
        h = h.float()
    af = inner.attn(h, AB, kc).float(); pr = torch.softmax(inner.w_cls(h).float()[:, :, 2:11], -1); pp = torch.einsum('btd,bnd->btn', pr, pr)
    viol = (((pp * (-af[:, HEADS].sum(1)).clamp_min(0) - 0.216) > 0) & OFFD[None])
    return ARG, dep, viol
res = []
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); ARG, _, viol = rollout(x)
    final = ARG[:, -1]; stable = (ARG == final[:, None, :]); commit = torch.clamp(T - torch.flip(stable, [1]).cummin(1).values.sum(1), max=T - 1)
    _, dep, _ = rollout(x, commit)
    g = G[b:b+n]; wrong = (final != g) & bl[b:b+n]; uns = wrong.any(1); first_wrong = torch.where(wrong, commit.float(), torch.full_like(commit.float(), 1e9)).min(1).values
    conf_int = viol.any(-1).float(); conf_true = ((final[:, :, None] == final[:, None, :]) & pm[None] & OFFD[None]).any(-1).float()
    earlier = (commit[:, None, :] <= commit[:, :, None]) & OFFD[None]                                    # n 이 t 와 같거나 먼저
    Dp = (dep.clamp_min(0) * earlier.float()); Dp = Dp / (Dp.sum(-1, keepdim=True) + 1e-6)                # t 의 결정을 지지한 n 들에 비례 배분 (행 정규화)
    for cname, conf in (("내부탐지", conf_int), ("실제충돌", conf_true)):
        blame = conf.clone(); acc = conf.clone()
        for _ in range(12): blame = torch.bmm(Dp.transpose(1, 2), blame.unsqueeze(-1)).squeeze(-1); acc = acc + blame   # 거슬러 흘림
        for B in (1, 3, 127):
            early = (commit <= B) & bl[b:b+n]; sc = torch.where(early, acc, torch.full_like(acc, -1)); c = sc.argmax(1)
            ar = torch.arange(n, device="cuda"); isw = wrong[ar, c]; ist = isw & (commit[ar, c].float() <= first_wrong + 0.5)
            res.append((cname, B, int(isw[uns].sum()), int(ist[uns].sum()), int(uns.sum())))
    # 지지 기여 자체의 성질: 오답 칸의 결정 지지 중 오답 칸에서 온 비율
    wsrc = (dep.clamp_min(0) * wrong[:, None, :].float()).sum(-1) / (dep.clamp_min(0).sum(-1) + 1e-6)
    res.append(("지지출처", -1, float(wsrc[wrong].mean()), float(wsrc[(final == g) & bl[b:b+n] & uns[:, None]].mean()), 0))
import collections; agg = collections.defaultdict(lambda: [0, 0, 0]); src = [[], []]
for r in res:
    if r[0] == "지지출처": src[0].append(r[2]); src[1].append(r[3]); continue
    a = agg[(r[0], r[1])]; a[0] += r[2]; a[1] += r[3]; a[2] += r[4]
for (cname, B), (w, t, u) in sorted(agg.items()):
    print(f"책임 흘리기 [{cname}] 이른 칸(결정≤{B}) 중 책임 최다 → 오답 {w/u:.2f}  진짜 뿌리 {t/u:.2f}   (미해결 {u})")
print(f"오답 칸의 결정 지지 중 오답 칸에서 온 비율 {np.mean(src[0]):.2f} | 미해결 퍼즐 정답 칸의 그 비율 {np.mean(src[1]):.2f}")
print("기준: 이른 칸 무작위 ≈ 오답 .25 / 뿌리 .20,  위상 pre-ness 최소 → .24 / .05")
