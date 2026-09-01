"""a_tn 의 부호가 대칭부에서 오는가 반대칭부에서 오는가 — 쌍 범주별 분해.

물음: 피어 칸은 음의 메시지, 비피어 중 정답값 보유 칸은 양의 메시지라는 관측이
      (a) 대칭 결합  a_tn ≈ a_nt   → 상호 인력/척력, 순서 없음
      (b) 반대칭 결합 a_tn ≈ −a_nt → 한쪽만 양보, 순서 있음
      둘 중 무엇인가.  S=(a_tn+a_nt)/2 (짝), A=(a_tn−a_nt)/2 (홀) 로 분해해 크기를 비교한다.
사용: python 2026-08-31/analysis/sign_symmetry.py [--n 256] [--segs 16]
"""
import argparse, json, os, sys, importlib.util
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "analysis"))
from common import ROOT, URM, load_test, make_batch, peer_mask        # noqa: E402
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--segs", type=int, default=16)
ap.add_argument("--out", default=os.path.join(ROOT, "2026-08-31", "results", "json", "sign_symmetry.json"))
args = ap.parse_args(); torch.set_grad_enabled(False)

inp, lab, _ = load_test(args.n); N = len(inp)
G  = (lab - 2).long()                      # 정답 숫자 0..8
bl = inp == 1                              # 빈칸
pm = torch.tensor(peer_mask(), device="cuda")     # [81,81] 피어

ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"),
                map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v
      for k, v in ck.items()}
m = mod.LT(dict(batch_size=args.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
                puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
                stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda()
m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; eta = torch.sigmoid(inner.eta_raw).float(); K = 8

CATS = ["peer", "nonpeer_ans", "nonpeer_other"]
acc = {c: {k: [] for k in ("a_tn", "a_nt", "S", "A", "absS", "absA")} for c in CATS}
acc_w = {c: {k: [] for k in ("a_tn", "a_nt", "S", "A", "absS", "absA")} for c in CATS}
exact = 0
# 빈칸별 순위 — 비피어 중 정답값 보유 칸이 상위에 오는가
rank_stats = {"nonpeer_ans_percentile": [], "top10_frac_ans": [], "base_frac_ans": []}

for b in range(0, N, args.bs):
    x = inp[b:b + args.bs]; n_ = len(x); gt = G[b:b + n_]; blb = bl[b:b + n_]
    h = inner.init_hidden.expand(n_, 81, -1).clone(); w = None
    for s in range(args.segs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for k in range(K):
                h = inner.boundary(h); h = h + inner.inj_gate * inj
                a = inner.attn(h, AB, kc)
                v = torch.einsum('btd,hcd->bthc', h, inner.w_sh)          # 수송용 (정규화 안 함)
                vn = v / (v.norm(dim=-1, keepdim=True) + eps)             # agree 용 (정규화)
                agree = torch.einsum('bthc,bnhc->bhtn', vn, vn)
                Gm = a * agree
                w = Gm if w is None else w + eta * (Gm - w)
                aeff = w                                          # λ=1 로 학습됨 → 전달 = w
                o = torch.einsum('bhtn,bnhc->bthc', aeff.to(v.dtype), v)
                f = torch.einsum('bthc,hcd->btd', o, inner.w_sh)
                h = inner.phi(h + f)
            lg = inner.w_cls(h).float()[:, :, 2:11]
        h = h.float()
    P = lg.argmax(-1)
    exact += int(((P == gt) | ~blb).all(1).sum())

    A_ = a.float()                                                # 마지막 블록의 커널 [B,H,T,T]
    W_ = w.float()
    # 빈칸별: 비피어 칸을 w_tn 로 정렬했을 때 정답값 보유 칸의 백분위
    Wh_ = W_.mean(1)                                              # [B,T,T]
    same_ = (gt[:, None, :] == gt[:, :, None])
    npeer = (~pm[None].expand(n_, -1, -1)) & ~torch.eye(81, dtype=torch.bool, device="cuda")[None]
    for bi in range(n_):
        for t in torch.nonzero(blb[bi]).flatten().tolist():
            cand = npeer[bi, t]
            if cand.sum() < 5: continue
            vals = Wh_[bi, t][cand]; isans = same_[bi, t][cand]
            if isans.sum() == 0 or isans.all(): continue
            order = vals.argsort(descending=True)
            ranks = torch.empty_like(order); ranks[order] = torch.arange(len(order), device="cuda")
            pct = 1.0 - ranks[isans].float().mean() / (len(order) - 1)     # 1.0 = 최상위
            rank_stats["nonpeer_ans_percentile"].append(float(pct))
            k10 = max(1, len(order) // 10)
            rank_stats["top10_frac_ans"].append(float(isans[order[:k10]].float().mean()))
            rank_stats["base_frac_ans"].append(float(isans.float().mean()))
    for M_, store in ((A_, acc), (W_, acc_w)):
        Sy = 0.5 * (M_ + M_.transpose(-1, -2)); As = 0.5 * (M_ - M_.transpose(-1, -2))
        Mh = M_.mean(1); Sh = Sy.mean(1); Ah = As.mean(1)          # 헤드 평균 [B,T,T]
        Mt = M_.transpose(-1, -2).mean(1)
        # 값(라벨 기준) 일치 마스크: n 의 값 == t 의 정답
        same = (gt[:, None, :] == gt[:, :, None])                  # [B,T(t),T(n)]
        peer = pm[None].expand(n_, -1, -1)
        tblank = blb[:, :, None].expand(-1, -1, 81)                # t 가 빈칸
        masks = dict(peer=tblank & peer,
                     nonpeer_ans=tblank & ~peer & same,
                     nonpeer_other=tblank & ~peer & ~same)
        for c, msk in masks.items():
            msk = msk & ~torch.eye(81, dtype=torch.bool, device="cuda")[None]
            if msk.sum() == 0: continue
            store[c]["a_tn"].append(float(Mh[msk].mean())); store[c]["a_nt"].append(float(Mt[msk].mean()))
            store[c]["S"].append(float(Sh[msk].mean()));    store[c]["A"].append(float(Ah[msk].mean()))
            store[c]["absS"].append(float(Sh[msk].abs().mean())); store[c]["absA"].append(float(Ah[msk].abs().mean()))

res = {}
print(f"완답 {exact}/{N}  (segs={args.segs})\n")
for name, store in (("커널 a", acc), ("결합 w (= 실제 전달, λ=1)", acc_w)):
    print(f"=== {name} ===")
    print(f"{'범주':<15}{'a_tn':>9}{'a_nt':>9}{'S=(a+aᵀ)/2':>13}{'A=(a−aᵀ)/2':>13}{'|S|':>9}{'|A|':>9}{'|A|/|S|':>9}")
    for c in CATS:
        if not store[c]["a_tn"]: continue
        v = {k: float(np.mean(store[c][k])) for k in store[c]}
        r = v["absA"] / (v["absS"] + 1e-12)
        print(f"{c:<15}{v['a_tn']:>9.4f}{v['a_nt']:>9.4f}{v['S']:>13.4f}{v['A']:>13.4f}{v['absS']:>9.4f}{v['absA']:>9.4f}{r:>9.3f}")
        res[f"{name}|{c}"] = dict(v, ratio_absA_absS=r)
    print()
print("=== 빈칸별 순위 (비피어 칸만, w 헤드평균으로 정렬) ===")
rs = {k: float(np.mean(v)) for k, v in rank_stats.items() if v}
print(f"  정답값 보유 칸의 평균 백분위 (1.0=최상위, 0.5=무작위): {rs.get('nonpeer_ans_percentile', float('nan')):.4f}")
print(f"  상위 10% 안의 정답값 보유 비율: {rs.get('top10_frac_ans', float('nan')):.4f}"
      f"   (기저 비율 {rs.get('base_frac_ans', float('nan')):.4f})")
lift = rs.get('top10_frac_ans', 0) / (rs.get('base_frac_ans', 1e-9))
print(f"  리프트: {lift:.2f}x   표본 {len(rank_stats['nonpeer_ans_percentile']):,} 빈칸\n")
res["rank"] = dict(rs, lift=lift, n=len(rank_stats["nonpeer_ans_percentile"]))
os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(res, open(args.out, "w"), indent=1)
print("saved", args.out)
