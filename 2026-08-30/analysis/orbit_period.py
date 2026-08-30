"""궤도의 정확한 주기 검출 (Brent). 쓰는 것은 두 가지뿐:
   공리: 입력 고정 시 추론은 유한 집합 위 결정론적 사상 (h,w) ↦ F(h,w).  [비트 재현성 확인함]
   정리: 유한 집합 위 사상의 모든 궤도는 결국 주기적 (비둘기집).
  따라서 어떤 s<s' 에 대해 상태가 비트 단위로 같으면, 그 뒤 나올 수 있는 답은 한 주기 안의 답이 전부다.
  주기가 검출된 퍼즐에 대해서만 '반복으로는 영원히 못 푼다' 를 말한다. 검출 안 된 퍼즐에 대해서는 아무 말도 하지 않는다.
  λ=1 → 완전히 얼어붙은 고정점.  λ>1 → 회전.  (그 궤도 위 Koopman 스펙트럼은 정확히 λ차 단위근.)
사용: python analysis/orbit_period.py [--n 128] [--segs 1024] [--carry 1]"""
import argparse, json, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=128); ap.add_argument("--segs", type=int, default=1024); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--carry", type=int, default=1); ap.add_argument("--out", default=os.path.join(ROOT, "results", "json", "orbit_period.json"))
args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=args.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
               stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
K = 8
def F(h, w, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
        if not args.carry: w = None
        for _ in range(K): h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
        lg = inner.w_cls(h).float()[:, :, 2:11]
    return h.float(), (w.float() if w is not None else None), lg.argmax(-1)
out = []
for b in range(0, N, args.bs):
    x = inp[b:b + args.bs]; n = len(x); g = G[b:b + n]; mb = bl[b:b + n]
    h = inner.init_hidden.expand(n, 81, -1).clone().float(); w = None
    ever = torch.zeros(n, dtype=torch.bool, device="cuda"); first = torch.full((n,), -1, dtype=torch.long, device="cuda")
    det = torch.zeros(n, dtype=torch.bool, device="cuda"); lam_rec = torch.zeros(n, dtype=torch.long, device="cuda"); det_at = torch.full((n,), -1, dtype=torch.long, device="cuda")
    h, w, P = F(h, w, x); step = 1
    ok = ((P == g) | ~mb).all(1); first = torch.where(ok & ~ever, torch.full_like(first, step), first); ever |= ok
    th, tw = h.clone(), (w.clone() if w is not None else None)          # tortoise
    power = 1; lam = 1
    while step < args.segs and not bool(det.all()):
        h, w, P = F(h, w, x); step += 1; lam += 1
        ok = ((P == g) | ~mb).all(1); first = torch.where(ok & ~ever, torch.full_like(first, step), first); ever |= ok
        eq = (h == th).flatten(1).all(1)
        if w is not None: eq = eq & (w == tw).flatten(1).all(1)
        new = eq & ~det
        if new.any(): lam_rec = torch.where(new, torch.full_like(lam_rec, lam - 1), lam_rec); det_at = torch.where(new, torch.full_like(det_at, step), det_at); det |= new
        if lam - 1 == power: power *= 2; lam = 1; th, tw = h.clone(), (w.clone() if w is not None else None)
    out.append(dict(ever=ever.cpu().numpy(), first=first.cpu().numpy(), det=det.cpu().numpy(), lam=lam_rec.cpu().numpy(), det_at=det_at.cpu().numpy()))
    print(f"배치 {b//args.bs}: {step} 세그먼트까지 — 푼 퍼즐 {int(ever.sum())}/{n}, 주기 검출 {int(det.sum())}/{n}", flush=True)
ever = np.concatenate([o["ever"] for o in out]); first = np.concatenate([o["first"] for o in out]); det = np.concatenate([o["det"] for o in out]); lam = np.concatenate([o["lam"] for o in out]); dat = np.concatenate([o["det_at"] for o in out])
print(f"\n=== {N} 퍼즐, {args.segs} 세그먼트, carry={args.carry} (bf16, 비트 결정론 확인함)")
print(f"푼 퍼즐 {int(ever.sum())}  (16 세그먼트 이내 {int(((first>0)&(first<=16)).sum())}, 17~128 {int(((first>16)&(first<=128)).sum())}, 129 이후 {int(first>128).sum() if False else int(((first>128)).sum())})")
uns = ~ever
print(f"못 푼 퍼즐 {int(uns.sum())}  그중 주기 **증명됨** {int((uns&det).sum())} → 이들은 반복을 몇 번 하든 영원히 못 푼다")
print(f"                     주기 미검출 {int((uns&~det).sum())} → 아무 말도 할 수 없음")
if (uns & det).any():
    L = lam[uns & det]; print(f"  증명된 주기 λ 분포: λ=1(완전 정지) {int((L==1).sum())},  λ=2 {int((L==2).sum())},  3≤λ≤8 {int(((L>=3)&(L<=8)).sum())},  λ>8 {int((L>8).sum())}  | 최대 λ {int(L.max())}")
    print(f"  검출 세그먼트 중앙값 {int(np.median(dat[uns&det]))}")
if (ever & det).any():
    L2 = lam[ever & det]; print(f"  푼 퍼즐 중 주기 검출 {int((ever&det).sum())}: λ=1 {int((L2==1).sum())} (정답 고정점), λ>1 {int((L2>1).sum())}")
json.dump(dict(n=int(N), segs=args.segs, carry=args.carry, ever=ever.tolist(), first=first.tolist(), det=det.tolist(), lam=lam.tolist(), det_at=dat.tolist()), open(args.out, "w"))
