"""추론 시 자기 교정 — 위반 간선의 반발 결합을 위반이 사라질 때까지 지수 성장 (Ercsey-Ravasz & Toroczkai 형 느린 변수).
학습된 stdp1 그대로(추가 학습 없음). 빠른 변수 = h (원래 스텝), 느린 변수 = 간선별 g_tn ≥ 0:
    pp_tn  = Σ_d p_t(d) p_n(d)            읽기 헤드(w_cls)가 보는 "같은 숫자" 확률 (값 벡터 코사인은 공통 성분 탓에 비선택적: 정밀도 1%)
    R_tn   = Σ_h relu(−a_h,tn)  [sig=s8, 헤드 선택 없음]   또는   relu(−Σ_{h∈0,2,4} a_h,tn)  [sig=s4, 피어에서 음수인 헤드만]
    viol_tn = relu(pp·R − θ0)            반발 결합인데 같은 숫자 = 모델 자신의 커널+읽기가 보는 위반. 유닛 정의 없음 (512 퍼즐 진단: s4 θ0=.216 정밀도 .86 재현율 .90)
    v̄_tn  ← v̄ + β(viol − v̄)              저역통과 (β=1/16: 시정수 2 세그먼트). 느린 변수는 정착된 위반만 봐야 한다 — 매 블록 성장시키면 초기 일시 충돌에서 폭발
    g_tn   ← min(cap, g(1 + ρ·v̄) + ρ·v̄)  위반이 지속되면 지수 성장;  v̄≈0 이면 g ← g(1−δ) 천천히 감쇠 (탈출 뒤 증폭 잔류 방지)
    a_eff,h = w_h − g ⊙ relu(−w_h) ⊙ pp    이미 반발인 헤드의 결합을 증폭. 같은 숫자가 아니게 되면(만족) 추가 힘 0
ρ=0 이면 학습된 추론(w 세그먼트 초기화 모드)과 동일.
진단(라벨/피어는 판정에만): 표시된 간선 중 실제 피어 충돌(같은 argmax 인 피어) 비율 = 정밀도, 실제 충돌 중 표시 비율 = 재현율.
사용: python analysis/ctds.py [--rho 0 0.2] [--tau 0.3] [--cap 30] [--segs 64] [--n 2048] [--reset 1] [--out JSON]"""
import argparse, json, os, sys, time, importlib.util, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ap = argparse.ArgumentParser()
ap.add_argument("--stdp", default=os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"))
ap.add_argument("--n", type=int, default=2048); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--segs", type=int, default=64)
ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.2]); ap.add_argument("--cap", type=float, default=10.0)
ap.add_argument("--reset", type=int, default=1); ap.add_argument("--flag", type=float, default=0.216); ap.add_argument("--tau", type=float, default=0.15)
ap.add_argument("--sig", default="s4"); ap.add_argument("--heads", type=int, nargs="+", default=[0, 2, 4])
ap.add_argument("--beta", type=float, default=1 / 16); ap.add_argument("--decay", type=float, default=0.02); ap.add_argument("--warm", type=int, default=16)
ap.add_argument("--uncommit", type=float, default=0.0, help=">0: 칸 수준 느린 변수 u_t (성장률 = 이 값 × 칸의 총 위반). 경계 항을 1/(1+u_t) 로 줄여 굳힘 해제")
ap.add_argument("--spread", type=float, default=0.0, help="u 의 이웃 확산 계수: u ← u + spread·Σ_n Â_tn u_n (Â = 헤드 평균 |a| 를 행 정규화)")
ap.add_argument("--marks", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64, 96, 128])
ap.add_argument("--out", default=os.path.join(ROOT, "results", "json", "ctds_stdp1_228501.json"))
args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
K = 8; S = args.segs; marks = [k for k in args.marks if k <= S]; OFFD = (~torch.eye(81, dtype=torch.bool, device="cuda")).float()

ck = torch.load(args.stdp, map_location="cpu", weights_only=False); sd = ck.get("model_state_dict", ck)
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") or k.startswith("model.") else k): v for k, v in sd.items()}
cfg = dict(batch_size=args.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
           psi_zero=False, addr_dim=0, stdp=True, stdp_target="product", stdp_eta_init=0.1, stdp_lam_init=0.25, stdp_lam_fixed=1.0, amp=True)
m = mod.LT(cfg).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eta = torch.sigmoid(inner.eta_raw).float(); eps = cfg_eps = inner.config.eps; tau = args.tau


def run(rho):
    ex = np.zeros(S); va = np.zeros(S); ce = np.zeros(S); fl = np.zeros(S); nv = np.zeros(S); prec = np.zeros(S); rec = np.zeros(S); gmax = np.zeros(S); ng = np.zeros(S)
    solved = torch.zeros(N, S, dtype=torch.bool, device="cuda"); nbl = int(bl.sum())
    for b in range(0, N, args.bs):
        x = inp[b:b + args.bs]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); w = None; P = None
        g = torch.zeros(n, 81, 81, device="cuda"); vbar = torch.zeros(n, 81, 81, device="cuda"); u = torch.zeros(n, 81, 1, device="cuda")
        for s in range(S):
            if args.reset: w = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
                for _ in range(K):
                    # 위반 신호는 블록 시작(전달·Φ 뒤) 상태에서 읽는다 — 주입 직후 상태는 빈칸들이 공통 성분으로 닮아 보여 정밀도가 무너짐
                    pr = torch.softmax(inner.w_cls(h).float()[:, :, 2:11], -1); pp = torch.einsum('btd,bnd->btn', pr, pr)   # 읽기 헤드 "같은 숫자" 확률
                    af = inner.attn(h, AB, kc).float(); R = (-af).clamp_min(0).sum(1) if args.sig == "s8" else (-af[:, args.heads].sum(1)).clamp_min(0)
                    viol = (pp * R - args.flag).clamp_min(0) * OFFD                                 # [n,T,T] 모델 내부 위반량 (대각 제외: 자기결합은 위반이 아님)
                    hb = inner.boundary(h); h = (h + (hb - h) / (1 + u)) if args.uncommit > 0 else hb        # 굳힘 해제: 경계 항을 1/(1+u_t) 로
                    h = h + inner.inj_gate * inj
                    a = inner.attn(h, AB, kc)                                                       # [n,H,T,T]
                    vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
                    agree = torch.einsum('bthc,bnhc->bhtn', vn, vn)
                    Gm = a * agree
                    w = Gm if w is None else w + eta * (Gm - w)                                     # 학습된 규칙 그대로
                    if s >= args.warm:                                                              # 느린 변수는 빠른 변수가 정착한 뒤에만
                        vbar = vbar + args.beta * (viol - vbar)
                        if rho > 0: g = torch.minimum(torch.where(vbar > 0.01, g * (1 + rho * vbar) + rho * vbar, g * (1 - args.decay)), torch.full_like(g, args.cap))
                        if args.uncommit > 0:
                            vt = vbar.sum(-1, keepdim=True)                                         # 칸의 총(저역통과) 위반
                            u = torch.minimum(torch.where(vt > 0.01, u * (1 + args.uncommit * vt) + args.uncommit * vt, u * (1 - args.decay)), torch.full_like(u, args.cap))
                            if args.spread > 0:
                                A = af.abs().mean(1) * OFFD; A = A / (A.sum(-1, keepdim=True) + 1e-6)
                                u = torch.minimum(u + args.spread * torch.bmm(A, u), torch.full_like(u, args.cap))
                    wf = w.float(); a_eff = wf - (g * pp).unsqueeze(1) * (-wf).clamp_min(0)
                    o = torch.einsum('bhtn,bnhc->bthc', a_eff.to(vv.dtype), vv); f = torch.einsum('bthc,hcd->btd', o, inner.w_sh)
                    h = inner.phi(h + f)
                lg = inner.w_cls(h).float()
            h = h.float(); w = w.float()
            lg9 = lg[:, :, 2:11]; Pn = lg9.argmax(-1); gg = G[b:b + n]; mm = bl[b:b + n]
            ok = ((Pn == gg) | ~mm).all(-1); fin = torch.where(mm, Pn, x - 2)
            conf = (fin[:, :, None] == fin[:, None, :]) & pm[None]                                  # 실제 피어 충돌 (진단)
            valid = ~conf.any(-1).any(-1)
            ex[s] += int(ok.sum()); va[s] += int(valid.sum()); ce[s] += int(((Pn == gg) & mm).sum())
            if P is not None: fl[s] += int(((Pn != P) & mm).sum())
            P = Pn; solved[b:b + n, s] = ok
            flagged = viol > 0                                                                      # 마지막 블록 기준
            nv[s] += float(flagged.sum()) / 2; tp = float((flagged & conf).sum()); prec[s] += tp; rec[s] += float(conf.sum())
            gmax[s] = max(gmax[s], float(g.max()), float(u.max())); ng[s] += float((g > 1.0).sum()) / 2 + float((u > 1.0).sum())
    ever = solved.cumsum(1).bool().sum(0).cpu().numpy(); keep16 = (solved & solved[:, 15:16]).sum(0).cpu().numpy() if S >= 16 else ever
    d = dict(exact=ex.astype(int).tolist(), valid=va.astype(int).tolist(), cell=(ce / nbl).round(4).tolist(), flip=(fl / nbl).round(4).tolist(),
             ever=ever.astype(int).tolist(), keep16=keep16.astype(int).tolist(), flagged_per_puzzle=(nv / N).round(2).tolist(),
             precision=np.divide(prec, np.maximum(nv * 2, 1)).round(3).tolist(), recall=np.divide(prec, np.maximum(rec, 1)).round(3).tolist(),
             conflicts_per_puzzle=(rec / 2 / N).round(2).tolist(), g_max=gmax.round(2).tolist(), g_gt1_per_puzzle=(ng / N).round(2).tolist())
    return d


def show(name, d):
    print(f"\n[{name}]  seg:   " + "".join(f"{k:>7d}" for k in marks))
    for key, fmt in (("exact", "d"), ("valid", "d"), ("ever", "d"), ("keep16", "d"), ("cell", ".4f"), ("flip", ".4f"), ("conflicts_per_puzzle", ".2f"),
                     ("flagged_per_puzzle", ".2f"), ("precision", ".3f"), ("recall", ".3f"), ("g_max", ".1f"), ("g_gt1_per_puzzle", ".1f")):
        print(f"  {key[:12]:<12} " + "".join(f"{d[key][k - 1]:>7{fmt}}" for k in marks))
    sys.stdout.flush()


res = {}
for rho in args.rho:
    t = time.time(); d = run(rho); d["sec"] = round(time.time() - t, 1); res[f"rho{rho}"] = d; show(f"rho={rho} sig={args.sig} θ0={args.flag} cap={args.cap} warm={args.warm} decay={args.decay} uncommit={args.uncommit} spread={args.spread}", d)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(n=N, segs=S, sig=args.sig, heads=args.heads, cap=args.cap, reset=args.reset, flag=args.flag, res=res), open(args.out, "w"))
print("saved", args.out)
