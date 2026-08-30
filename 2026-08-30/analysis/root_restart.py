"""뿌리 되돌리기: 첫 결정 중 외부 지지가 가장 약한 칸(라벨 없음)의 값을 금지하고 재전개 (DPLL 분기의 한 걸음).
  뿌리 후보 = 결정 블록이 가장 이른 빈칸들 중 결정 시 외부 지지 최소.  금지 = 주입에 −δ·Wc⁺[x] (읽기 의사역행렬: 숫자 x 로짓만 −δ).
  채택 = 모델 내부 충돌 탐지(pp·relu(−Σ_{0,2,4} a) > θ0, 대각 제외) 가 0.  라벨은 채점에만.
사용: python analysis/root_restart.py [--n 512] [--rounds 3] [--delta 10] [--tabu_all 0]"""
import argparse, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--delta", type=float, default=10.0)
ap.add_argument("--segs", type=int, default=16); ap.add_argument("--flag", type=float, default=0.216); ap.add_argument("--heads", type=int, nargs="+", default=[0, 2, 4]); args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; Wc = inner.w_cls.weight[2:11].float(); Wp = Wc.T @ torch.linalg.inv(Wc @ Wc.T)      # [d,9] 의사역행렬: Wc @ Wp = I
K = 8; T = args.segs * K; OFFD = ~torch.eye(81, dtype=torch.bool, device="cuda")

def run(x, bias):
    """bias [n,81,d] 를 주입에 더해 16 세그먼트. 반환: 최종 예측, 결정 블록, 결정 시 외부 지지, 내부 충돌 수"""
    n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); ARG = torch.zeros(n, T, 81, dtype=torch.long, device="cuda"); EXT = torch.zeros(n, T, 81, device="cuda"); k = 0
    for s in range(args.segs):
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x)) + bias
            for _ in range(K):
                h = inner.boundary(h); h = h + inner.inj_gate * inj
                a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
                wm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn) if w is None else w
                h, w = inner.step(h, AB, kc, w, None, None)
                l = inner.w_cls(h).float()[:, :, 2:11]; P = l.argmax(-1)
                wf = wm.float() * OFFD[None, None]; o = torch.einsum('bhtn,bnhc->bthc', wf, vv.float()); dl = torch.einsum('bthc,hcd->btd', o, inner.w_sh.float()) @ Wc.T
                dsel = dl.gather(-1, P.unsqueeze(-1)).squeeze(-1); dl2 = dl.scatter(-1, P.unsqueeze(-1), -1e9).max(-1).values
                ARG[:, k] = P; EXT[:, k] = dsel - dl2; k += 1
        h = h.float()
    # 내부 충돌 탐지 (최종 상태)
    pr = torch.softmax(inner.w_cls(h).float()[:, :, 2:11], -1); pp = torch.einsum('btd,bnd->btn', pr, pr)
    af = inner.attn(h, AB, kc).float(); R = (-af[:, args.heads].sum(1)).clamp_min(0); viol = ((pp * R - args.flag).clamp_min(0) * OFFD[None]) > 0
    final = ARG[:, -1]; stable = (ARG == final[:, None, :]); commit = torch.clamp(T - torch.flip(stable, [1]).cummin(1).values.sum(1), max=T - 1)
    ext = EXT.gather(1, commit[:, None, :]).squeeze(1)
    return final, commit, ext, viol.sum((1, 2)) // 2

tot_solved = 0; tot_acc_wrong = 0; rows = []
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); g = G[b:b + n]; m_bl = bl[b:b + n]; bias = torch.zeros(n, 81, inner.d, device="cuda")
    final, commit, ext, nconf = run(x, bias); solved = ((final == g) | ~m_bl).all(1); accepted = nconf == 0
    ever = solved.clone(); acc_any = accepted.clone(); acc_wrong = accepted & ~solved
    r0 = dict(solved=int(solved.sum()), accepted=int(accepted.sum()), acc_wrong=int(acc_wrong.sum()))
    tabu = torch.zeros(n, 81, 9, dtype=torch.bool, device="cuda"); hist = [r0]; root_hits = []; root_wrong = []
    for rd in range(args.rounds):
        # 미채택 퍼즐만: 뿌리 후보 = 결정 블록 최소인 빈칸들 중 외부 지지 최소 (금지된 (칸,값) 은 제외)
        cm = commit.clone().float(); cm[~m_bl] = 1e9; already = tabu.gather(-1, final.unsqueeze(-1)).squeeze(-1); cm[already] = 1e9
        t0 = cm.min(1, keepdim=True).values; cand = (cm <= t0 + 0.5); score = torch.where(cand, ext, torch.full_like(ext, 1e9)); root = score.argmin(1)
        rootval = final.gather(1, root[:, None]).squeeze(1); is_wrong_root = (rootval != g.gather(1, root[:, None]).squeeze(1))
        first_wrong_block = torch.where((final != g) & m_bl, commit.float(), torch.full_like(cm, 1e9)).min(1).values
        is_true_root = is_wrong_root & (commit.gather(1, root[:, None]).squeeze(1).float() <= first_wrong_block + 0.5)
        act = ~acc_any
        root_wrong.append((int((is_wrong_root & act).sum()), int(act.sum()))); root_hits.append(int((is_true_root & act).sum()))
        tabu[torch.arange(n, device="cuda"), root, rootval] |= act
        # 금지 편향 재구성 (누적): 각 (칸, 값) 에 −δ Wp[:, 값]
        bias = -args.delta * torch.einsum('ntv,dv->ntd', tabu.float(), Wp)
        final2, commit2, ext2, nconf2 = run(x, bias); solved2 = ((final2 == g) | ~m_bl).all(1); accepted2 = nconf2 == 0
        upd = act                                                                       # 미채택이던 퍼즐만 갱신
        final = torch.where(upd[:, None], final2, final); commit = torch.where(upd[:, None], commit2, commit); ext = torch.where(upd[:, None], ext2, ext)
        new_acc = upd & accepted2; ever |= upd & solved2; acc_wrong |= new_acc & ~solved2; acc_any |= new_acc
        hist.append(dict(solved=int((upd & solved2).sum()), accepted=int(new_acc.sum()), acc_wrong=int((new_acc & ~solved2).sum()), active=int(upd.sum())))
    rows.append((hist, root_wrong, root_hits, int(ever.sum()), int(acc_any.sum()), int(acc_wrong.sum())))
    print(f"배치 {b//128}: 라운드0 완답 {r0['solved']} 채택 {r0['accepted']}(오답채택 {r0['acc_wrong']}) | " + " | ".join(f"R{i+1}: 대상 {h['active']} 완답 +{h['solved']} 채택 +{h['accepted']}(오답 {h['acc_wrong']})" for i, h in enumerate(hist[1:])) + f" || 누적 완답 {int(ever.sum())} 채택 {int(acc_any.sum())}(오답 {int(acc_wrong.sum())})")
    print("   뿌리 후보 적중: " + " | ".join(f"R{i+1} 후보가 오답칸 {rw[0]}/{rw[1]}, 진짜 뿌리(첫 오답) {rh}/{rw[1]}" for i, (rw, rh) in enumerate(zip(root_wrong, root_hits))), flush=True)
S0 = sum(r[0][0]['solved'] for r in rows); E = sum(r[3] for r in rows); A = sum(r[4] for r in rows); AW = sum(r[5] for r in rows)
print(f"\n합계 {N}: 기준 완답 {S0} → 되돌리기 {args.rounds}회 뒤 완답 {E} (+{E-S0}) | 내부 탐지 채택 {A} 중 오답 채택 {AW}")
