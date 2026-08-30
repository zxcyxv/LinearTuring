"""미해결 퍼즐의 오답 구조: 오답 칸 수 분포, 각 칸의 '결정 시점'(그 뒤로 argmax 불변인 첫 블록)과 결정 시 여유·외부 지지, 뿌리(첫 오답 결정) 뒤 파생 수.
  외부 지지 = 결정 블록에서 다른 칸들이 보낸 메시지의 (결정 숫자 로짓 − 다른 숫자 최대 로짓). 라벨은 채점에만.
사용: python analysis/failure_structure.py [--n 512] [--segs 16]"""
import argparse, os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=512); ap.add_argument("--segs", type=int, default=16); args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; Wc = inner.w_cls.weight[2:11].float(); K = 8; T = args.segs * K
ARG = torch.zeros(N, T, 81, dtype=torch.long, device="cuda"); GAP = torch.zeros(N, T, 81, device="cuda"); EXT = torch.zeros(N, T, 81, device="cuda")
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); k = 0
    for s in range(args.segs):
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for _ in range(K):
                h = inner.boundary(h); h = h + inner.inj_gate * inj
                a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
                wm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn) if w is None else None
                h, w = inner.step(h, AB, kc, w, None, None)
                l = inner.w_cls(h).float()[:, :, 2:11]; P = l.argmax(-1); t2 = l.topk(2, -1).values
                # 외부 메시지(자기 제외)의 숫자별 로짓 변화: f_ext = Σ_{n≠t} w_tn WᵀW v_n
                wf = (w if wm is None else wm).float() * (1 - torch.eye(81, device="cuda"))[None, None]
                o = torch.einsum('bhtn,bnhc->bthc', wf, vv.float()); fe = torch.einsum('bthc,hcd->btd', o, inner.w_sh.float()); dl = fe @ Wc.T   # [n,81,9]
                dsel = dl.gather(-1, P.unsqueeze(-1)).squeeze(-1); dl2 = dl.scatter(-1, P.unsqueeze(-1), -1e9).max(-1).values
                ARG[b:b + n, k] = P; GAP[b:b + n, k] = t2[..., 0] - t2[..., 1]; EXT[b:b + n, k] = dsel - dl2; k += 1
        h = h.float()
final = ARG[:, -1]; wrong = (final != G) & bl; unsolved = wrong.any(1); nw = wrong.sum(1)
stable = (ARG == final[:, None, :]); commit = T - 1 - torch.flip(stable, [1]).cummin(1).values.sum(1)     # 마지막 불변 구간의 시작 블록 (0 = 처음부터 불변)
commit = torch.clamp(commit + 1, max=T - 1)                                                                  # 불변이 시작된 블록
cg = GAP.gather(1, commit[:, None, :]).squeeze(1); ce = EXT.gather(1, commit[:, None, :]).squeeze(1)
print(f"미해결 {int(unsolved.sum())}/{N}.  오답 칸 수 분포: 1-4: {int(((nw>=1)&(nw<=4)).sum())}  5-10: {int(((nw>=5)&(nw<=10)).sum())}  11-20: {int(((nw>=11)&(nw<=20)).sum())}  21-30: {int(((nw>=21)&(nw<=30)).sum())}  31+: {int((nw>=31).sum())}")
U = unsolved[:, None]; rightb = (final == G) & bl
q = lambda v: f"중앙값 {v.median():.2f} (10% {v.quantile(.1):.2f}, 90% {v.quantile(.9):.2f})"
print(f"결정 블록(0..{T-1}): 오답 칸 {q(commit[wrong].float())} | 미해결 퍼즐의 정답 빈칸 {q(commit[rightb & U].float())} | 해결 퍼즐의 빈칸 {q(commit[rightb & ~U].float())}")
print(f"결정 시 외부 지지(결정 숫자 − 차선, 로짓): 오답 칸 {q(ce[wrong])} | 미해결 정답 빈칸 {q(ce[rightb & U])} | 해결 퍼즐 빈칸 {q(ce[rightb & ~U])}")
print(f"결정 시 여유(1등−2등): 오답 칸 {q(cg[wrong])} | 미해결 정답 빈칸 {q(cg[rightb & U])} | 해결 퍼즐 빈칸 {q(cg[rightb & ~U])}")
# 뿌리: 퍼즐별 가장 먼저 결정된 오답 칸. 그 뒤에 결정된 오답 수, 뿌리의 외부 지지, 뿌리 결정 전에 결정된 정답 칸 수
roots = []; after = []; rext = []; before_right = []; neg_ext_frac = []
for i in torch.where(unsolved)[0].tolist():
    wc = torch.where(wrong[i])[0]; t0 = commit[i, wc].min(); r = wc[commit[i, wc] == t0][0]
    roots.append(int(t0)); after.append(int((commit[i, wc] > t0).sum())); rext.append(float(ce[i, r])); before_right.append(int(((commit[i] < t0) & rightb[i]).sum()))
    neg_ext_frac.append(float((ce[i, wc] < 0).float().mean()))
roots = np.array(roots); after = np.array(after); rext = np.array(rext)
print(f"뿌리(첫 오답 결정) 블록: 중앙값 {np.median(roots):.0f} (10% {np.percentile(roots,10):.0f}, 90% {np.percentile(roots,90):.0f}) | 뿌리 뒤에 결정된 오답 수: 중앙값 {np.median(after):.0f} / 퍼즐당 오답 {nw[unsolved].float().mean():.1f}")
print(f"뿌리 결정 시 외부 지지: 중앙값 {np.median(rext):.2f} (10% {np.percentile(rext,10):.2f}, 90% {np.percentile(rext,90):.2f}) | 뿌리보다 먼저 결정된 정답 빈칸 수: 중앙값 {np.median(before_right):.0f}")
print(f"오답 칸 중 결정 시 외부 지지가 음수(메시지는 반대)인 비율: 퍼즐 평균 {np.mean(neg_ext_frac):.2f}")
# 결정 시 외부 지지로 오답을 얼마나 가르는가 (같은 퍼즐 안)
def auc(pos, neg):
    pos = pos.cpu().numpy(); neg = neg.cpu().numpy(); allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1; return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
print(f"AUC(오답 vs 같은 퍼즐 정답 빈칸): 결정 시 외부 지지(낮을수록 오답) {1-auc(ce[wrong], ce[rightb & U]):.3f} | 결정 블록(늦을수록 오답) {auc(commit[wrong].float(), commit[rightb & U].float()):.3f} | 결정 시 여유(작을수록) {1-auc(cg[wrong], cg[rightb & U]):.3f}")
