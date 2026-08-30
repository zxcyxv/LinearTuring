# 뿌리 후보 규칙 비교 (라벨은 채점에만): 미해결 퍼즐에서 후보가 오답 칸일 확률 / 진짜 뿌리(첫 오답)일 확률
import os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis')); from common import ROOT, URM, load_test, make_batch
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", "core/minimal.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); inp, lab, _ = load_test(512); N = len(inp); G = (lab - 2).long(); bl = inp == 1
ck = torch.load("checkpoints/R1B8_min_stdp1_step228501.pt", map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; Wc = inner.w_cls.weight[2:11].float(); K = 8; T = 16 * K; OFFD = ~torch.eye(81, dtype=torch.bool, device="cuda")
ARG = torch.zeros(N, T, 81, dtype=torch.long, device="cuda"); EXT = torch.zeros(N, T, 81, device="cuda"); GAP = torch.zeros(N, T, 81, device="cuda")
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); k = 0
    for s in range(16):
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for _ in range(K):
                h = inner.boundary(h); h = h + inner.inj_gate * inj
                a = inner.attn(h, AB, kc); vv = torch.einsum('btd,hcd->bthc', h, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
                wm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn) if w is None else w
                h, w = inner.step(h, AB, kc, w, None, None)
                l = inner.w_cls(h).float()[:, :, 2:11]; P = l.argmax(-1); t2 = l.topk(2, -1).values
                wf = wm.float() * OFFD[None, None]; o = torch.einsum('bhtn,bnhc->bthc', wf, vv.float()); dl = torch.einsum('bthc,hcd->btd', o, inner.w_sh.float()) @ Wc.T
                dsel = dl.gather(-1, P.unsqueeze(-1)).squeeze(-1); dl2 = dl.scatter(-1, P.unsqueeze(-1), -1e9).max(-1).values
                ARG[b:b+n, k] = P; EXT[b:b+n, k] = dsel - dl2; GAP[b:b+n, k] = t2[..., 0] - t2[..., 1]; k += 1
        h = h.float()
final = ARG[:, -1]; wrong = (final != G) & bl; uns = wrong.any(1); stable = (ARG == final[:, None, :]); commit = torch.clamp(T - torch.flip(stable, [1]).cummin(1).values.sum(1), max=T - 1)
ext_c = EXT.gather(1, commit[:, None, :]).squeeze(1); ext_min4 = torch.stack([EXT.gather(1, torch.clamp(commit + j, max=T-1)[:, None, :]).squeeze(1) for j in range(4)], 0).min(0).values
ext_mean = EXT.mean(1); gap_c = GAP.gather(1, commit[:, None, :]).squeeze(1)
first_wrong = torch.where(wrong, commit.float(), torch.full_like(commit.float(), 1e9)).min(1).values
idx = torch.where(uns)[0]
print(f"미해결 {len(idx)}. 뿌리(첫 오답) 결정 블록 분포: " + ", ".join(f"≤{B}: {int((first_wrong[idx] <= B).sum())}" for B in (0, 1, 3, 7, 15, 31)))
print("이른 블록에 결정된 빈칸 수(퍼즐당 중앙값): " + ", ".join(f"≤{B}: {float(((commit <= B) & bl)[idx].sum(1).float().median()):.0f}" for B in (0, 1, 3, 7, 15)))
def evalrule(name, score, cand):
    sc = torch.where(cand & bl, score, torch.full_like(score, 1e9)); c = sc.argmin(1)
    isw = wrong[torch.arange(N, device="cuda"), c]; istrue = isw & (commit[torch.arange(N, device="cuda"), c].float() <= first_wrong + 0.5)
    print(f"  {name:<44s} 후보=오답 {float(isw[idx].float().mean()):.2f}   후보=진짜 뿌리 {float(istrue[idx].float().mean()):.2f}")
for B in (0, 1, 3, 7, 15, 127):
    evalrule(f"결정≤{B} 중 결정시 지지 최소", ext_c, commit <= B)
    evalrule(f"결정≤{B} 중 결정후 4블록 지지 최소", ext_min4, commit <= B)
evalrule("결정≤3 중 결정시 여유 최소", gap_c, commit <= 3); evalrule("결정≤7 중 전체 평균 지지 최소", ext_mean, commit <= 7); evalrule("전체 중 전체 평균 지지 최소", ext_mean, commit <= 127)
evalrule("무작위 빈칸(기준)", torch.rand(N, 81, device="cuda"), commit <= 127)
print(f"참고: 오답 칸 비율(무작위 기대) = {float(wrong[idx].sum() / bl[idx].sum()):.2f}")
