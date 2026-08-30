# 인과 추적으로 뿌리 찾기 (라벨은 채점에만): 의존 간선 n→t = (n 이 t 보다 먼저 결정) & (|a_tn| 상위 k), 충돌 칸들의 조상 중 이른 칸에서 충돌 후손 수 최대인 칸
import os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis')); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", "core/minimal.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); inp, lab, _ = load_test(512); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load("checkpoints/R1B8_min_stdp1_step228501.pt", map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; Wc = inner.w_cls.weight[2:11].float(); K = 8; T = 16 * K; OFFD = ~torch.eye(81, dtype=torch.bool, device="cuda"); HEADS = [0, 2, 4]
ARG = torch.zeros(N, T, 81, dtype=torch.long, device="cuda"); EXT = torch.zeros(N, T, 81, device="cuda"); AMAG = torch.zeros(N, 81, 81, device="cuda"); VIOL = torch.zeros(N, 81, 81, dtype=torch.bool, device="cuda")
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
                l = inner.w_cls(h).float()[:, :, 2:11]; P = l.argmax(-1)
                wf = wm.float() * OFFD[None, None]; o = torch.einsum('bhtn,bnhc->bthc', wf, vv.float()); dl = torch.einsum('bthc,hcd->btd', o, inner.w_sh.float()) @ Wc.T
                dsel = dl.gather(-1, P.unsqueeze(-1)).squeeze(-1); dl2 = dl.scatter(-1, P.unsqueeze(-1), -1e9).max(-1).values
                ARG[b:b+n, k] = P; EXT[b:b+n, k] = dsel - dl2; k += 1
        h = h.float()
    af = inner.attn(h, AB, kc).float(); AMAG[b:b+n] = af.abs().mean(1)
    pr = torch.softmax(inner.w_cls(h).float()[:, :, 2:11], -1); pp = torch.einsum('btd,bnd->btn', pr, pr); R = (-af[:, HEADS].sum(1)).clamp_min(0)
    VIOL[b:b+n] = ((pp * R - 0.216) > 0) & OFFD[None]
final = ARG[:, -1]; wrong = (final != G) & bl; uns = wrong.any(1); stable = (ARG == final[:, None, :]); commit = torch.clamp(T - torch.flip(stable, [1]).cummin(1).values.sum(1), max=T - 1)
ext_c = EXT.gather(1, commit[:, None, :]).squeeze(1); first_wrong = torch.where(wrong, commit.float(), torch.full_like(commit.float(), 1e9)).min(1).values
idx = torch.where(uns)[0]; ar = torch.arange(N, device="cuda")
def report(name, cand):
    isw = wrong[ar, cand]; istrue = isw & (commit[ar, cand].float() <= first_wrong + 0.5)
    print(f"  {name:<52s} 후보=오답 {float(isw[idx].float().mean()):.2f}   후보=진짜 뿌리 {float(istrue[idx].float().mean()):.2f}")
for k_nb in (8, 20):
    # 의존 간선: n→t if commit_n < commit_t and n ∈ |a_t·| 상위 k_nb. 조상 폐포.
    topk = AMAG.topk(k_nb, -1).indices; nb = torch.zeros(N, 81, 81, dtype=torch.bool, device="cuda").scatter_(-1, topk, True)     # nb[t,n]: n 은 t 의 이웃
    earlier = commit[:, None, :] < commit[:, :, None]                                                                             # earlier[t,n]: n 이 t 보다 먼저
    D = (nb & earlier).float()                                                                                                     # D[t,n] = n→t
    Cl = D.clone()
    for _ in range(7): Cl = ((Cl + Cl @ Cl) > 0).float()                                                                           # 조상 폐포 Cl[t,e]: e 는 t 의 조상
    for conf_name, conf in (("내부탐지 충돌", VIOL), ("(참고) 실제 피어충돌", (final[:, :, None] == final[:, None, :]) & pm[None] & OFFD[None])):
        cc = conf.any(-1).float()                                                                                                  # 충돌 칸
        score = torch.einsum('bte,bt->be', Cl, cc) + cc                                                                            # e 의 충돌 후손 수 (자기 포함)
        for B in (1, 3, 7):
            early = (commit <= B) & bl; sc = torch.where(early, score, torch.full_like(score, -1)); sc2 = sc - 1e-3 * ext_c.clamp(-10, 10)   # 동률이면 지지 약한 쪽
            report(f"k={k_nb} {conf_name} 결정≤{B} 조상 중 충돌후손 최다", sc2.argmax(1))
        # 이른 칸 제한 없이: 조상 폐포에서 가장 많은 충돌 후손을 가진 칸 (자기 후손 수로 정규화 안 함)
        sc = torch.where(bl, score, torch.full_like(score, -1)); report(f"k={k_nb} {conf_name} 전체 빈칸 중 충돌후손 최다", sc.argmax(1))
print("기준: 무작위 빈칸 → 오답 0.51 / 진짜 뿌리 0.03,  결정≤1 지지 최소 → 0.26 / 0.19")
