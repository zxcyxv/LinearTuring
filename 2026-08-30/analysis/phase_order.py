# 위상 순서(반대칭 결합) 가 결정 순서를 기록하는가. 라벨은 채점에만.
import os, sys, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis')); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", "core/minimal.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); inp, lab, _ = load_test(512); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
ck = torch.load("checkpoints/R1B8_min_stdp1_step228501.pt", map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
K = 8; T = 16 * K; OFFD = ~torch.eye(81, dtype=torch.bool, device="cuda"); psi = inner.psi.detach().float().flatten().cpu().numpy() if hasattr(inner, "psi") else None
print("학습된 ψ (헤드별):", np.round(psi, 2) if psi is not None else "속성 없음", "| 기준 π/2 =", round(np.pi/2, 2))
ARG = torch.zeros(N, T, 81, dtype=torch.long, device="cuda"); A16 = torch.zeros(N, 8, 81, 81, device="cuda"); A1 = torch.zeros(N, 8, 81, 81, device="cuda")
for b in range(0, N, 128):
    x = inp[b:b + 128]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); k = 0
    for s in range(16):
        w = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
            for _ in range(K):
                h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
                ARG[b:b+n, k] = inner.w_cls(h).float()[:, :, 2:11].argmax(-1); k += 1
            if s == 0: A1[b:b+n] = inner.attn(h.float(), AB, kc).float()
        h = h.float()
    A16[b:b+n] = inner.attn(h, AB, kc).float()
final = ARG[:, -1]; wrong = (final != G) & bl; uns = wrong.any(1); stable = (ARG == final[:, None, :]); commit = torch.clamp(T - torch.flip(stable, [1]).cummin(1).values.sum(1), max=T - 1)
first_wrong = torch.where(wrong, commit.float(), torch.full_like(commit.float(), 1e9)).min(1).values; idx = torch.where(uns)[0]; ar = torch.arange(N, device="cuda")
earlier = (commit[:, None, :] < commit[:, :, None])                 # earlier[t,n]: n 이 t 보다 먼저 결정
later = (commit[:, None, :] > commit[:, :, None])
def auc(pos, neg):
    pos = pos.cpu().numpy(); neg = neg.cpu().numpy()
    if len(pos) > 200000: pos = np.random.default_rng(0).choice(pos, 200000, replace=False)
    if len(neg) > 200000: neg = np.random.default_rng(1).choice(neg, 200000, replace=False)
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1; return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
for name, A in (("세그먼트 1 끝", A1), ("세그먼트 16 끝", A16)):
    anti = A - A.transpose(-1, -2)                                   # anti[h,t,n] = a_tn − a_nt
    print(f"\n[{name}] 반대칭 부분 a_tn − a_nt 가 'n 이 t 보다 먼저 결정' 을 예측하는 AUC (피어 쌍 / 비피어 쌍), 헤드별:")
    for hh in range(8):
        v = anti[:, hh]; mask_p = pm[None] & OFFD[None]; mask_n = ~pm[None] & OFFD[None]
        print(f"   헤드 {hh}: 피어 {auc(v[earlier & mask_p], v[later & mask_p]):.3f}   비피어 {auc(v[earlier & mask_n], v[later & mask_n]):.3f}   |a_tn−a_nt| 평균 {v.abs().mean():.3f}")
    # 순 pre-ness: s_t = Σ_n (a_nt − a_tn) (t 가 n 들에게 '먼저' 인 정도), 헤드 합
    s = -anti.sum(1).sum(-1)                                         # Σ_n (a_nt − a_tn)
    print("   순 pre-ness 가 결정 블록과 갖는 순위상관(퍼즐 내, 빈칸):", end=" ")
    cors = []
    for i in range(0, N, 4):
        v = s[i][bl[i]].cpu().numpy(); c = commit[i][bl[i]].float().cpu().numpy()
        if len(v) > 5: cors.append(np.corrcoef(v.argsort().argsort(), c.argsort().argsort())[0, 1])
    print(f"{np.nanmean(cors):+.3f}")
    for sign in (1, -1):
        sc = torch.where(bl, sign * s, torch.full_like(s, -1e9)); c = sc.argmax(1)
        isw = wrong[ar, c]; ist = isw & (commit[ar, c].float() <= first_wrong + 0.5)
        print(f"   순 pre-ness {'최대' if sign==1 else '최소'} 칸 → 오답 {float(isw[idx].float().mean()):.2f}  진짜 뿌리 {float(ist[idx].float().mean()):.2f}   (그 칸의 결정 블록 중앙값 {float(commit[ar, c][idx].float().median()):.0f})")
