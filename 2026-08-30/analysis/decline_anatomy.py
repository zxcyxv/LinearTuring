"""퍼즐 57 고리 칸 여유의 느린 하강 — 정체 규명 (stdp1, w 세그먼트 초기화, 규칙 없음).
  여유 m_t(h) = W_cls[현재 숫자]·h_t − W_cls[정답]·h_t 는 h 에 선형 → 블록 안의 각 하위 단계(경계·주입·전달·Φ)의 기여로 정확히 분해.
  전달 기여는 보낸 칸 집단별(자기·고리·충돌상대·다른 빈칸·주어진 칸)로 다시 분해. 세그먼트별로 합산해 어느 항이 일관된 음의 부호를 갖는지 본다.
  추가: 상태 크기 ‖h_t‖ 의 표류, 가장자리 칸 튐 사건과 하강의 관계.
사용: python analysis/decline_anatomy.py [퍼즐=57] [세그먼트=120]"""
import sys, os, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); PZ = int(sys.argv[1]) if len(sys.argv) > 1 else 57; S = int(sys.argv[2]) if len(sys.argv) > 2 else 120; K = 8
inp, lab, _ = load_test(512); G = (lab - 2).long(); bl = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; b0 = (PZ // 128) * 128; x = inp[b0:b0 + 128]; i = PZ - b0; h = inner.init_hidden.expand(128, 81, -1).clone()
Wc = inner.w_cls.weight[2:11].float(); cyc = [2, 4, 20, 21]; part = [76, 12, 26]; Gp = G[PZ]; blank = bl[PZ]
groups = {"self": None, "cycle": torch.tensor(cyc, device="cuda"), "partners": torch.tensor(part, device="cuda")}
other_blank = torch.tensor([c for c in range(81) if blank[c] and c not in cyc + part], device="cuda"); given = torch.tensor([c for c in range(81) if not blank[c]], device="cuda")
T = S * K; C = {k: np.zeros((T, 4)) for k in ("boundary", "inject", "transport", "phi", "t_self", "t_cycle", "t_partners", "t_otherblank", "t_given")}
MARG = np.zeros((T, 4)); NORM = np.zeros((T, 4)); PER = np.zeros((T, 3), int); CUR = np.zeros((T, 4), int)
def margin(hh, cur):    # [4]: 현재 숫자 로짓 − 정답 로짓 (cur: 현재 argmax 인덱스), float32 로
    with torch.autocast("cuda", enabled=False):
        lg = hh[i, cyc].float() @ Wc.T; return (lg.gather(-1, cur[:, None]).squeeze(-1) - lg[torch.arange(4, device="cuda"), Gp[cyc]]).float()
k = 0
for s in range(S):
    w = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
        for _ in range(K):
            cur = (inner.w_cls(h).float()[i, cyc, 2:11]).argmax(-1); m0 = margin(h, cur)
            h1 = inner.boundary(h); m1 = margin(h1, cur); h2 = h1 + inner.inj_gate * inj; m2 = margin(h2, cur)
            a = inner.attn(h2, AB, kc); vv = torch.einsum('btd,hcd->bthc', h2, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
            wm = a * torch.einsum('bthc,bnhc->bhtn', vn, vn) if w is None else w + torch.sigmoid(inner.eta_raw) * (a * torch.einsum('bthc,bnhc->bhtn', vn, vn) - w)
            # 전달 f 를 보낸 칸별로 (퍼즐 i, 대상 4칸만)
            with torch.autocast("cuda", enabled=False):
                f_by = torch.zeros(4, 81, 832, device="cuda")
                for j, t in enumerate(cyc):
                    o = torch.einsum('hn,nhc->nhc', wm[i, :, t, :].float(), vv[i].float()); f_by[j] = torch.einsum('nhc,hcd->nd', o, inner.w_sh.float())
                dl = torch.einsum('jnd,vd->jnv', f_by, Wc)                                      # [4,81,9] 보낸 칸별 로짓 변화
                dm = dl.gather(-1, cur[:, None, None].expand(-1, 81, 1)).squeeze(-1) - dl[torch.arange(4, device="cuda"), :, Gp[cyc]]   # [4,81] 여유 변화
            for j, t in enumerate(cyc):
                C["t_self"][k, j] = float(dm[j, t]); C["t_cycle"][k, j] = float(dm[j, [c for c in cyc if c != t]].sum()); C["t_partners"][k, j] = float(dm[j, part].sum())
                C["t_otherblank"][k, j] = float(dm[j, other_blank].sum()); C["t_given"][k, j] = float(dm[j, given].sum())
            h3, w = inner.step(h2, AB, kc, w, None, None)                                        # step = 전달 + Φ
            # 전달만의 상태: h2 + f (Φ 전) 를 다시 구성
            with torch.autocast("cuda", enabled=False):
                f_all = torch.einsum('bhtn,bnhc->bthc', wm[i:i+1].float(), vv[i:i+1].float()); f_all = torch.einsum('bthc,hcd->btd', f_all, inner.w_sh.float())
                hf = h2[i:i+1].float() + f_all; lg3 = hf[0, cyc] @ Wc.T; m3 = (lg3.gather(-1, cur[:, None]).squeeze(-1) - lg3[torch.arange(4, device="cuda"), Gp[cyc]]).float()
            m4 = margin(h3, cur)
            C["boundary"][k] = (m1 - m0).cpu().numpy(); C["inject"][k] = (m2 - m1).cpu().numpy(); C["transport"][k] = (m3 - m2).cpu().numpy(); C["phi"][k] = (m4 - m3).cpu().numpy()
            MARG[k] = m4.cpu().numpy(); NORM[k] = h3[i, cyc].float().norm(dim=-1).cpu().numpy(); CUR[k] = cur.cpu().numpy()
            PER[k] = inner.w_cls(h3).float()[i, part, 2:11].argmax(-1).cpu().numpy(); h = h3; k += 1
    h = h.float()
np.savez_compressed(f"results/json/decline_anatomy_{PZ}.npz", MARG=MARG, NORM=NORM, PER=PER, CUR=CUR, **C)
# ---- 분석: 세그먼트 합
seg = lambda A: A.reshape(S, K, -1).sum(1)
names = ["boundary", "inject", "transport", "phi"]; tr = ["t_self", "t_cycle", "t_partners", "t_otherblank", "t_given"]
print(f"세그먼트당 여유 변화의 항별 합 (20~110 세그먼트 평균, 칸 2/4/20/21). 여유 = 현재(틀린) 숫자 로짓 − 정답 로짓. 음수 = 정답 쪽으로 밈")
tot = seg(MARG[:, :] - np.vstack([MARG[:1], MARG[:-1]]))[20:110].mean(0)
print("  실제 세그먼트당 여유 변화 (합):", np.round(tot, 3))
for nm in names: print(f"  {nm:<10s}", np.round(seg(C[nm])[20:110].mean(0), 3), " | 부호 일관성(세그먼트 중 음수 비율):", np.round((seg(C[nm])[20:110] < 0).mean(0), 2))
print("  전달을 보낸 집단별:")
for nm in tr: print(f"    {nm:<13s}", np.round(seg(C[nm])[20:110].mean(0), 3), " | 음수 비율:", np.round((seg(C[nm])[20:110] < 0).mean(0), 2))
print("\n상태 크기 ‖h_t‖ (세그먼트 끝): seg20", np.round(NORM[20*K-1], 1), " seg60", np.round(NORM[60*K-1], 1), " seg100", np.round(NORM[100*K-1], 1), " seg110", np.round(NORM[110*K-1], 1))
# 가장자리 튐 사건과 하강
per_seg = PER.reshape(S, K, 3); flick = np.array([(per_seg[s] != per_seg[s][-1]).any() or (s > 0 and (per_seg[s][-1] != per_seg[s-1][-1]).any()) for s in range(S)])
dM = seg(MARG - np.vstack([MARG[:1], MARG[:-1]]))
print(f"\n가장자리(76·12·26) 튐이 있던 세그먼트 수 (20~110): {flick[20:110].sum()} / 90")
print("  튐 세그먼트의 여유 변화 평균:", np.round(dM[20:110][flick[20:110]].mean(0), 3), " | 튐 없는 세그먼트:", np.round(dM[20:110][~flick[20:110]].mean(0), 3))
