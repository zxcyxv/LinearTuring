"""세그먼트 사상 F(h) (8 블록, w 초기화, float32) 의 야코비안 스펙트럼 — 갇힌 상태와 풀린 상태에서 (퍼즐 57).
  J v 는 forward-mode 자동미분(torch.func.jvp), 고유값은 ARPACK(Arnoldi). 그리고 실제 탈출 변위(bf16 궤적 H, 세그 111→115)를 고유벡터에 사영.
사용: python analysis/jacobian_spectrum.py [퍼즐=57] [k=24]"""
import sys, os, importlib.util, numpy as np, torch, scipy.sparse.linalg as sla
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
PZ = int(sys.argv[1]) if len(sys.argv) > 1 else 57; KEIG = int(sys.argv[2]) if len(sys.argv) > 2 else 24; K = 8; SEG0 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
inp, lab, _ = load_test(512); G = (lab - 2).long(); bl = inp == 1
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=1, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=False)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
for p in m.parameters(): p.requires_grad_(False)
x = inp[PZ:PZ + 1]
with torch.no_grad(): AB = inner.W_C(); kc = inner.kernel(); kcb = inner.kernel(inner.beta); inj = inner.injection(make_batch(x, x)).float()
AB = tuple(t.detach() for t in AB); kc = tuple(t.detach() for t in kc); kcb = tuple(t.detach() for t in kcb)
def F(h):
    """세그먼트 사상 (float32, w 초기화). h [1,81,832]"""
    w = None
    for _ in range(K):
        h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, kcb)
    return h
Wc = inner.w_cls.weight[2:11].float().detach(); D = 81 * 832
H = np.load(os.path.join(ROOT, "results", "json", f"dyn_macro_{PZ}_H.npy"))                  # bf16 궤적, 블록별
def state_at(seg): return torch.tensor(H[seg * K - 1].reshape(1, 81, 832), device="cuda", dtype=torch.float32)
def report_state(h, name):
    with torch.no_grad():
        P = (h[0] @ Wc.T).argmax(-1); wrong = ((P != G[PZ]) & bl[PZ]); h2 = F(h); drift = (h2 - h).norm() / h.norm()
    print(f"[{name}] 오답 칸 {int(wrong.sum())} {torch.where(wrong)[0].tolist()} | 세그먼트 사상 한 번의 상대 변위 ‖F(h)−h‖/‖h‖ = {float(drift):.4f}")
def spectrum(h0, name):
    h0 = h0.clone()
    def matvec(v):
        vt = torch.tensor(np.asarray(v, dtype=np.float32).reshape(1, 81, 832), device="cuda")
        _, jv = torch.func.jvp(F, (h0,), (vt,)); return jv.detach().reshape(-1).cpu().numpy().astype(np.float64)
    op = sla.LinearOperator((D, D), matvec=matvec, dtype=np.float64)
    lam, V = sla.eigs(op, k=KEIG, which="LM", tol=1e-4, maxiter=4000, ncv=max(2 * KEIG + 1, 60))
    order = np.argsort(-np.abs(lam)); lam = lam[order]; V = V[:, order]
    print(f"[{name}] 상위 |λ| (세그먼트당):", " ".join(f"{abs(l):.4f}{'' if abs(l.imag) < 1e-3 else 'i'}" for l in lam[:16]))
    real = [q for q in range(len(lam)) if abs(lam[q].imag) < 1e-3]; print(f"    실수 고유값: " + " ".join(f"{lam[q].real:+.4f}" for q in real[:8]))
    for q in list(range(min(4, len(lam)))):
        amp = np.abs(V[:, q]).reshape(81, 832).sum(1); top = np.argsort(amp)[::-1][:8]; print(f"    모드 {q} |λ|={abs(lam[q]):.4f} 각={np.angle(lam[q]):+.2f}: 진폭 상위 칸 {top.tolist()}  (고리 2,4,20,21 / 상대 76,12,26)  칸 에너지 집중도 {(np.sort(amp)[::-1][:8].sum()/amp.sum()):.2f}")
    return lam, V
h_stuck = state_at(SEG0); h_solved = state_at(120)
report_state(h_stuck, f"갇힘 seg{SEG0}"); report_state(h_solved, "풀림 seg120")
lam_s, V_s = spectrum(h_stuck, f"갇힘 seg{SEG0}"); lam_f, V_f = spectrum(h_solved, "풀림 seg120")
# 탈출 변위를 갇힘 고유벡터에 사영
d = (H[115 * K - 1] - H[111 * K - 1]).astype(np.float64); d = d / np.linalg.norm(d)
steps = {f"{s}→{s+1}": (H[(s + 1) * K - 1] - H[s * K - 1]).astype(np.float64) for s in (111, 112, 113, 114)}
def proj_frac(vec, V, idx):
    B = np.concatenate([V[:, idx].real, V[:, idx].imag], 1); Q, _ = np.linalg.qr(B); c = Q.T @ vec; return float(np.linalg.norm(c) ** 2 / np.linalg.norm(vec) ** 2)
print("\n탈출 변위(111→115)의 에너지 중 갇힘 야코비안 고유벡터 부분공간이 설명하는 비율:")
for kk in (1, 2, 4, 8, 16, KEIG): print(f"    상위 {kk:>2d} 모드: {proj_frac(d, V_s, list(range(kk))):.3f}")
soft = [q for q in range(len(lam_s)) if abs(lam_s[q]) < 1.0]; print(f"    |λ|<1 중 가장 무른 모드(|λ| 최대 = {abs(lam_s[soft[0]]) if soft else float('nan'):.4f}) 단독: {proj_frac(d, V_s, [soft[0]]) if soft else float('nan'):.3f}")
print("    무작위 24차원 부분공간(기준):", round(float(np.mean([proj_frac(d, np.random.default_rng(i).standard_normal((D, KEIG)) + 0j, list(range(KEIG))) for i in range(3)])), 4))
for nm, st in steps.items(): st = st / np.linalg.norm(st); print(f"    세그먼트 {nm} 변위: 상위 4 모드 {proj_frac(st, V_s, [0,1,2,3]):.3f}  상위 {KEIG} 모드 {proj_frac(st, V_s, list(range(KEIG))):.3f}")
np.savez_compressed(os.path.join(ROOT, "results", "json", f"jacobian_{PZ}_seg{SEG0}.npz"), lam_stuck=lam_s, lam_solved=lam_f, V_stuck_top8=V_s[:, :8], V_solved_top4=V_f[:, :4])
