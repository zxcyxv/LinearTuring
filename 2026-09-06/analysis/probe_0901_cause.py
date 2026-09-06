"""9/1 R1B8_min_faith@310527 외삽 이득의 원인 — 추론 시점 절제 + w 구조.

변형 (mode):
  base       원본
  freeze16   seg>=16 에서 w 갱신 정지 (읽기는 그대로)
  lam0       a_eff = a  (w 를 전달에서 뺌)
  G_a        쓰기 목표 G = a          (v2.1 의 Γ=a 와 같은 쓰기; β·agree 둘 다 제거)
  G_abeta    쓰기 목표 G = a_β        (agree 제거)
  G_a_agree  쓰기 목표 G = a·agree    (β 제거)
  seg_reset  세그먼트 첫 블록에서 w ← tgt (세그먼트 간 기억 제거)
  delta_init     모든 헤드 δ=0.05 (초기값; 9/1@10k 의 값) — 학습된 시정수 분화를 되돌림
  freeze16_slow  seg>=16 에서 느린 헤드(δ<0.01)만 w 동결 — 느린 적분기가 이득을 나르는가
  freeze16_fast  seg>=16 에서 빠른 헤드(δ>=0.01)만 w 동결
  G_neg_a_agree  쓰기 목표 G = −a·agree  (β 커널 없이 부호만 하드코딩 — β 의 역할이 −1 뿐인가)
  G_neg_a        쓰기 목표 G = −a
  delta_half     느린 헤드 δ 를 절반으로 (시정수 2배 — 외삽이 더 멀리 가는가)
  delta_x2       느린 헤드 δ 를 2배로
             ※ 기존 w_memory_ablation_0901.py 의 seg_reset 은 LT.forward 가 carry.fresh 를 halted 로 덮어써서 무효였다.
               여기서는 step 안에서 블록 카운터로 직접 초기화한다.

기록: 세그먼트별 퍼즐 해결 마스크 [SEG,NB], 셀 정확도, churn (비단서 칸 예측 변경률).
base 에서는 seg16·seg128 끝에서 w·a·a_eff 를 쌍 종류(피어/비피어 × 같은숫자/다른숫자)별로 집계.

사용: python probe_0901_cause.py [--modes base,freeze16,...] [--nb 512] [--seg 128] [--out DIR]
"""
import os, sys, argparse, importlib.util, time, yaml, numpy as np, torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--modes", default="base,freeze16,lam0,G_a,G_abeta,G_a_agree,seg_reset,delta_init,freeze16_slow,freeze16_fast")
ap.add_argument("--nb", type=int, default=512)
ap.add_argument("--seg", type=int, default=128)
ap.add_argument("--bsz", type=int, default=64)
ap.add_argument("--out", default="/workspace/LinearTuring/2026-09-06/results")
ap.add_argument("--tag", default="probe_0901")
ap.add_argument("--ckpt", default="", help="train_0901.py 형식 체크포인트 (예: run_0901/milestones/step_10000.pt). 비우면 9/1@310527")
args = ap.parse_args()

ROOT = "/workspace/LinearTuring"; HIST = os.path.join(ROOT, "refs", "ckpt_hist")
torch.set_grad_enabled(False)
BSZ, NB, SEG = args.bsz, args.nb, args.seg
os.makedirs(args.out, exist_ok=True)

# ---- 모델 (minimal_0901.py + R1B8_min_faith@310527, EMA 가 아니라 raw+model_state — 기존 절제 스크립트와 동일 로딩)
sys.path.insert(0, HIST)
s = importlib.util.spec_from_file_location("minimal", os.path.join(HIST, "minimal_0901.py")); mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod)
cfg = yaml.safe_load(open(os.path.join(HIST, "R1B8_min_faith_config.yaml")))["arch"]; cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
cfg.update(batch_size=BSZ, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, loops=SEG + 1)
if args.ckpt:
    # train_0901.py 형식: model.inner.layers.0.X -> inner.X, embed_scale -> inj_gate. EMA(ema_shadow) 를 model_state_dict 위에 덮음 (마일스톤 외삽과 동일)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    full = dict(ck["model_state_dict"]); full.update(ck.get("ema_shadow", {}))
    def _remap(k):
        k = k.replace("_orig_mod.", "")
        k = k[len("model."):] if k.startswith("model.") else k
        k = k.replace("inner.layers.0.", "inner.").replace("inner.embed_scale", "inner.inj_gate")
        return k
    full = {_remap(k): v for k, v in full.items()}
    assert ck["cfg"]["num_layers"] == 1 and ck["cfg"]["block_order"] == "pre" and ck["cfg"]["legacy_gauge"], ck["cfg"]
else:
    ck = torch.load(os.path.join(HIST, "R1B8_min_faith_step310527.pt"), map_location="cpu", weights_only=False)
    full = dict(ck["raw_model_state_dict"]); full.update(ck["model_state_dict"])
    full = {k.replace("_orig_mod.model.", ""): v for k, v in full.items()}
m = mod.LT(cfg).cuda().eval()
missing = m.load_state_dict(full, strict=False)
I = m.inner
print(f"[load] step={ck.get('step')}  missing={list(missing.missing_keys)}  unexpected={len(missing.unexpected_keys)}", flush=True)
assert cfg["stdp_target"] == "faithful" and cfg["addr_dim"] == 0 and not cfg["gate"]

# ---- 데이터
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
X = z["test_inputs"][:NB].reshape(NB, 81).astype(int); Y = z["test_labels"][:NB].reshape(NB, 81).astype(int)

# ---- 피어 마스크 [T,T]
T = 81; r = torch.arange(T) // 9; c = torch.arange(T) % 9; b = (r // 3) * 3 + (c // 3)
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (b[:, None] == b[None])) & ~torch.eye(T, dtype=torch.bool)
peer = peer.cuda(); offdiag = ~torch.eye(T, dtype=torch.bool, device="cuda")

# ---- 학습된 스칼라
lam = torch.sigmoid(I.lam_raw).flatten().tolist(); eta = torch.sigmoid(I.eta_raw).flatten().tolist(); gain = F.softplus(I.gain_raw).flatten().tolist()
print(f"[scalars] λ={[round(x,3) for x in lam]}\n          δ={[round(x,4) for x in eta]}\n          gain={[round(x,3) for x in gain]}"
      f"\n          inj_gate={float(I.inj_gate):.3f}  γ={float(I.gamma):.4g}  α={[round(x,3) for x in I.alpha.flatten().tolist()]}", flush=True)

# ---- step 치환 (원본 minimal_0901.py:step 의 faithful 경로를 mode 분기로 다시 씀)
mode = {"m": "base", "seg": 0, "blk": 0}
SLOW = (torch.sigmoid(I.eta_raw) < 0.01).view(1, -1, 1, 1)          # [1,H,1,1] 느린 헤드 마스크
print(f"[slow heads] δ<0.01: {SLOW.flatten().int().tolist()}", flush=True)
def step_mode(h, AB, kc, w=None, fresh=None, kcb=None, vprev=None):
    md = mode["m"]
    xy = I.addr(h, AB); a = I.attn_xy(xy, kc)
    v = torch.einsum('btd,hcd->bthc', h, I.w_sh)
    eta_ = torch.sigmoid(I.eta_raw); lam_ = torch.sigmoid(I.lam_raw)
    if md == "delta_init": eta_ = torch.full_like(eta_, 0.05)
    elif md == "delta_half": eta_ = torch.where(SLOW.view(-1, 1, 1), eta_ * 0.5, eta_)
    elif md == "delta_x2":   eta_ = torch.where(SLOW.view(-1, 1, 1), eta_ * 2.0, eta_)
    vv = v / (v.norm(dim=-1, keepdim=True) + I.config.eps)
    agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
    ab = I.attn_xy(xy, kcb)
    if md == "G_neg_a_agree": G = -a * agree
    elif md == "G_neg_a":   G = -a
    elif md == "G_a":       G = a
    elif md == "G_abeta":   G = ab
    elif md == "G_a_agree": G = a * agree
    else:                   G = ab * agree                       # faithful (원본)
    tgt = F.softplus(I.gain_raw) * G
    if w is None or (md == "seg_reset" and mode["blk"] == 0):
        w = tgt
    else:
        if fresh is not None: w = torch.where(fresh.view(-1, 1, 1, 1), tgt, w)
        if md in ("freeze16_slow", "freeze16_fast") and mode["seg"] >= 16:
            frz = SLOW if md == "freeze16_slow" else ~SLOW
            w = torch.where(frz, w, (1 - eta_) * w + eta_ * tgt)
        elif not (md == "freeze16" and mode["seg"] >= 16):
            w = (1 - eta_) * w + eta_ * tgt
    a_eff = a if md == "lam0" else (1 - lam_) * a + lam_ * w
    o = torch.einsum('bhtn,bnhc->bthc', a_eff, v); f = torch.einsum('bthc,hcd->btd', o, I.w_sh)
    mode["blk"] = (mode["blk"] + 1) % I.config.blocks_per_seg
    return I.phi(h + f), w
I.step = step_mode

def pair_stats(w, a, a_eff, pred, given):
    """[B,H,T,T] 텐서들을 쌍 종류별로 집계. 비단서 칸 쌍만 (두 칸 다 비단서)."""
    same = (pred[:, :, None] == pred[:, None, :])                      # [B,T,T]
    ng = ~given; both_ng = ng[:, :, None] & ng[:, None, :]             # [B,T,T]
    out = {}
    for pname, pm in (("peer", peer), ("nonpeer", ~peer & offdiag)):
        for sname, sm in (("same", same), ("diff", ~same)):
            mk = (pm[None] & sm & both_ng)[:, None].expand_as(w)        # [B,H,T,T]
            n = int(mk.sum())
            if n == 0: continue
            out[f"{pname}_{sname}"] = dict(n=n, w=float(w[mk].mean()), a=float(a[mk].mean()), a_eff=float(a_eff[mk].mean()),
                                           w_neg=float((w[mk] < 0).float().mean()), a_neg=float((a[mk] < 0).float().mean()))
    return out

def run(md):
    mode["m"] = md; mode["blk"] = 0
    solved = np.zeros((SEG, NB), dtype=bool); cellacc = np.zeros(SEG); churn = np.zeros(SEG); nchurn = 0
    pstats = {}
    t0 = time.time()
    AB = I.W_C(); kc = I.kernel()
    for bi in range(0, NB, BSZ):
        x = torch.from_numpy(X[bi:bi+BSZ].astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y[bi:bi+BSZ].astype(np.int32) + 1).cuda().long()
        given = x > 1; ngiven = int((~given).sum())
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(len(x), dtype=torch.int32, device="cuda"))
        prev = None
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
            carry = m.initial_carry(batch)
            for s_ in range(SEG):
                mode["seg"] = s_; mode["blk"] = 0
                carry, out = m(carry, batch)
                pred = out["logits"].argmax(-1)
                ok = pred == y
                solved[s_, bi:bi+len(x)] = ok.all(-1).cpu().numpy()
                cellacc[s_] += float(ok[~given].float().sum())
                if prev is not None: churn[s_] += float(((pred != prev) & ~given).float().sum())
                prev = pred
                if md == "base" and (s_ + 1) in (16, SEG):
                    h = carry.current_hidden; w = carry.coupling.float()
                    a = I.attn(h, AB, kc).float()
                    lam_t = torch.sigmoid(I.lam_raw).float()
                    a_eff = (1 - lam_t) * a + lam_t * w
                    st = pair_stats(w, a, a_eff, pred, given)
                    acc = pstats.setdefault(s_ + 1, {})
                    for k, d in st.items():
                        e = acc.setdefault(k, dict(n=0, w=0., a=0., a_eff=0., w_neg=0., a_neg=0.))
                        for kk in ("w", "a", "a_eff", "w_neg", "a_neg"): e[kk] += d[kk] * d["n"]
                        e["n"] += d["n"]
        nchurn += ngiven
        print(f"  [{md}] {bi+len(x)}/{NB}  {time.time()-t0:.0f}s  누적 seg16 {int(solved[min(15,SEG-1)].sum())} seg{SEG} {int(solved[-1].sum())}", flush=True)
    cellacc /= nchurn; churn /= nchurn
    for s_, acc in pstats.items():
        for k, e in acc.items():
            for kk in ("w", "a", "a_eff", "w_neg", "a_neg"): e[kk] /= max(e["n"], 1)
    return dict(solved=solved, cellacc=cellacc, churn=churn, pstats=pstats)

res = {}
for md in args.modes.split(","):
    res[md] = run(md)
    np.savez(os.path.join(args.out, f"{args.tag}_{md}.npz"), solved=res[md]["solved"], cellacc=res[md]["cellacc"], churn=res[md]["churn"])

# ---- 표
def er(sol):
    u16 = ~sol[15]; uS = ~sol[-1]
    cond = float((sol[-1] & u16).sum() / max(u16.sum(), 1))          # seg16 미해결 → segS 해결
    lost = float((u16 & sol[15]).sum())                                 # (정의상 0) — 자리만
    unsolve = float(((~sol[-1]) & sol[15]).sum() / max(sol[15].sum(), 1))  # seg16 해결 → segS 미해결
    red = float((u16.sum() - uS.sum()) / max(u16.sum(), 1))
    return cond, unsolve, red
print(f"\n9/1 R1B8_min_faith@310527 · {NB}퍼즐 · seg{SEG}")
cols = [i for i in (15, 31, 63) if i < SEG - 1] + [SEG - 1]
print(f"{'변형':>10} | " + " | ".join(f"seg{i+1:<3}" for i in cols) + f" | 이득 | 오차감소율 | 조건부해결(미해결16→해결) | 해결16→미해결 | cell16 | cell{SEG} | churn16 | churn{SEG}")
for md, r_ in res.items():
    sol = r_["solved"]; cond, unsolve, red = er(sol); ca = r_["cellacc"]; ch = r_["churn"]
    print(f"{md:>10} | " + " | ".join(f"{int(sol[i].sum()):6d}" for i in cols)
          + f" | {int(sol[-1].sum()-sol[15].sum()):+4d} | {red*100:8.1f}% | {cond*100:10.1f}% | {unsolve*100:6.1f}% | {ca[15]:.4f} | {ca[-1]:.4f} | {ch[15]:.4f} | {ch[-1]:.4f}")

if "base" in res:
    print(f"\n[base] w·a·a_eff 쌍 종류별 (비단서 칸 쌍, 헤드 평균). 부호 가설: 피어∧같은숫자(위반) 에서 w<0 인가")
    for s_, acc in res["base"]["pstats"].items():
        print(f"  seg{s_}:")
        for k in ("peer_same", "peer_diff", "nonpeer_same", "nonpeer_diff"):
            if k in acc:
                e = acc[k]; print(f"    {k:>13} n={e['n']:>9d}  w={e['w']:+.4f} (w<0 {e['w_neg']*100:4.1f}%)  a={e['a']:+.4f} (a<0 {e['a_neg']*100:4.1f}%)  a_eff={e['a_eff']:+.4f}")
print("\n저장:", args.out, args.tag)
