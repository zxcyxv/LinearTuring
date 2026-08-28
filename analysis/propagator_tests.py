"""전파자/결정자 판별 셋 (학습 불필요). 체크포인트 폴더의 config.yaml 로 구조 자동 인식.
  T1 밴드 비움(해 다수) 입력: 빈칸 로짓 엔트로피가 남는가 (전파자) vs 한 해로 무너지는가 (결정자). 유효 격자 비율도.
  T2 hidden-only 깊이-1 칸: 쌍선형을 끈 채(경계 항등) 풀리는가.
  T3 정상 입력: 셀/완답, 그리고 결정화(확신 최고 칸 top-1, 깊이 2, 유효 채택) 완답.
사용: python propagator_tests.py --ckpt <ckpt.pt> [--n 256]"""
import argparse, json, numpy as np, torch, subprocess, sys, os
from common import load_lt, load_test, make_batch, rollout, logits, peer_mask, ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=256); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--no_t3", action="store_true")
args = ap.parse_args(); torch.set_grad_enabled(False)
m = load_lt(args.ckpt, batch_size=args.bs); print("구조:", {k: getattr(m.config, k) for k in ("sheaf_units", "lam_mode", "bilinear", "blocks_per_seg")})
inp, lab, depth = load_test(args.n); I = inp.cpu().numpy(); LB = lab.cpu().numpy(); g = LB - 2; N = len(LB); pm = peer_mask()
r_ = np.arange(81) // 9; c_ = np.arange(81) % 9; b_ = (r_ // 3) * 3 + c_ // 3
def run(x, act=None):
    out = np.zeros((len(x), 81, 9), np.float32)
    for i in range(0, len(x), args.bs):
        b = slice(i, min(i + args.bs, len(x))); h = rollout(m, make_batch(x[b], x[b]), act=act); out[b] = logits(m, h)[:, :, 2:11].cpu().numpy()
    return out
def ent(lg): p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); return -(p * np.log(p + 1e-9)).sum(-1)
def valid(board, x):
    fin = board.copy(); giv = x > 1; fin[giv] = x[giv] - 1
    return ~np.stack([(fin[:, pm[t]] == fin[:, t:t + 1]).any(1) for t in range(81)], 1).any(1)
res = {}
# T1
X = LB.copy(); BL = [t for t in range(81) if 3 <= t // 9 <= 5]; X[:, BL] = 1; lg = run(torch.tensor(X, dtype=torch.int32, device="cuda"))
e = ent(lg)[:, BL]; bd = lg.argmax(-1) + 1; v = valid(bd, X)
res["T1_band"] = {"blank_entropy_mean": float(e.mean()), "frac_blank_entropy>1": float((e > 1).mean()), "cell_acc": float((bd[:, BL] == g[:, BL] + 1).mean()), "valid_grid_frac": float(v.mean())}
# T2 단계별 분해: hidden-only / naked-single 깊이-1 칸에서 정답 자리 여유(정답 로짓 − 최대 타후보)가 블록 안 어느 단계에서 생기나
def naked(i, t): return set(range(9)) - set(g[i, [n for n in np.where(pm[t])[0] if I[i, n] > 1]])
def hidden_only(i, t):
    for u in (r_ == r_[t], c_ == c_[t], b_ == b_[t]):
        if all(g[i, t] not in naked(i, n) for n in np.where(u)[0] if I[i, n] == 1 and n != t): return True
    return False
groups = {"hidden_only": [(i, t) for i in range(N) for t in range(81) if depth[i, t] == 1 and len(naked(i, t)) >= 2 and hidden_only(i, t)],
          "naked_single": [(i, t) for i in range(N) for t in range(81) if depth[i, t] == 1 and len(naked(i, t)) == 1]}
stages = ["post_sheaf", "post_bnd", "post_inj", "post_step"] if getattr(m.config, "sheaf_units", False) else ["post_bnd", "post_inj", "post_step"]
STEPS = [1, 2, 3, 7, 15]
for gname, cells in groups.items():
    need = {}; [need.setdefault(i, []).append(t) for i, t in cells]
    delta = {s: {st: [] for st in stages} for s in STEPS}; acc = {s: [] for s in STEPS}
    for i0 in range(0, N, args.bs):
        b = slice(i0, min(i0 + args.bs, N)); cur = {}
        def margin(h, i, t):
            lg = logits(m, h)[i - i0, t, 2:11].float().cpu().numpy(); ans = g[i, t]; alt = [c for c in naked(i, t) if c != ans] or [c for c in range(9) if c != ans]
            return lg[ans] - max(lg[c] for c in alt), lg.argmax() == ans
        def hook(loop, blk, stage, h, a, i0=i0, b=b):
            s = loop * m.config.blocks_per_seg + blk
            if s not in STEPS: return
            for i in range(i0, b.stop):
                for t in need.get(i, []):
                    mg, ok = margin(h, i, t)
                    if stage == "pre": cur[(i, t)] = mg
                    elif stage in stages:
                        delta[s][stage].append(mg - cur[(i, t)]); cur[(i, t)] = mg
                        if stage == "post_step": acc[s].append(ok)
        rollout(m, make_batch(inp[b], inp[b]), hook=hook)
    res[f"T2_{gname}"] = {s: {"acc": float(np.mean(acc[s])), **{st: float(np.mean(delta[s][st])) for st in stages}} for s in STEPS}
    print(f"[T2 {gname} {len(cells)}칸] 블록별 정답률 | 단계별 정답 여유 증가분(평균)")
    for s in STEPS: print(f"  블록 {s:2d}: acc {np.mean(acc[s]):.3f} | " + "  ".join(f"{st} {np.mean(delta[s][st]):+.2f}" for st in stages))
print(json.dumps(res, indent=1))
if args.no_t3: sys.exit(0)
# T3 결정화 (branch.py 재사용)
out = subprocess.run([sys.executable, "branch.py", "--ckpt", args.ckpt, "--depth", "2", "--topk", "1", "--order", "confident", "--accept", "valid"], capture_output=True, text=True)
t = out.stdout; print("T3 결정화:", json.loads(t[t.index("{"):])["branch"] if "{" in t else out.stderr[-500:])
