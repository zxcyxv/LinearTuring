"""세그먼트 외삽: 학습은 loops=16 으로 했다. 추론에서 세그먼트를 더 주면 어떻게 되는가.

기대하는 것은 단조증가가 아니라 '눈에 띄는 증가' — 어느 세그먼트 수에서 계단이 있는가.
지표는 학습 하네스와 동일: accuracy = 81칸 중 맞은 비율, exact = 81칸 전부 맞은 퍼즐 수.
사용: python 2026-09-01/analysis/seg_extrap.py [--segs 64] [--n 2048] [--bs 128] [--weights ema|raw]
"""
import argparse, json, os, sys, time, importlib.util
import numpy as np, torch
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
URM = os.path.join(ROOT, "refs", "URM")
TEST = os.path.join(URM, "data", "sudoku-extreme-1k-aug-1000", "test")
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(URM, "checkpoints", "R1B8_min_faith", "step_310527.pt"))
ap.add_argument("--segs", type=int, default=64)
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
ap.add_argument("--config", default=None, help="arch config yaml (기본: 체크포인트 옆 config.yaml)")
ap.add_argument("--tag", default="faith")
ap.add_argument("--lam", type=float, default=-1.0, help=">=0 이면 λ 를 이 값으로 고정 (0 = 기억 끔, 대조군)")
ap.add_argument("--out", default=os.path.join(ROOT, "2026-09-01", "results", "json", "seg_extrap.json"))
args = ap.parse_args()
if args.out.endswith("seg_extrap.json"): args.out = args.out.replace("seg_extrap.json", f"seg_extrap_{args.tag}.json")
torch.set_grad_enabled(False)

# ---- 데이터 (학습 하네스가 쓰는 test 셋 그대로)
inp = torch.from_numpy(np.load(f"{TEST}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{TEST}/all__labels.npy")).to(torch.int32)
pid = torch.from_numpy(np.load(f"{TEST}/all__puzzle_identifiers.npy")).to(torch.int32)
if args.n: inp, lab, pid = inp[:args.n], lab[:args.n], pid[:args.n]
N = len(inp)

# ---- 모델 (학습 config 그대로, loops 만 늘림)
import yaml
cfg_path = args.config or os.path.join(os.path.dirname(args.ckpt), "config.yaml")
cfg = yaml.safe_load(open(cfg_path))["arch"]
cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
cfg.update(batch_size=args.bs, seq_len=81, vocab_size=11,
           num_puzzle_identifiers=int(pid.max().item()) + 1, loops=args.segs + 1)
if args.lam >= 0: cfg["stdp_lam_fixed"] = args.lam
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
# ema_shadow 는 파라미터만 담는다 — 버퍼(init_hidden, puzzle_emb.weights)는 raw 에서 가져와야 한다
# model_state_dict = EMA 적용본 (ema_shadow 와 비트 동일, 확인함). ema_shadow 는 파라미터만이라 버퍼가 빠진다.
full = dict(ck["raw_model_state_dict"])
if args.weights == "ema": full.update(ck["model_state_dict"])
sd = {k.replace("_orig_mod.model.", ""): v for k, v in full.items()}
m = mod.LT(cfg).cuda().eval()
npi = cfg["num_puzzle_identifiers"]
miss = m.load_state_dict(sd, strict=False)
assert not miss.unexpected_keys, miss.unexpected_keys
if miss.missing_keys:   # gain_raw 는 stdp1 이후 추가된 파라미터 — init=1.0 이 곧 구 동작
    assert set(miss.missing_keys) <= {"inner.gain_raw"}, miss.missing_keys
    print(f"  누락(기본값 사용): {miss.missing_keys}")
print(f"step={ck['step']}  weights={args.weights}  loops={cfg['loops']}  npi={npi}")

print(f"  λ={torch.sigmoid(m.inner.lam_raw).flatten().tolist() if cfg['stdp_lam_fixed']<0 else cfg['stdp_lam_fixed']}")
print(f"  δ={torch.sigmoid(m.inner.eta_raw).flatten().mean().item():.4f}  "
      f"G={torch.nn.functional.softplus(m.inner.gain_raw).flatten().mean().item():.4f}")

IGN = -100
corr = np.zeros(args.segs); exact = np.zeros(args.segs); churn = np.zeros(args.segs)
t0 = time.time()
for b in range(0, N, args.bs):
    x = inp[b:b+args.bs].cuda(); y = lab[b:b+args.bs].cuda().long(); p = pid[b:b+args.bs].cuda()
    n_ = len(x)
    batch = dict(inputs=x, labels=y, puzzle_identifiers=p)
    carry = mod.LTCarry(
        current_hidden=torch.empty(n_, 81, cfg["hidden_size"], device="cuda"),
        steps=torch.zeros(n_, dtype=torch.int32, device="cuda"),
        halted=torch.ones(n_, dtype=torch.bool, device="cuda"),
        current_data={k: torch.empty_like(v) for k, v in batch.items()})
    mask = y != IGN; cnt = mask.sum(-1).clamp_min(1)
    prev = None
    for s in range(args.segs):
        carry, out = m(carry, batch)
        pred = out["logits"].argmax(-1)
        ok = mask & (pred == y)
        corr[s] += (ok.sum(-1).float() / cnt).sum().item()
        exact[s] += (ok.sum(-1) == mask.sum(-1)).sum().item()
        if prev is not None: churn[s] += (pred != prev).float().mean(-1).sum().item()
        prev = pred
    print(f"  batch {b//args.bs+1}/{-(-N//args.bs)}  {time.time()-t0:.0f}s", flush=True)

corr /= N; churn /= N
print(f"\n{'seg':>5} | {'acc':>7} {'exact':>6} {'exact%':>7} | {'Δacc':>8} | {'churn':>7}")
base_a, base_e = corr[15], exact[15]
for s in range(args.segs):
    mark = " ←학습" if s == 15 else ""
    print(f"{s+1:5d} | {corr[s]:7.4f} {int(exact[s]):6d} {100*exact[s]/N:6.2f}% | "
          f"{corr[s]-base_a:+8.4f} | {churn[s]:7.4f}{mark}")
print(f"\n학습 세그먼트(16): acc {base_a:.4f}  완답 {int(base_e)}/{N} ({100*base_e/N:.2f}%)")
bi = int(np.argmax(corr)); be = int(np.argmax(exact))
print(f"최고 acc:  seg {bi+1}  {corr[bi]:.4f}  ({corr[bi]-base_a:+.4f})")
print(f"최고 완답: seg {be+1}  {int(exact[be])}/{N}  ({int(exact[be]-base_e):+d})")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(dict(step=int(ck["step"]), weights=args.weights, n=N, lam=args.lam,
               acc=corr.tolist(), exact=exact.tolist(), churn=churn.tolist()),
          open(args.out, "w"), indent=1)
print(f"\n저장: {args.out}")
