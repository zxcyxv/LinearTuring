"""세그먼트 외삽 — 캐글 단독 학습 체크포인트(train_kaggle.py 형식)용.

9/1 `seg_extrap.py` 와 같은 프로토콜: 가중치 고정, 추론에서 세그먼트만 늘림. 채점도 동일
(accuracy = 81칸 중 맞은 비율, exact = 81칸 전부 맞은 퍼즐 수, churn = 직전 세그먼트 대비 답 바뀐 칸 비율).
모델 클래스는 kaggle/train_kaggle.py 에서 그대로 import (import 시 CUDA 를 건드리지 않는 파일).
데이터는 kaggle/upload/sudoku_lt_1k.npz 의 test 2048 (학습 eval 과 동일 인코딩: +1).
사용: python 2026-09-05/analysis/seg_extrap_kaggle.py --ckpt step_182450.zip --segs 256 --bs 256
"""
import argparse, json, os, sys, time, importlib.util
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("train_kaggle", os.path.join(ROOT, "kaggle", "train_kaggle.py"))
tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ROOT, "step_182450.zip"))
ap.add_argument("--npz", default=os.path.join(ROOT, "kaggle", "upload", "sudoku_lt_1k.npz"))
ap.add_argument("--segs", type=int, default=256)
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
ap.add_argument("--tag", default="kaggle182450")
ap.add_argument("--out", default=os.path.join(ROOT, "2026-09-05", "results", "json", "seg_extrap.json"))
args = ap.parse_args()
if args.out.endswith("seg_extrap.json"): args.out = args.out.replace("seg_extrap.json", f"seg_extrap_{args.tag}.json")
torch.set_grad_enabled(False)

# ---- 데이터 (train_kaggle.eval_batches 와 같은 인코딩)
z = np.load(args.npz)
inp = torch.from_numpy(z["test_inputs"].reshape(-1, 81).astype(np.int32) + 1)
lab = torch.from_numpy(z["test_labels"].reshape(-1, 81).astype(np.int32) + 1)
if args.n: inp, lab = inp[:args.n], lab[:args.n]
N = len(inp)
assert N % args.bs == 0, f"puzzle_emb 가 batch_size 고정이라 N({N}) 은 bs({args.bs}) 의 배수여야 한다"

# ---- 모델 (체크포인트 cfg 그대로, batch_size·loops 만 교체)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=args.bs, seq_len=81, num_puzzle_identifiers=1, loops=args.segs + 1)
sd = ck["model_state_dict"] if args.weights == "ema" else ck["raw_model_state_dict"]
sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in tk.strip_prefix(sd).items()}
m = tk.LT(cfg).cuda().eval()
miss = m.load_state_dict(sd, strict=True)
L0 = m.inner.layers[0]
print(f"step={ck['step']}  weights={args.weights}  loops={cfg['loops']}  bs={args.bs}  N={N}")
print(f"  cfg: d={cfg['hidden_size']} H={cfg['num_heads']} blocks={cfg['blocks_per_seg']} layers={cfg['num_layers']} "
      f"order={cfg['block_order']} stdp={cfg['stdp_target']}/{cfg['stdp_window']}")
for li, L in enumerate(m.inner.layers):
    print(f"  layer{li}: λ={torch.sigmoid(L.lam_raw).flatten().mean().item():.4f}  "
          f"δ={torch.sigmoid(L.eta_raw).flatten().mean().item():.4f}  "
          f"G={torch.nn.functional.softplus(L.gain_raw).flatten().mean().item():.4f}")

corr = np.zeros(args.segs); exact = np.zeros(args.segs); churn = np.zeros(args.segs)
t0 = time.time()
for b in range(0, N, args.bs):
    x = inp[b:b+args.bs].cuda(); y = lab[b:b+args.bs].cuda().long()
    n_ = len(x)
    batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(n_, dtype=torch.int32, device="cuda"))
    with torch.device("cuda"):
        carry = m.initial_carry(batch)
    mask = y != tk.IGNORE_LABEL_ID; cnt = mask.sum(-1).clamp_min(1)
    prev = None
    for s in range(args.segs):
        carry, out = m(carry, batch)
        pred = out["logits"].argmax(-1)
        ok = mask & (pred == y)
        corr[s] += (ok.sum(-1).float() / cnt).sum().item()
        exact[s] += (ok.sum(-1) == mask.sum(-1)).sum().item()
        if prev is not None: churn[s] += (pred != prev).float().mean(-1).sum().item()
        prev = pred
    print(f"  batch {b//args.bs+1}/{N//args.bs}  {time.time()-t0:.0f}s  "
          f"(seg16 누적 exact {int(exact[15])}, seg{args.segs} {int(exact[-1])})", flush=True)

corr /= N; churn /= N
print(f"\n{'seg':>5} | {'acc':>7} {'exact':>6} {'exact%':>7} | {'Δacc':>8} | {'churn':>7}")
base_a, base_e = corr[15], exact[15]
for s in [i for i in range(args.segs) if (i + 1) in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256) or i + 1 == args.segs]:
    mark = " ←학습" if s == 15 else ""
    print(f"{s+1:5d} | {corr[s]:7.4f} {int(exact[s]):6d} {100*exact[s]/N:6.2f}% | {corr[s]-base_a:+8.4f} | {churn[s]:7.4f}{mark}")
print(f"\n학습 세그먼트(16): acc {base_a:.4f}  완답 {int(base_e)}/{N} ({100*base_e/N:.2f}%)")
bi = int(np.argmax(corr)); be = int(np.argmax(exact))
print(f"최고 acc:  seg {bi+1}  {corr[bi]:.4f}  ({corr[bi]-base_a:+.4f})")
print(f"최고 완답: seg {be+1}  {int(exact[be])}/{N} ({100*exact[be]/N:.2f}%)  ({int(exact[be]-base_e):+d})")
print(f"기준(9/1 faithful@310527 seg192): 77.49%  →  {'넘음' if 100*exact[be]/N > 77.49 else '미달'}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(dict(step=int(ck["step"]), ckpt=os.path.basename(args.ckpt), weights=args.weights, n=N, bs=args.bs,
               acc=corr.tolist(), exact=exact.tolist(), churn=churn.tolist()), open(args.out, "w"), indent=1)
print(f"\n저장: {args.out}")
