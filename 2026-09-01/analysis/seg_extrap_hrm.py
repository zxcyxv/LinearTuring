"""HRM 공식 체크포인트(sapientinc/HRM-checkpoint-sudoku-extreme)의 세그먼트 외삽.

그 체크포인트는 우리와 **같은 데이터셋**(data/sudoku-extreme-1k-aug-1000)에 **같은 16 세그먼트**
(halt_max_steps=16)로 학습됐다. 빌드 스크립트상 test 는 증강도 서브샘플링도 없으므로 우리 2,048 과 동일하다.
지표·프로토콜을 `seg_extrap.py` 와 똑같이 맞춰 Δ(외삽 이득)를 직접 잰다.

기본은 ACT 끔(`act_inference=False`, HRM 의 추론 기본값) — 매 스텝 결정론적으로 끝까지 돌린다.
`--act` 를 주면 Q 헤드 조기정지를 켠다 (HRM 논문 Fig 5(c) 의 프로토콜).
사용: python 2026-09-01/analysis/seg_extrap_hrm.py [--segs 256] [--bs 256] [--act]
"""
import argparse, json, os, sys, time
import numpy as np, torch, yaml
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
URM = os.path.join(ROOT, "refs", "URM")
TEST = os.path.join(URM, "data", "sudoku-extreme-1k-aug-1000", "test")
sys.path.insert(0, URM)
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1 as HRM   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ROOT, "refs", "ckpt_hrm", "checkpoint"))
ap.add_argument("--cfg",  default=os.path.join(ROOT, "refs", "ckpt_hrm", "all_config.yaml"))
ap.add_argument("--segs", type=int, default=256)
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--act", action="store_true", help="Q 헤드 조기정지 켬 (기본은 끔 = HRM 추론 기본값)")
ap.add_argument("--tag", default=None)
args = ap.parse_args()
args.tag = args.tag or ("hrm_act" if args.act else "hrm")
OUT = os.path.join(ROOT, "2026-09-01", "results", "json", f"seg_extrap_{args.tag}.json")
torch.set_grad_enabled(False)

inp = torch.from_numpy(np.load(f"{TEST}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{TEST}/all__labels.npy")).to(torch.int32)
pid = torch.from_numpy(np.load(f"{TEST}/all__puzzle_identifiers.npy")).to(torch.int32)
if args.n: inp, lab, pid = inp[:args.n], lab[:args.n], pid[:args.n]
N = len(inp)
meta = json.load(open(f"{TEST}/dataset.json"))

arch = yaml.safe_load(open(args.cfg))["arch"]
arch = {k: v for k, v in arch.items() if k not in ("name", "loss")}
arch.update(batch_size=args.bs, seq_len=meta["seq_len"], vocab_size=meta["vocab_size"],
            num_puzzle_identifiers=meta["num_puzzle_identifiers"],
            halt_max_steps=args.segs + 1, act_inference=bool(args.act))
sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
sd = {k.replace("_orig_mod.model.", ""): v for k, v in sd.items()}
m = HRM(arch).cuda().eval()
info = m.load_state_dict(sd, strict=False)
assert not info.unexpected_keys, info.unexpected_keys
assert not [k for k in info.missing_keys if "puzzle_emb" not in k], info.missing_keys
print(f"HRM  파라미터 {sum(p.numel() for p in m.parameters())/1e6:.2f}M  "
      f"halt_max_steps={arch['halt_max_steps']}  ACT={'on' if args.act else 'off'}  N={N}")
print(f"  H_cycles={arch['H_cycles']} L_cycles={arch['L_cycles']} "
      f"H_layers={arch['H_layers']} L_layers={arch['L_layers']} hidden={arch['hidden_size']}")

def to_cuda(x):
    if torch.is_tensor(x): return x.cuda()
    if isinstance(x, dict): return {k: to_cuda(v) for k, v in x.items()}
    if hasattr(x, "__dataclass_fields__"):
        return replace(x, **{f: to_cuda(getattr(x, f)) for f in x.__dataclass_fields__})
    return x

IGN = -100
corr = np.zeros(args.segs); exact = np.zeros(args.segs); churn = np.zeros(args.segs); halt = np.zeros(args.segs)
t0 = time.time()
for b in range(0, N, args.bs):
    x = inp[b:b+args.bs].cuda(); y = lab[b:b+args.bs].cuda().long(); p = pid[b:b+args.bs].cuda()
    n_ = len(x); batch = dict(inputs=x, labels=y, puzzle_identifiers=p)
    carry = to_cuda(m.initial_carry(batch))
    mask = y != IGN; cnt = mask.sum(-1).clamp_min(1)
    prev = None
    for s in range(args.segs):
        carry, out = m(carry, batch)
        pred = out["logits"].argmax(-1)
        ok = mask & (pred == y)
        corr[s] += (ok.sum(-1).float() / cnt).sum().item()
        exact[s] += (ok.sum(-1) == mask.sum(-1)).sum().item()
        halt[s] += carry.halted.sum().item()
        if prev is not None: churn[s] += (pred != prev).float().mean(-1).sum().item()
        prev = pred
    print(f"  batch {b//args.bs+1}/{-(-N//args.bs)}  {time.time()-t0:.0f}s", flush=True)

corr /= N; churn /= N; halt /= N
base_a, base_e = corr[15], exact[15]
print(f"\n{'seg':>5} | {'acc':>7} {'완답':>5} {'완답%':>7} | {'Δ%p':>7} | {'churn':>8} | {'halted':>6}")
for s in [0, 3, 7, 15, 23, 31, 47, 63, 95, 127, 191, 255]:
    if s >= args.segs: continue
    print(f"{s+1:5d} | {corr[s]:7.4f} {int(exact[s]):5d} {100*exact[s]/N:6.2f}% | "
          f"{100*(exact[s]-base_e)/N:+6.2f} | {churn[s]:8.5f} | {halt[s]:6.3f}")
print(f"\n학습 세그먼트(16): acc {base_a:.4f}  완답 {int(base_e)}/{N} ({100*base_e/N:.2f}%)")
bi = int(np.argmax(corr)); be = int(np.argmax(exact))
print(f"최고 acc  : seg {bi+1}  {corr[bi]:.4f} ({corr[bi]-base_a:+.4f})")
print(f"최고 완답 : seg {be+1}  {int(exact[be])} ({100*exact[be]/N:.2f}%, {100*(exact[be]-base_e)/N:+.2f}%p)")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(model="HRM sapientinc/HRM-checkpoint-sudoku-extreme", n=N, act=bool(args.act),
               acc=corr.tolist(), exact=exact.tolist(), churn=churn.tolist(), halted=halt.tolist()),
          open(OUT, "w"), indent=1)
print(f"\n저장: {OUT}")
