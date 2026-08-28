"""학습 없이 경계 게이트 함수만 치환해 추론: silu / g·½ (쌍선형) / 2차·4차 테일러 / relu / gelu / 항등 / 상수(=선형 경계).
사용: python gate_swap.py [--ckpt PATH] [--n 512] [--out JSON]
원 결과: results/json/gate_swap_R1B8_9765.json (SwiGLU 판 R1B8 @9765, 512퍼즐)."""
import argparse, json
import numpy as np, torch, torch.nn.functional as F
from common import load_lt, load_test, make_batch, rollout, logits, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=f"{ROOT}/checkpoints/R1B8_swiglu_step9765.pt")
ap.add_argument("--bilinear", type=int, default=0, help="체크포인트가 쌍선형 판이면 1")
ap.add_argument("--n", type=int, default=512); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
inp, lab, _ = load_test(args.n)
acts = {"silu": F.silu, "g/2 (쌍선형)": lambda x: x / 2, "g/2+g²/4": lambda x: x / 2 + x * x / 4,
        "g/2+g²/4−g⁴/48": lambda x: x / 2 + x * x / 4 - x ** 4 / 48,
        "relu": F.relu, "gelu": F.gelu, "g (항등)": lambda x: x, "상수 ½ (=선형 경계)": lambda x: torch.full_like(x, 0.5)}
res = {}
for name, act in acts.items():
    ok = []
    for i in range(0, args.n, args.bs):
        b = slice(i, i + args.bs)
        h = rollout(m, make_batch(inp[b], lab[b]), act=act)
        ok.append((logits(m, h).argmax(-1).to(torch.int32) == lab[b]).cpu().numpy())
    ok = np.concatenate(ok); res[name] = {"cell": round(float(ok.mean()), 4), "exact": int(ok.all(1).sum())}
    print(f"{name:22s} 셀 {res[name]['cell']:.4f}  완답 {res[name]['exact']}/{args.n}", flush=True)
if args.out: json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
