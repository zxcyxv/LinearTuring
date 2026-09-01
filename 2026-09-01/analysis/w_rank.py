"""결합 기억 w 의 랭크 — 세그먼트가 진행되며 어떻게 변하는가.

물음: w 는 81×81 이지만 실제로 몇 차원을 쓰고 있는가. 그리고 그 값이 세그먼트에 따라
      올랐다 내려가는가(가설을 벌렸다 접는가), 아니면 단조인가.
비교: 같은 시점의 a (전달 커널) 스펙트럼. w 가 a 보다 두꺼우면 아다마르 랭크 증폭이 실제로 쓰이는 것.
지표: 참여비 PR = (Σσ)²/Σσ²  ·  안정랭크 SR = Σσ²/σ₁²  ·  σ>10⁻²σ₁ 개수
사용: python 2026-09-01/analysis/w_rank.py [--n 64] [--segs 192]
"""
import argparse, json, os, sys, importlib.util
import numpy as np, torch, yaml
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
URM = os.path.join(ROOT, "refs", "URM"); TEST = os.path.join(URM, "data", "sudoku-extreme-1k-aug-1000", "test")
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "R1B8_min_faith_step310527.pt"))
ap.add_argument("--cfg",  default=os.path.join(ROOT, "checkpoints", "R1B8_min_faith_config.yaml"))
ap.add_argument("--n", type=int, default=64); ap.add_argument("--segs", type=int, default=192)
ap.add_argument("--marks", type=int, nargs="+", default=[1,2,4,8,12,16,24,32,48,64,96,128,192])
ap.add_argument("--out", default=os.path.join(ROOT, "2026-09-01", "results", "json", "w_rank.json"))
args = ap.parse_args(); torch.set_grad_enabled(False)

inp = torch.from_numpy(np.load(f"{TEST}/all__inputs.npy")).to(torch.int32)[:args.n]
lab = torch.from_numpy(np.load(f"{TEST}/all__labels.npy")).to(torch.int32)[:args.n]
pid = torch.from_numpy(np.load(f"{TEST}/all__puzzle_identifiers.npy")).to(torch.int32)[:args.n]
N = len(inp)

cfg = yaml.safe_load(open(args.cfg))["arch"]; cfg = {k: v for k, v in cfg.items() if k not in ("name","loss")}
cfg.update(batch_size=N, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, loops=args.segs + 1)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
full = dict(ck["raw_model_state_dict"]); full.update(ck["model_state_dict"])
m = mod.LT(cfg).cuda().eval()
m.load_state_dict({k.replace("_orig_mod.model.", ""): v for k, v in full.items()}, strict=True)
inner = m.inner
print(f"step={ck['step']}  N={N}  segs={args.segs}")

def spec(M):
    """M [P,H,T,T] → 헤드·퍼즐별 지표 평균."""
    s = torch.linalg.svdvals(M.double())                      # [P,H,T]
    tot1 = s.sum(-1); tot2 = (s * s).sum(-1)
    pr = tot1 * tot1 / tot2.clamp_min(1e-30)                  # 참여비
    sr = tot2 / (s[..., 0] ** 2).clamp_min(1e-30)             # 안정랭크
    n2 = (s > 1e-2 * s[..., :1]).sum(-1).double()
    n3 = (s > 1e-3 * s[..., :1]).sum(-1).double()
    return [float(x.mean()) for x in (pr, sr, n2, n3)]

x = inp.cuda(); y = lab.cuda().long(); p = pid.cuda()
batch = dict(inputs=x, labels=y, puzzle_identifiers=p)
carry = mod.LTCarry(current_hidden=torch.empty(N, 81, cfg["hidden_size"], device="cuda"),
                    steps=torch.zeros(N, dtype=torch.int32, device="cuda"),
                    halted=torch.ones(N, dtype=torch.bool, device="cuda"),
                    current_data={k: torch.empty_like(v) for k, v in batch.items()})
AB = inner.W_C(); kc = inner.kernel()
rows = []
for s in range(args.segs):
    carry, out = m(carry, batch)
    if (s + 1) in args.marks:
        w = carry.coupling.float()
        a = inner.attn(carry.current_hidden.float(), AB, kc).float()
        ex = ((out["logits"].argmax(-1) == y).all(-1)).float().mean().item()
        rows.append(dict(seg=s + 1, exact=ex, w=spec(w), a=spec(a)))
        r = rows[-1]
        print(f"seg {s+1:4d}  완답 {100*ex:5.1f}% | w: PR {r['w'][0]:6.2f} SR {r['w'][1]:6.2f} "
              f"n(1e-2) {r['w'][2]:5.1f} n(1e-3) {r['w'][3]:5.1f} | a: PR {r['a'][0]:6.2f} SR {r['a'][1]:6.2f} "
              f"n(1e-2) {r['a'][2]:5.1f} n(1e-3) {r['a'][3]:5.1f}", flush=True)
print(f"\n(T=81 이 상한. 2p=104 이므로 특징공간 M 은 스도쿠에서 어떤 w 든 표현 가능)")
json.dump(dict(step=int(ck["step"]), n=N, rows=rows), open(args.out, "w"), indent=1)
print(f"저장: {args.out}")
