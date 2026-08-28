"""경계 게이트 사전활성 g 의 분포 — silu 가 다항식 구간에 있는지 판정.
loop 0/3/15 × 블록 0/7 에서 h 원소 std · ‖h_t‖ · g std · |g|>1,2,3 비율.
사용: python gate_stats.py [--ckpt PATH --bilinear 0|1] [--n 128]"""
import argparse, torch
from common import load_lt, load_test, make_batch, rollout, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--n", type=int, default=128)
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.n)
inp, lab, _ = load_test(args.n)
inner = m.inner
gamma = inner.core.gamma.item()
print(f"γ = {gamma:.4f}  → Φ 포화 반경 1/√(2γ·½) = {(1/(2*gamma*0.5))**0.5:.2f}")

def hook(loop, blk, stage, h, a):
    if stage != "pre" or loop not in (0, 3, 15) or blk not in (0, 7): return
    g, u = inner.b_gate_up(h).chunk(2, -1)
    print(f"  L{loop} b{blk}: h원소 std {h.std().item():.3f} ‖h_t‖ {h.norm(dim=-1).mean().item():.2f} | "
          f"g std {g.std().item():.3f} |g|>1 {(g.abs()>1).float().mean().item():.3f} "
          f">2 {(g.abs()>2).float().mean().item():.4f} >3 {(g.abs()>3).float().mean().item():.5f} | u std {u.std().item():.3f}")
rollout(m, make_batch(inp, lab), hook=hook)
