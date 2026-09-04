"""학습 1스텝(= 1 세그먼트 fwd+bwd+opt)의 연산 병목 실측.

단위는 COST.md 와 같다: 1 step = model(carry,batch) 1회 + backward + optimizer.step().
URM pretrain.py:544 이 정확히 그 구조 (세그먼트마다 backward, carry 는 detach 되어 이어짐).

사용:
  python 2026-09-04/analysis/bench_train.py --bs 128                 # eager
  python 2026-09-04/analysis/bench_train.py --bs 128 --compile
  python 2026-09-04/analysis/bench_train.py --bs 128 --profile       # 연산자별 CUDA 시간
  python 2026-09-04/analysis/bench_train.py --bs 8 --grid 30         # ARC 크기
"""
import argparse, os, sys, time, json, math
import torch

ROOT = os.environ.get("LT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.join(ROOT, "refs", "URM"))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "sudoku"))

from models.losses import ACTLossHead          # noqa: E402
from adam_atan2 import AdamATan2               # noqa: E402
import minimal                                 # noqa: E402



def patch_sparse_emb():
    """CastedSparseEmbedding.forward 의 범위 검사(`if torch.any(...)`)를 제거한다.
    데이터 의존 분기라 dynamo 가 여기서 그래프를 끊는다 (compile_inspect.py 참조)."""
    from models.sparse_embedding import CastedSparseEmbedding as CSE
    def fwd(self, inputs):
        if not self.training:
            return self.weights[inputs].to(self.cast_to)
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)
        return self.local_weights.to(self.cast_to)
    CSE.forward = fwd

def build(args):
    if args.patch_emb:
        patch_sparse_emb()
    T = args.grid * args.grid
    cfg = dict(batch_size=args.bs, seq_len=T, vocab_size=args.vocab, num_puzzle_identifiers=1,
               puzzle_emb_ndim=args.d, hidden_size=args.d, num_heads=args.heads, loops=16,
               grid=args.grid, blocks_per_seg=args.blocks, num_layers=args.layers, mlp_expansion=args.expansion,
               amp=True, stdp=args.stdp, stdp_target=args.target, stdp_window=args.window,
               stdp_eta_init=0.05, stdp_gain_init=1.0, stdp_lam_fixed=-1.0)
    m = minimal.LT(cfg).cuda()
    head = ACTLossHead(m, "stablemax_cross_entropy").cuda().train()
    if args.compile:
        head = torch.compile(head, dynamic=False)
    return head, T


def make_batch(args, T):
    g = torch.Generator(device="cuda").manual_seed(0)
    inputs = torch.randint(1, args.vocab, (args.bs, T), device="cuda", generator=g)
    labels = torch.randint(1, args.vocab, (args.bs, T), device="cuda", generator=g)
    pid = torch.zeros(args.bs, dtype=torch.int32, device="cuda")
    return {"inputs": inputs, "labels": labels, "puzzle_identifiers": pid}


def make_opt(head):
    named = [(n, p) for n, p in head.named_parameters() if p.requires_grad]
    keys = ("psi", "theta", "alpha_raw", "gamma_raw", "inj_gate", "gain_raw", "eta_raw", "lam_raw", "beta", "mu")
    decay = [p for n, p in named if p.ndim > 1 and not any(k in n for k in keys)]
    nodec = [p for n, p in named if p.ndim <= 1 or any(k in n for k in keys)]
    return AdamATan2([{"params": decay, "weight_decay": 1.0}, {"params": nodec, "weight_decay": 0.0}],
                     lr=1e-4, betas=(0.9, 0.95), weight_decay=1.0)


def one_step(head, opt, carry, batch, bs):
    carry, loss, _, _, _ = head(return_keys=[], carry=carry, batch=batch, compute_target_q=False)
    ((1.0 / bs) * loss).backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    return carry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--d", type=int, default=832)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--expansion", type=float, default=4.0)
    ap.add_argument("--vocab", type=int, default=11)
    ap.add_argument("--stdp", type=int, default=1)
    ap.add_argument("--target", default="faithful")
    ap.add_argument("--window", default="beta")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--fwd-only-split", action="store_true")
    ap.add_argument("--patch-emb", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    head, T = build(args)
    batch = make_batch(args, T)
    opt = make_opt(head)
    with torch.device('cuda'):          # pretrain.py:539 와 동일
        carry = head.initial_carry(batch)

    for _ in range(args.warmup):
        carry = one_step(head, opt, carry, batch, args.bs)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(args.iters)]
    t0 = time.perf_counter()
    for i in range(args.iters):
        ev[i][0].record()
        carry = one_step(head, opt, carry, batch, args.bs)
        ev[i][1].record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / args.iters * 1e3
    ms = sorted(a.elapsed_time(b) for a, b in ev)
    med = ms[len(ms) // 2]
    peak = torch.cuda.max_memory_allocated() / 2**30

    nparam = sum(p.numel() for p in head.parameters())
    res = dict(tag=args.tag, bs=args.bs, T=T, d=args.d, blocks=args.blocks, stdp=bool(args.stdp),
               target=args.target, window=args.window, compiled=args.compile,
               ms_median=round(med, 2), ms_wall=round(wall, 2), it_s=round(1000 / med, 2),
               peak_GB=round(peak, 3), params=nparam)
    print(json.dumps(res))

    if args.fwd_only_split:
        # forward 만 / forward+backward 를 나눠 재 backward 몫을 뺀다
        ev2 = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(args.iters)]
        with torch.no_grad():
            for i in range(args.iters):
                ev2[i][0].record()
                head(return_keys=[], carry=carry, batch=batch, compute_target_q=False)
                ev2[i][1].record()
        torch.cuda.synchronize()
        f = sorted(a.elapsed_time(b) for a, b in ev2)[args.iters // 2]
        print(json.dumps(dict(tag=args.tag + "_fwd_nograd", ms_median=round(f, 2),
                              frac_of_step=round(f / med, 3))))

    if args.profile:
        from torch.profiler import profile, ProfilerActivity, record_function
        # 의미 단위 귀속: 코어 메서드를 record_function 으로 감싼다 (저장소 코드는 안 건드림)
        inner = (head._orig_mod if hasattr(head, "_orig_mod") else head).model.inner
        for name in ("boundary", "addr", "attn_xy", "injection", "phi"):
            fn = getattr(inner, name)
            def mk(fn=fn, name=name):
                def wrapped(*a, **k):
                    with record_function("LT::" + name):
                        return fn(*a, **k)
                return wrapped
            setattr(inner, name, mk())
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                carry = one_step(head, opt, carry, batch, args.bs)
            torch.cuda.synchronize()
        ka = prof.key_averages()

        aten = [e for e in ka if e.key.startswith("aten::") and e.self_device_time_total > 0]
        tot = sum(e.self_device_time_total for e in aten)
        print(f"\n[ATEN] self CUDA 합 {tot/5/1e3:.1f} ms/step  (측정 step {med:.1f} ms)")
        print(f"{'op':<34}{'ms/step':>10}{'%':>7}{'calls/step':>11}")
        for e in sorted(aten, key=lambda e: -e.self_device_time_total)[:16]:
            print(f"{e.key[:32]:<34}{e.self_device_time_total/5/1e3:>10.2f}"
                  f"{100*e.self_device_time_total/tot:>7.1f}{e.count/5:>11.1f}")

        sem = [e for e in ka if e.key.startswith("LT::")]
        if sem:
            print(f"\n[의미 단위 — forward 만, 총 device 시간 기준]")
            print(f"{'구간':<20}{'ms/step':>10}{'calls/step':>11}")
            for e in sorted(sem, key=lambda e: -e.device_time_total):
                print(f"{e.key[4:]:<20}{e.device_time_total/5/1e3:>10.2f}{e.count/5:>11.1f}")


if __name__ == "__main__":
    main()
