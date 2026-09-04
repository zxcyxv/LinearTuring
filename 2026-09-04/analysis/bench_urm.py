"""URM 기준선의 학습 1스텝 비용 — 우리와 같은 단위·같은 배치로.

표준 스도쿠 프로토콜 (refs/URM/scripts/URM_sudoku.sh):
  arch=urm loops=16 H_cycles=2 L_cycles=6 num_layers=4  (hidden 512·H8·expansion 4·rope)
  global_batch_size=128, lr 1e-4, wd 1.0
주의: 배포된 config/arch/urm.yaml 에 profile: true 가 있어 pretrain.py:183 의 should_compile 이
      False 가 된다 → 표준 설정의 URM 은 torch.compile 없이 돈다. --compile 로 켜서도 잰다.
"""
import argparse, os, sys, time, json
import torch

ROOT = os.environ.get("LT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.join(ROOT, "refs", "URM"))
sys.path.insert(0, os.path.join(ROOT, "sudoku"))
from models.losses import ACTLossHead     # noqa: E402
from models.urm.urm import URM            # noqa: E402
from adam_atan2 import AdamATan2          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--H-cycles", type=int, default=2)
    ap.add_argument("--L-cycles", type=int, default=6)
    ap.add_argument("--vocab", type=int, default=11)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--tag", default="urm")
    a = ap.parse_args()

    torch.manual_seed(0)
    T = a.grid * a.grid
    cfg = dict(batch_size=a.bs, seq_len=T, puzzle_emb_ndim=a.hidden, num_puzzle_identifiers=1,
               vocab_size=a.vocab, num_layers=a.layers, hidden_size=a.hidden, expansion=4.0,
               num_heads=a.heads, pos_encodings="rope", loops=16,
               L_cycles=a.L_cycles, H_cycles=a.H_cycles)
    with torch.device("cuda"):
        m = URM(cfg)
        head = ACTLossHead(m, "stablemax_cross_entropy").train()
        if a.compile:
            head = torch.compile(head, dynamic=False)

    g = torch.Generator(device="cuda").manual_seed(0)
    batch = {"inputs": torch.randint(1, a.vocab, (a.bs, T), device="cuda", generator=g),
             "labels": torch.randint(1, a.vocab, (a.bs, T), device="cuda", generator=g),
             "puzzle_identifiers": torch.zeros(a.bs, dtype=torch.int32, device="cuda")}

    named = [(n, p) for n, p in head.named_parameters() if p.requires_grad]
    dec = [p for n, p in named if p.ndim > 1]
    nod = [p for n, p in named if p.ndim <= 1]
    opt = AdamATan2([{"params": dec, "weight_decay": 1.0}, {"params": nod, "weight_decay": 0.0}],
                    lr=1e-4, betas=(0.9, 0.95), weight_decay=1.0)

    with torch.device("cuda"):
        carry = head.initial_carry(batch)

    def step(carry):
        carry, loss, *_ = head(return_keys=[], carry=carry, batch=batch, compute_target_q=False)
        ((1.0 / a.bs) * loss).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        return carry

    for _ in range(a.warmup):
        carry = step(carry)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(a.iters)]
    for i in range(a.iters):
        ev[i][0].record(); carry = step(carry); ev[i][1].record()
    torch.cuda.synchronize()
    ms = sorted(x.elapsed_time(y) for x, y in ev)[a.iters // 2]

    print(json.dumps(dict(tag=a.tag, bs=a.bs, T=T, hidden=a.hidden, layers=a.layers,
                          H=a.H_cycles, L=a.L_cycles, compiled=a.compile,
                          ms_median=round(ms, 2), it_s=round(1000 / ms, 2),
                          peak_GB=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                          params=sum(p.numel() for p in head.parameters()),
                          layer_applications=a.H_cycles * a.L_cycles * a.layers)))

    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                carry = step(carry)
            torch.cuda.synchronize()
        aten = [e for e in prof.key_averages() if e.key.startswith("aten::") and e.self_device_time_total > 0]
        tot = sum(e.self_device_time_total for e in aten)
        print(f"\n[ATEN] self CUDA 합 {tot/5/1e3:.1f} ms/step")
        print(f"{'op':<34}{'ms/step':>10}{'%':>7}{'calls/step':>11}")
        for e in sorted(aten, key=lambda e: -e.self_device_time_total)[:14]:
            print(f"{e.key[:32]:<34}{e.self_device_time_total/5/1e3:>10.2f}"
                  f"{100*e.self_device_time_total/tot:>7.1f}{e.count/5:>11.1f}")


if __name__ == "__main__":
    main()
