"""torch.compile 이 이 모델에서 실제로 어디까지 융합하는지.

1) dynamo 그래프 분해: 그래프 수 · 그래프 브레이크와 그 이유
2) inductor 가 생성한 커널 목록: 융합 커널(triton_*_fused_*) vs 남은 외부 호출(mm/bmm)
3) 프로파일에서 커널별 시간
"""
import argparse, os, sys, re, json, collections
import torch

ROOT = os.environ.get("LT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.join(ROOT, "refs", "URM"))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "sudoku"))
from models.losses import ACTLossHead   # noqa: E402
import minimal                          # noqa: E402



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

def build(a):
    if a.patch_emb:
        patch_sparse_emb()
    T = a.grid * a.grid
    cfg = dict(batch_size=a.bs, seq_len=T, vocab_size=11, num_puzzle_identifiers=1,
               puzzle_emb_ndim=a.d, hidden_size=a.d, num_heads=8, loops=16, grid=a.grid,
               blocks_per_seg=a.blocks, mlp_expansion=4.0, amp=True, stdp=bool(a.stdp),
               stdp_target="faithful", stdp_window=a.window, stdp_eta_init=0.05,
               stdp_gain_init=1.0, stdp_lam_fixed=-1.0)
    m = minimal.LT(cfg).cuda()
    return ACTLossHead(m, "stablemax_cross_entropy").cuda().train(), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--d", type=int, default=832)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--stdp", type=int, default=1)
    ap.add_argument("--window", default="psi")
    ap.add_argument("--patch-emb", action="store_true")
    a = ap.parse_args()

    head, T = build(a)
    g = torch.Generator(device="cuda").manual_seed(0)
    batch = {"inputs": torch.randint(1, 11, (a.bs, T), device="cuda", generator=g),
             "labels": torch.randint(1, 11, (a.bs, T), device="cuda", generator=g),
             "puzzle_identifiers": torch.zeros(a.bs, dtype=torch.int32, device="cuda")}
    with torch.device("cuda"):
        carry = head.initial_carry(batch)

    # ---- 1) 그래프 브레이크 --------------------------------------------------
    import torch._dynamo as dynamo
    dynamo.reset()
    exp = dynamo.explain(lambda: head(return_keys=[], carry=carry, batch=batch, compute_target_q=False))()
    print(f"[DYNAMO] 그래프 {exp.graph_count}개 · 그래프 브레이크 {exp.graph_break_count}개 · "
          f"op {exp.op_count}개")
    for i, r in enumerate(exp.break_reasons):
        print(f"  break {i+1}: {str(r.reason)[:140]}")
        for fs in (r.user_stack or [])[-2:]:
            print(f"      {fs.filename.split('/')[-1]}:{fs.lineno} in {fs.name}")

    # ---- 2) inductor 커널 --------------------------------------------------
    dynamo.reset()
    outdir = os.path.join(ROOT, "2026-09-04", "results", "inductor")
    os.makedirs(outdir, exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = outdir
    from torch._inductor import config as ind
    ind.trace.enabled = True
    ind.trace.output_code = True
    ch = torch.compile(head, dynamic=False)
    c2, loss, *_ = ch(return_keys=[], carry=carry, batch=batch, compute_target_q=False)
    (loss / a.bs).backward()
    torch.cuda.synchronize()

    codes = []
    for root, _, files in os.walk(outdir):
        for f in files:
            if f == "output_code.py":
                codes.append(os.path.join(root, f))
    print(f"\n[INDUCTOR] output_code.py {len(codes)}개 (fwd/bwd 그래프별)")
    tot = collections.Counter()
    for path in sorted(codes):
        src = open(path).read()
        kern = re.findall(r"^(triton_\w+?)\s*=\s*async_compile\.triton", src, re.M)
        names = re.findall(r"def (triton_[a-z]+_fused_[a-z0-9_]+)", src)
        extern = re.findall(r"extern_kernels\.(\w+)", src)
        rel = os.path.relpath(path, outdir)
        print(f"  {rel.split('/')[0][:38]:<40} triton {len(kern):>3}  extern {len(extern):>3} "
              f"({', '.join(sorted(set(extern))) or '-'})")
        tot["triton"] += len(kern); tot["extern"] += len(extern)
        for n in names:
            ops = n.split("_fused_")[1]
            tot[("fusion", tuple(ops.split("_")[:-1]))] += 1
    print(f"  합계: triton 커널 {tot['triton']}개, extern(cuBLAS) 호출 {tot['extern']}개")

    print("\n[융합 묶음 상위] — 커널 이름에 들어간 aten op 집합")
    fus = [(k[1], v) for k, v in tot.items() if isinstance(k, tuple) and k[0] == "fusion"]
    for ops, n in sorted(fus, key=lambda x: -x[1])[:18]:
        print(f"  x{n:<3} {'+'.join(ops)}")


if __name__ == "__main__":
    main()
