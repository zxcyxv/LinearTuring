"""결합 기억 (위상 STDP / 적응 Kuramoto 규칙) 을 학습 없이 추론에 얹는다.  STDP.md 참조.
  매 블록 스텝:  a = attn(h);  w ← w + η (a − w)  (w 초기 = 첫 a);  결합 = (1−λ) a + λ w
사용: python stdp_infer.py [--n 2048] [--eta 0.1 0.05] [--lam 0.25 0.5] [--loops 16]"""
import argparse, numpy as np, torch
from common import load_lt, load_test, make_batch, logits, CKPT_DEFAULT
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--n", type=int, default=2048); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--eta", type=float, nargs="+", default=[0.1, 0.05]); ap.add_argument("--lam", type=float, nargs="+", default=[0.25, 0.5]); ap.add_argument("--loops", type=int, default=16)
args = ap.parse_args(); torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=True, batch_size=args.bs); inner = m.inner; core = inner.core; K = m.config.blocks_per_seg
inp, lab, _ = load_test(args.n); N = len(inp); I = inp.cpu().numpy(); G = lab.cpu().numpy() - 2; bl = I == 1; fc = core.kernel_fast(); AB = core.W_C()

def run(eta, lam):
    P = np.zeros((N, 81), int)
    for b in range(0, N, args.bs):
        x = inp[b:b + args.bs]; n = len(x); inj = inner._injection(make_batch(x, x)); h = inner.init_hidden.expand(n, 81, -1).clone(); w = None
        for _ in range(args.loops):
            for _ in range(K):
                h = inner._boundary(h) + inner.inj_gate * inj; hp = core.phi(h, .5); a, *_ = core.attn_fast(hp, *fc, AB=AB)
                w = a.clone() if w is None else w + eta * (a - w)
                f, *_ = core.field(hp, None, None, a_fixed=(1 - lam) * a + lam * w, AB=AB, fast_ctx=fc); h = core.phi(hp + f, .5)
        P[b:b + n] = logits(m, h)[:, :, 2:11].argmax(-1).cpu().numpy()
    return int(((P == G) | ~bl).all(1).sum()), float((P == G)[bl].mean())
ex, c = run(1.0, 0.0); print(f"기준 (w 없음)        완답 {ex:4d}  셀 {c:.4f}", flush=True)
for eta in args.eta:
    for lam in args.lam:
        ex, c = run(eta, lam); print(f"η={eta:.3g} λ={lam:.2f}       완답 {ex:4d}  셀 {c:.4f}", flush=True)
