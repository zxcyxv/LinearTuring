"""182k: w 의 시정수를 바꿔(δ 배율) 외삽 이득이 어떻게 변하는가. 기억(느린 w)인지 추적(빠른 w)인지 가른다. 512퍼즐 seg128."""
import os, numpy as np, torch, time
os.environ["SEGS"] = "128"
src = open("/workspace/LinearTuring/2026-09-05/analysis/winfer_exact.py").read().split('if __name__ == "__main__":')[0]
exec(src)
BSZ = 64; NB = 512; SEG = 128
eta0 = [L.eta_raw.data.clone() for L in layers]
def set_delta(mult):
    for L, e0 in zip(layers, eta0):
        d0 = torch.sigmoid(e0); d = (d0 * mult).clamp(1e-4, 0.999); L.eta_raw.data = torch.log(d / (1 - d))
def run(tag):
    ex = np.zeros(SEG); t0 = time.time()
    for b in range(0, NB, BSZ):
        x = torch.from_numpy(X[b:b+BSZ].astype(np.int32)+1).cuda(); y = torch.from_numpy(Y[b:b+BSZ].astype(np.int32)+1).cuda().long()
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(len(x), dtype=torch.int32, device="cuda"))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
            carry = m.initial_carry(batch)
            for s in range(SEG): carry, out = m(carry, batch); ex[s] += (out["logits"].argmax(-1) == y).all(-1).sum().item()
    print(f"  [{tag}] {time.time()-t0:.0f}s  seg16 {int(ex[15])} seg32 {int(ex[31])} seg64 {int(ex[63])} seg128 {int(ex[-1])}  이득 {int(ex[-1]-ex[15]):+d}", flush=True)
    return ex
print("δ 원값: L0 %.4f  L1 %.4f  (시정수 %.0f / %.0f 블록)" % (torch.sigmoid(eta0[0]).mean(), torch.sigmoid(eta0[1]).mean(), 1/torch.sigmoid(eta0[0]).mean(), 1/torch.sigmoid(eta0[1]).mean()))
for mult in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
    set_delta(mult); run(f"δ×{mult:<4}")
