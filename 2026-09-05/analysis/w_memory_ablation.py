"""182k(이전 규칙) 외삽 이득이 w 기억에 의존하는가. 512퍼즐, seg128.
변형: 기준 | w 동결(seg>16 부터 갱신 중지) | w 매 세그먼트 초기화(기억 시정수 ≤ 1세그) | w 매 블록 초기화(기억 없음, w=Γ 즉석)."""
import os, numpy as np, torch, time
os.environ["SEGS"] = "128"
src = open("/workspace/LinearTuring/2026-09-05/analysis/winfer_exact.py").read().split('if __name__ == "__main__":')[0]
exec(src)
BSZ = 64; NB = 512; SEG = 128
orig_step = I.step; mode = {"m": "base", "seg": 0}
def step_patched(L, h, AB, kc, w=None, fresh=None, kcb=None, vprev=None, apply_phi=True):
    if mode["m"] == "freeze" and mode["seg"] >= 16 and w is not None:
        hout, _ = orig_step(L, h, AB, kc, w, fresh, kcb, vprev, apply_phi); return hout, w          # 갱신 버림
    if mode["m"] == "block_reset" and w is not None:
        hout, w2 = orig_step(L, h, AB, kc, None, None, kcb, vprev, apply_phi); return hout, w2      # 매 블록 w=Γ 즉석
    return orig_step(L, h, AB, kc, w, fresh, kcb, vprev, apply_phi)
I.step = step_patched
def run(m_):
    mode["m"] = m_; ex = np.zeros(SEG); t0 = time.time()
    for b in range(0, NB, BSZ):
        x = torch.from_numpy(X[b:b+BSZ].astype(np.int32)+1).cuda(); y = torch.from_numpy(Y[b:b+BSZ].astype(np.int32)+1).cuda().long()
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(len(x), dtype=torch.int32, device="cuda"))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
            carry = m.initial_carry(batch)
            for s in range(SEG):
                mode["seg"] = s
                if m_ == "seg_reset" and s > 0: carry = replace(carry, fresh=torch.ones(len(x), dtype=torch.bool, device="cuda"))
                carry, out = m(carry, batch); ex[s] += (out["logits"].argmax(-1) == y).all(-1).sum().item()
        print(f"  [{m_}] {b+len(x)}/{NB}  {time.time()-t0:.0f}s  누적 seg16 {int(ex[15])} seg128 {int(ex[-1])}", flush=True)
    return ex
from dataclasses import replace
res = {k: run(k) for k in ["base", "freeze", "seg_reset", "block_reset"]}
print(f"\n{'변형':>14} | seg16 | seg32 | seg64 | seg128 | 이득(16→128)")
for k, e in res.items(): print(f"{k:>14} | {int(e[15]):5d} | {int(e[31]):5d} | {int(e[63]):5d} | {int(e[127]):6d} | {int(e[127]-e[15]):+d}")
