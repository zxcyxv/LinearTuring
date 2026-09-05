"""배정 수준의 중첩이 유지되는가. 182k, 512퍼즐, seg64. 최종 오답 칸(가짜 해)에서 정답 숫자의 확률과 순위, 최종 정답 칸에서 결정 시점의 확률 궤적."""
import os, numpy as np, torch, time
os.environ["SEGS"] = "64"
src = open("/workspace/LinearTuring/2026-09-05/analysis/winfer_exact.py").read().split('if __name__ == "__main__":')[0]
exec(src)
BSZ = 64; NB = 512; SEG = 64; segs_show = [1, 2, 4, 8, 16, 32, 64]
acc = {k: [] for k in ("wrong_ptrue", "wrong_rank2", "wrong_pmax", "late_ptrue", "early_ptrue")}
p_true_all = []; wrong_final_all = []; solved_seg_all = []
t0 = time.time()
for b in range(0, NB, BSZ):
    x = torch.from_numpy(X[b:b+BSZ].astype(np.int32)+1).cuda(); y = torch.from_numpy(Y[b:b+BSZ].astype(np.int32)+1).cuda().long()
    batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(len(x), dtype=torch.int32, device="cuda"))
    P = []; 
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
        carry = m.initial_carry(batch)
        for s in range(SEG):
            carry, out = m(carry, batch); P.append(torch.softmax(out["logits"].float(), -1))
    P = torch.stack(P)                                    # [S,B,T,V]
    pt = P.gather(-1, y[None, :, :, None].expand(SEG, -1, -1, 1)).squeeze(-1)   # 정답 확률 [S,B,T]
    pred = P.argmax(-1); correct = pred == y[None]        # [S,B,T]
    given = x > 1                                          # 단서 칸
    p_true_all.append(pt.cpu()); wrong_final_all.append((~correct[-1] & ~given[None][0]).cpu()); solved_seg_all.append(correct.all(-1).cpu())
    print(f"  {b+len(x)}/{NB}  {time.time()-t0:.0f}s", flush=True)
pt = torch.cat(p_true_all, 1); wf = torch.cat(wrong_final_all, 0); sol = torch.cat(solved_seg_all, 1)   # pt [S,N,T], wf [N,T], sol [S,N]
unsolved = ~sol[-1]; print(f"seg{SEG} 미해결 {int(unsolved.sum())}/{NB}, 그중 최종 오답 칸 {int(wf.sum())}개")
print(f"\n최종 오답 칸(가짜 해 칸)의 정답 확률 p(정답), 세그먼트별 중앙값 / 평균 / p<0.01 비율 / 정답이 2위 이내 비율")
for s in segs_show:
    v = pt[s-1][wf]; print(f"  seg{s:>3}: 중앙값 {v.median():.3f}  평균 {v.mean():.3f}  p<0.01 {(v<0.01).float().mean():.2f}", flush=True)
# 늦게 풀리는 퍼즐(seg16 미해결, seg64 해결)에서, seg16 시점 오답이던 칸의 정답 확률 궤적
late = (~sol[15]) & sol[-1]; early = sol[15]
pred16_wrong = (pt[15] < 0.5)                              # seg16 에 정답 확률 0.5 미만이던 칸
mask_late = late[:, None] & pred16_wrong; mask_early_wrong = None
print(f"\n늦게 풀린 퍼즐 {int(late.sum())}개: seg16 에 정답 확률<0.5 였다가 결국 맞게 되는 칸 {int(mask_late.sum())}개의 p(정답) 궤적")
for s in segs_show:
    v = pt[s-1][mask_late]; print(f"  seg{s:>3}: 중앙값 {v.median():.3f}  평균 {v.mean():.3f}  p<0.01 {(v<0.01).float().mean():.2f}", flush=True)
