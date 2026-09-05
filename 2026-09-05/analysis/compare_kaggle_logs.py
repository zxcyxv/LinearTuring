"""두 캐글 로그를 같은 스텝끼리 나란히: [LT] step 줄(train batch acc, lm_loss)과 [EVAL] 줄. 사용: python3 compare_kaggle_logs.py A.log B.log [라벨A 라벨B]"""
import re, sys
def parse(p):
    t = open(p, encoding="utf-8", errors="ignore").read().replace("\r", "\n")
    tr = {int(s): (float(l), float(a)) for s, l, a in re.findall(r"\[LT\] step (\d+)\s+lm_loss ([\d.]+).*?acc ([\d.]+)", t)}
    ev = {int(s): (float(a), int(e), int(n)) for s, a, e, n in re.findall(r"\[EVAL\] step (\d+)\s+acc ([\d.]+)\s+exact (\d+)/(\d+)", t)}
    return tr, ev
A, B = parse(sys.argv[1]), parse(sys.argv[2]); la, lb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
steps = sorted(set(A[0]) & set(B[0]))
print(f"{'step':>6} | {la+' loss':>10} {la+' acc':>8} | {lb+' loss':>10} {lb+' acc':>8} | Δacc")
for s in steps: print(f"{s:>6} | {A[0][s][0]:10.4f} {A[0][s][1]:8.4f} | {B[0][s][0]:10.4f} {B[0][s][1]:8.4f} | {B[0][s][1]-A[0][s][1]:+.3f}")
if steps:
    import statistics as st
    print(f"공통 {len(steps)}스텝 train acc 평균: {la} {st.mean(A[0][s][1] for s in steps):.4f}  {lb} {st.mean(B[0][s][1] for s in steps):.4f}")
print("\n[EVAL]"); 
for s in sorted(set(A[1]) | set(B[1])):
    a = A[1].get(s); b = B[1].get(s)
    print(f"{s:>6} | {la}: " + (f"acc {a[0]:.4f} exact {a[1]}/{a[2]}" if a else "-") + f" | {lb}: " + (f"acc {b[0]:.4f} exact {b[1]}/{b[2]}" if b else "-"))
