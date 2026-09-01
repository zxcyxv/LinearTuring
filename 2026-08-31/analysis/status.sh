#!/bin/bash
# 런 상태 한 눈에.  사용: bash 2026-08-31/analysis/status.sh [런이름]
R=${1:-R1B8_min_faith}; ROOT=${LT_ROOT:-/root/LinearTuring}; cd "$ROOT"
echo "=== $R ==="
if pgrep -f "run_name=$R" >/dev/null; then
  echo "상태: 실행 중 (pid $(pgrep -f "run_name=$R" | head -1), 경과 $(ps -o etime= -p $(pgrep -f "run_name=$R" | head -1) | tr -d ' '))"
else
  echo "상태: !!! 실행 중 아님"; tr '\r' '\n' < results/logs/$R.log | tail -15
fi
tr '\r' '\n' < results/logs/$R.log | grep -oE "[0-9]+/390625 \[[0-9:]+<[0-9:]+, *[0-9.]+it/s" | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo
echo "=== held-out vs 기준선(stdp1) ==="
python3 - "$R" <<'PY'
import csv, re, sys
R=sys.argv[1]
base={int(r['step']):(float(r['acc']),int(r['exact'])) for r in csv.DictReader(open("results/curves/R1B8_min_stdp1_eval.csv"))}
mini={int(r['step']):(float(r['acc']),int(r['exact'])) for r in csv.DictReader(open("results/curves/R1B8_min_eval.csv"))}
t=open(f"results/logs/{R}.log",'rb').read().decode('utf8','ignore').replace('\r','\n')
ev=sorted({(int(m.group(1)),float(m.group(2)),int(m.group(3))) for m in re.finditer(r"\[EVAL\] step (\d+)\s+set all\s+acc ([\d.]+)\s+exact (\d+)/", t)})
if not ev: print("  아직 eval 없음"); raise SystemExit
print(f"{'step':>7} | {'acc':>7} {'ex':>5} | {'stdp1':>7} {'ex':>4} {'Δ':>7} | {'minimal':>8} {'ex':>4} {'Δ':>7}")
for s,a,e in ev:
    ba,be=base.get(s,(float('nan'),0)); ma,me=mini.get(s,(float('nan'),0))
    f=lambda x,y: f"{x:>7.4f} {y:>4}" if x==x else f"{'—':>7} {'—':>4}"
    g=lambda x: f"{a-x:>+7.4f}" if x==x else f"{'—':>7}"
    print(f"{s:>7} | {a:>7.4f} {e:>5} | {f(ba,be)} {g(ba)} | {f(ma,me)} {g(ma)}")
print("\n※ 넘어야 할 기준은 minimal(기억 없음) 이다 — stdp1 이 아니라.")
print("※ minimal 은 7,812 에서 중단됐다: 1953 .4939/0  3906 .5829/0  5859 .6331/27  7812 .6676/52")
print("※ 39,060 이전에는 완답이 계단 이전이라 평평 — 셀 acc 로만 판정")
PY
echo
echo "=== λ·η(δ)·G·β 헤드별 (최근 5개 체크포인트) ==="
python3 2026-08-31/analysis/track_lam.py refs/URM/checkpoints/$R 2>&1 | head -1
python3 2026-08-31/analysis/track_lam.py refs/URM/checkpoints/$R 2>&1 | tail -5
echo
echo "※ η(δ) 가 0.05 근처면 긴 기억 유지, 0.5 로 오르면 붕괴 (stdp1 이 그랬음)"
echo "※ G 가 1 을 넘으면 모델이 축적을 원하는 것"
echo "※ λ 가 0.25 근처면 기억이 소수 성분 — 이식 실험의 최적과 일치"
