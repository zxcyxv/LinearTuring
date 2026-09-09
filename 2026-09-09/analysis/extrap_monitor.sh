#!/bin/bash
# 지정한 스텝 구간에서 세그먼트 외삽을 여러 체크포인트에 대해 자동 측정.
#   단일 체크포인트 변동이 ±5%p 라 (92k 4.6% vs 98k 13.9%) 반드시 복수 표본을 받는다.
# 사용: nohup bash extrap_monitor.sh <TRAIN_PID> <목표1> [목표2 ...] > /dev/null 2>&1 &
set -u
REPO=/workspace/LinearTuring
LOGF=$REPO/2026-09-09/results/extrap_monitor.log
OUT=$REPO/2026-09-09/results/extrap_samples.txt
PID=${1:-0}; shift
mkdir -p "$(dirname "$LOGF")"
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOGF"; }
cur(){ cat $REPO/2026-09-09/train_v17*.log 2>/dev/null | grep -o 'step [0-9]*' | tail -1 | grep -o '[0-9]*'; }
say "monitor 시작 (pid=$PID, 목표: $*)"
for TGT in "$@"; do
  while :; do
    S=$(cur); S=${S:-0}
    [ "$S" -ge "$TGT" ] && break
    if [ "$PID" != "0" ] && ! kill -0 "$PID" 2>/dev/null; then say "학습 종료 감지 — 남은 목표 포기"; exit 0; fi
    sleep 120
  done
  CK=$(ls -t $REPO/2026-09-09/run_v17/step_*.pt 2>/dev/null | head -1)
  say "목표 $TGT 도달 (현재 $(cur)) — $(basename "$CK") 외삽 시작"
  R=$(cd $REPO && python 2026-09-09/analysis/seg_extrap.py 2>&1)
  echo "===== 목표 $TGT =====" >> "$OUT"; echo "$R" | grep -E "^# ckpt|^seg16|^오차" >> "$OUT"
  say "완료: $(echo "$R" | grep '^오차' | tr -d '\n')"
done
say "monitor 종료"
# 요약
{ echo; echo "=== 요약 ==="; grep -E "^# ckpt|^오차" "$OUT" | paste - - ; } >> "$OUT" 2>/dev/null
