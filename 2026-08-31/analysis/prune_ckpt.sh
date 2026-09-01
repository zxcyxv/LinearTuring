#!/bin/bash
# 체크포인트를 최신 KEEP 개만 남기고 지운다 (config.yaml 은 건드리지 않음).
# 사용: KEEP=10 INTERVAL=300 bash prune_ckpt.sh <체크포인트디렉터리>
D=${1:?체크포인트 디렉터리}; KEEP=${KEEP:-10}; INTERVAL=${INTERVAL:-300}
echo "[prune] $D  최신 $KEEP 개 유지, ${INTERVAL}초 주기  시작 $(date '+%F %T')"
while true; do
  if [ -d "$D" ]; then
    # step 번호 기준 정렬 → 오래된 것부터 초과분 삭제
    N=$(ls "$D"/step_*.pt 2>/dev/null | wc -l)
    if [ "$N" -gt "$KEEP" ]; then
      ls "$D"/step_*.pt 2>/dev/null | sed 's/.*step_\([0-9]*\)\.pt/\1 &/' | sort -n | head -n -"$KEEP" | cut -d' ' -f2- | while read -r f; do
        rm -f "$f" && echo "[prune] $(date '+%F %T') 삭제 $(basename "$f")"
      done
    fi
  fi
  # 학습이 끝났고 더 늘지 않으면 종료
  pgrep -f "pretrain.py" >/dev/null || { echo "[prune] 학습 종료 감지, prune 종료 $(date '+%F %T')"; exit 0; }
  sleep "$INTERVAL"
done
