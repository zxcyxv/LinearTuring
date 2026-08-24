#!/bin/bash
# 새 400-step 로그 라인이 생기면 종료 — 마지막 관측 스텝을 $1 로 받음
L=/workspace/LinearTuring/sudoku_runs/2026-08-24/LSH1040_B24.log
LAST=$1
while true; do
  CUR=$(tr '\r' '\n' < $L | grep -oE "step [0-9]+  lm_loss [0-9.]+  acc 0\.[1-9][0-9]*" | tail -1)
  STEP=$(echo $CUR | awk '{print $2}')
  if [ -n "$STEP" ] && [ "$STEP" -gt "$LAST" ]; then
    tr '\r' '\n' < $L | grep -oE "step [0-9]+  lm_loss [0-9.]+  acc 0\.[1-9][0-9]*" | awk '{printf "%6d %s\n", $2, $6}' | tail -3
    tr '\r' '\n' < $L | grep EVAL | tail -1
    exit 0
  fi
  sleep 15
done
