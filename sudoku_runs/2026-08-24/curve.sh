#!/bin/bash
# $1=로그 $2=라벨 — 훈련 셀 acc 곡선 (acc 0.0000 인 절반-로그는 제외)
echo "== $2 =="
tr '\r' '\n' < "$1" | grep -oE "step [0-9]+  lm_loss [0-9.]+  acc 0\.[1-9][0-9]*" | awk '{printf "%6d %s\n", $2, $6}'
