#!/bin/bash
cd /workspace/LinearTuring
# 임베딩 스케일 버그 수정 후 재실행. V=64 (원 과제), 논문 레시피
python3 hop_baseline.py --tag tfFIX_n1 --n 1 --L 2 --d 128 --nhead 4 \
  --lr 1e-4 --opt adam --sched const --bs 256 --steps 20000 --eval_every 500
# lr 을 올린 판도 함께 (과소조정 방지)
python3 hop_baseline.py --tag tfFIX_n1_hi --n 1 --L 2 --d 128 --nhead 4 \
  --lr 1e-3 --bs 256 --steps 20000 --eval_every 500
echo "=== TF FIX DONE ==="
