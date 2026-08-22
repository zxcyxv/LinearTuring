#!/bin/bash
cd /workspace/LinearTuring
# Sanford et al. (ICML 2024) 레시피 그대로: Adam, lr 1e-4, 상수, bs 32, 1e5 step, d=128 L=2 H=4 (413K)
python3 hop_baseline.py --tag tfLIT_n1 --n 1 --L 2 --d 128 --nhead 4 \
  --lr 1e-4 --opt adam --sched const --bs 32 --steps 100000 --eval_every 2500
# 같은 lr/옵티마이저, 배치만 우리와 맞춤 (샘플 수 10.2M — 우리 4k×256=1M 의 10배)
python3 hop_baseline.py --tag tfLIT_n1_bs256 --n 1 --L 2 --d 128 --nhead 4 \
  --lr 1e-4 --opt adam --sched const --bs 256 --steps 40000 --eval_every 1000
echo "=== TF LIT DONE ==="
