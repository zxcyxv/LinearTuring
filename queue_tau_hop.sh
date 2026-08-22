#!/bin/bash
cd /workspace/LinearTuring
# 이론 예측: 커널 기반은 L >= k. tau 가 우리 깊이 손잡이.
python3 hop_task.py --tag Ltau4_n2_full --n 2 --d 256 --tau 4 --steps 16000 --kernel_lr_mult 1
python3 hop_task.py --tag Ltau2_n2_full --n 2 --d 256 --tau 2 --steps 16000 --kernel_lr_mult 1
echo "=== TAU HOP DONE ==="
