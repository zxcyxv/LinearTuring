#!/bin/bash
cd /workspace/LinearTuring
# 기준선: 425K (L=4 d=92) — Model1 427,330 에 가장 근접. lr 2개만.
python3 hop_baseline.py --tag tf425_n1_lr2e-3 --n 1 --L 4 --d 92 --lr 2e-3 --steps 4000
python3 hop_baseline.py --tag tf425_n1_lr1e-3 --n 1 --L 4 --d 92 --lr 1e-3 --steps 4000
# 커널 lr 배수가 hop 과제에서 실제로 일을 하는지 — x30 은 MNIST 에서 온 미검증 기본값
python3 hop_task.py --tag hop256_n1_full_k1 --n 1 --d 256 --steps 4000 --kernel_lr_mult 1
echo "=== TF425 + KERNEL x1 DONE ==="
