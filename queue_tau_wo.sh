#!/bin/bash
cd /workspace/LinearTuring
# 주: tau=4 + 경계 W_O (어제 우승 모드: 수축 init rho=e^-lambda ~ 0.18)
python3 hop_task.py --tag Htau4_n2_wo --n 2 --d 256 --tau 4 --steps 16000 \
  --kernel_lr_mult 1 --boundary_wo --wo_mode contract
# 대조: tau=4 경계 없음 — W_O 가 차이를 만드는지 격리
python3 hop_task.py --tag Htau4_n2_plain --n 2 --d 256 --tau 4 --steps 16000 --kernel_lr_mult 1
# tau=2 + W_O : 깊이 요구량이 정확히 얼마인지
python3 hop_task.py --tag Htau2_n2_wo --n 2 --d 256 --tau 2 --steps 16000 \
  --kernel_lr_mult 1 --boundary_wo --wo_mode contract
echo "=== TAU_WO DONE ==="
