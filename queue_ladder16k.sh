#!/bin/bash
cd /workspace/LinearTuring
# 확정 사다리: 전 런 동일 예산 16000 step, 커널 lr x1 (hop 에서 무의미함이 확인됨)
python3 hop_task.py --tag L16_n1_frozen --n 1 --d 256 --steps 16000 --kernel_lr_mult 1 --freeze_A
python3 hop_task.py --tag L16_n2_full   --n 2 --d 256 --steps 16000 --kernel_lr_mult 1
python3 hop_task.py --tag L16_n2_frozen --n 2 --d 256 --steps 16000 --kernel_lr_mult 1 --freeze_A
python3 hop_task.py --tag L16_n1_full   --n 1 --d 256 --steps 16000 --kernel_lr_mult 1
echo "=== LADDER16K DONE ==="
