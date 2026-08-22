#!/bin/bash
cd /workspace/LinearTuring
# 어휘 크기가 원인인지: V=8 (논문의 |S|=4 에 가까움) 로 트랜스포머 재시도
python3 hop_baseline.py --tag tfV8_n1 --n 1 --V 8 --L 2 --d 128 --nhead 4 \
  --lr 1e-4 --opt adam --sched const --bs 256 --steps 8000 --eval_every 500
# 그래도 실패하면 lr 을 올려서 (OneCycle 2e-3) 한 번 더
python3 hop_baseline.py --tag tfV8_n1_hi --n 1 --V 8 --L 2 --d 128 --nhead 4 \
  --lr 1e-3 --bs 256 --steps 8000 --eval_every 500
# 우리 모델도 V=8 로 — 같은 조건 비교
python3 hop_task.py --tag hopV8_n1 --n 1 --V 8 --d 256 --steps 8000 --kernel_lr_mult 1
echo "=== DIAG DONE ==="
