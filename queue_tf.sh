#!/bin/bash
cd /workspace/LinearTuring
for lr in 2e-3 1e-3; do
  python3 hop_baseline.py --tag tf_L1_n1_lr$lr --n 1 --L 1 --d 184 --lr $lr --steps 4000
  python3 hop_baseline.py --tag tf_L2_n1_lr$lr --n 1 --L 2 --d 128 --lr $lr --steps 4000
  python3 hop_baseline.py --tag tf_L4_n1_lr$lr --n 1 --L 4 --d 92  --lr $lr --steps 4000
done
echo "=== TF BASELINE n1 DONE ==="
