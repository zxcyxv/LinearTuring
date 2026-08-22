#!/bin/bash
cd /workspace/LinearTuring
python3 hop_task.py --tag hop256_n1_full   --n 1 --d 256 --steps 4000
python3 hop_task.py --tag hop256_n1_frozen --n 1 --d 256 --steps 4000 --freeze_A
python3 hop_task.py --tag hop256_n2_full   --n 2 --d 256 --steps 4000
python3 hop_task.py --tag hop256_n2_frozen --n 2 --d 256 --steps 4000 --freeze_A
echo "=== HOP256 DONE ==="
