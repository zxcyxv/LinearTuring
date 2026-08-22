#!/bin/bash
cd /workspace/LinearTuring
while pgrep -f "tag hop256_n1_frozen" > /dev/null; do sleep 20; done
python3 hop_task.py --tag hop256_n2_full   --n 2 --d 256 --steps 8000
python3 hop_task.py --tag hop256_n2_frozen --n 2 --d 256 --steps 8000 --freeze_A
echo "=== HOP256B DONE ==="
