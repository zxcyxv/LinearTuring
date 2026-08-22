#!/bin/bash
# n-hop induction 사다리: n ∈ {1,2,3,4} × {A 자유, A 고정}
cd /workspace/LinearTuring
for n in 1 2 3 4; do
  python3 hop_task.py --tag hop_n${n}_full   --n $n --steps 2000
  python3 hop_task.py --tag hop_n${n}_frozen --n $n --steps 2000 --freeze_A
done
echo "=== HOP LADDER DONE ==="
