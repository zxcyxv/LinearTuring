#!/bin/bash
# 클래스별 단독 8런 (k=4) → 8규칙 동시 1런 (k=4). 순차.
cd /workspace/LinearTuring
for r in 4 184 232 30 45 90 54 110; do
  python ca_multi.py --tag cam_k4_r$r --rules $r --k 4 --steps 3000 --compile 2>&1 | grep -E "FINAL|rule|step  ?(1000|2000|3000) " 
done
python ca_multi.py --tag cam_k4_all8 --rules 4,184,232,30,45,90,54,110 --k 4 --steps 6000 --compile 2>&1 | grep -E "FINAL|rule|step "
