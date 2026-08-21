cd /workspace/mnist_model1
while ! grep -q "QUEUE10 DONE" runs/queue10.log; do sleep 20; done
# 그록킹 시점은 시드 민감 → 임계 깊이에서 시드 복제로 분포를 본다
for R in 2 4; do
  for s in 1 2 3 4; do
    python3 seq_task.py --tag pR${R}_full_s${s}   --T 16 --R $R --steps 6000 --seed $s
    python3 seq_task.py --tag pR${R}_frozen_s${s} --T 16 --R $R --steps 6000 --seed $s --freeze_A
  done
done
echo "=== SEEDS DONE ==="
