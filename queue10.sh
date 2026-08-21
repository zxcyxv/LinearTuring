cd /workspace/mnist_model1
for R in 1 2 3 4 6 8; do
  python3 seq_task.py --tag pR${R}_full   --T 16 --R $R --steps 6000
  python3 seq_task.py --tag pR${R}_frozen --T 16 --R $R --steps 6000 --freeze_A
done
echo "=== R SWEEP DONE ==="
for D in 16 32; do
  python3 seq_task.py --tag pD${D}_full   --T 16 --R 4 --d $D --steps 6000
  python3 seq_task.py --tag pD${D}_frozen --T 16 --R 4 --d $D --steps 6000 --freeze_A
done
echo "=== QUEUE10 DONE ==="
