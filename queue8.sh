cd /workspace/mnist_model1
for T in 16 32; do
  python3 seq_task.py --tag par${T}_full    --T $T --steps 6000
  python3 seq_task.py --tag par${T}_frozen  --T $T --steps 6000 --freeze_A
done
echo "=== QUEUE8 DONE ==="
