cd /workspace/mnist_model1
while pgrep -f "seq_task.py --tag par32_frozen" > /dev/null; do sleep 10; done
for T in 20 24 28; do
  python3 seq_task.py --tag par${T}_full   --T $T --steps 6000
  python3 seq_task.py --tag par${T}_frozen --T $T --steps 6000 --freeze_A
done
echo "=== QUEUE9 DONE ==="
