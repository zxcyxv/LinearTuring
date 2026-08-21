cd /workspace/mnist_model1
while kill -0 20077 2>/dev/null; do sleep 10; done
run(){ [ -f runs/$1_ca.json ] && { echo "skip $1"; return; }
       python3 ca_task.py --tag $1 --rule $2 --k $3 --T 32 --steps 6000 ${4:+--freeze_A} 2>&1 | tee runs/$1.log | tail -1; }
for k in 1 3 7 15; do
  run "ca90_k${k}_full"   90 $k
  run "ca90_k${k}_frozen" 90 $k 1
done
echo "=== RULE90 DONE ==="
for k in 1 2 4 8; do
  run "ca110_k${k}_full"   110 $k
  run "ca110_k${k}_frozen" 110 $k 1
done
echo "=== QUEUE12 DONE ==="
