cd /workspace/mnist_model1
while [ ! -f runs/CAP_DONE ]; do
  grep -q "CAP DONE" runs/queue_cap.log 2>/dev/null && touch runs/CAP_DONE
  sleep 30
done
split -n r/2 -d tau_jobs.txt runs/tchunk_
for c in runs/tchunk_0*; do bash tau_worker.sh "$c" & done
wait
echo "=== TAU DONE ==="
