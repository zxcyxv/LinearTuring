cd /workspace/mnist_model1
for c in runs/chunk_?.txt; do bash ca_worker.sh "$c" & done
wait
echo "=== QUEUE14 DONE ==="
