cd /workspace/mnist_model1
xargs -a ca_jobs.txt -d '\n' -P 4 -I{} bash ca_worker.sh "{}"
echo "=== QUEUE13 DONE ==="
