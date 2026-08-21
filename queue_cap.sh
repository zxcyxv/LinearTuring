cd /workspace/mnist_model1
while [ $(ls runs/ca*_ca.json 2>/dev/null | wc -l) -lt 16 ]; do sleep 30; done
bash cap_worker.sh cap_jobs.txt
echo "=== CAP DONE ==="
