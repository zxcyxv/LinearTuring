cd /workspace/mnist_model1
while ! grep -q "BWO DONE" runs/queue_bwo.log 2>/dev/null; do sleep 30; done
run(){ [ -f runs/$1_ca.json ] && { echo "skip $1"; return; }
       python3 ca_task.py --tag $1 --rule 110 --k 8 --T 32 --tau $2 --steps 2000 --bs 256 $3 > runs/$1.log 2>&1
       tail -1 runs/$1.log; }
run b110k8_fullWO_t4 4 "--boundary_wo"     # full + 경계 W_O : 2x2 의 네 번째 칸
echo "=== BWO2 DONE ==="
