cd /workspace/mnist_model1
until grep -q FINAL runs/b110k8_fullWO_t4_long.log 2>/dev/null; do sleep 30; done
run(){ [ -f runs/$1_ca.json ] && { echo "skip $1"; return; }
       python3 ca_task.py --tag $1 --rule 110 --k 8 --T 32 --tau 4 --steps 2000 --bs 256 --boundary_wo --wo_mode $2 > runs/$1.log 2>&1 &
}
run b110k8_woR_t4 residual; run b110k8_woO_t4 orth; wait
run b110k8_woC_t4 contract; run b110k8_woH_t4 perhead; wait
for t in woR woO woC woH; do tail -1 runs/b110k8_${t}_t4.log; done
echo "=== WOD DONE ==="
