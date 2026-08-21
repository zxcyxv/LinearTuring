cd /workspace/mnist_model1
while ! grep -q "CAP2 DONE" runs/queue_cap2.log 2>/dev/null; do sleep 30; done
run(){ [ -f runs/$1_ca.json ] && { echo "skip $1"; return; }
       python3 ca_task.py --tag $1 --rule 110 --k 8 --T 32 --tau $2 --steps 2000 --bs 256 $3 > runs/$1.log 2>&1
       tail -1 runs/$1.log; }
# 기준점(이미 있음): full τ=1 0.007 / full τ=4 0.131
run b110k8_noov_t4    4 "--no_ov"                    # 대조: noov, 경계 사영 없음
run b110k8_noovWO_t4  4 "--no_ov --boundary_wo"      # 본 실험
run b110k8_noovWO_t2  2 "--no_ov --boundary_wo"
echo "=== BWO DONE ==="
