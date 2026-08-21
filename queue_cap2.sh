cd /workspace/mnist_model1
while ! grep -q "TAU DONE" runs/queue_tau.log 2>/dev/null; do sleep 30; done
while read -r line; do
  set -- $line
  tag=$1; rule=$2; k=$3; fz=$4; d=$5
  [ -f runs/${tag}_ca.json ] && continue
  extra=""; [ "$fz" = "1" ] && extra="--freeze_A"
  python3 ca_task.py --tag $tag --rule $rule --k $k --T 32 --d $d --steps 2000 --bs 256 $extra > runs/${tag}.log 2>&1
  tail -1 runs/${tag}.log
done < cap2_jobs.txt
echo "=== CAP2 DONE ==="
