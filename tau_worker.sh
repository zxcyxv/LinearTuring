cd /workspace/mnist_model1
while read -r line; do
  set -- $line
  tag=$1; rule=$2; k=$3; fz=$4; tau=$5
  [ -f runs/${tag}_ca.json ] && { echo "skip $tag"; continue; }
  mkdir runs/${tag}.lock 2>/dev/null || continue
  extra=""; [ "$fz" = "1" ] && extra="--freeze_A"
  python3 ca_task.py --tag $tag --rule $rule --k $k --T 32 --tau $tau --steps 2000 --bs 256 $extra > runs/${tag}.log 2>&1
  rmdir runs/${tag}.lock 2>/dev/null
  tail -1 runs/${tag}.log
done < "$1"
