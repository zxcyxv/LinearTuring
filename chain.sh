cd /workspace/mnist_model1
for t in noov noisy; do
  while [ ! -f runs/${t}_log.json ]; do sleep 15; done
  sleep 3
  echo "=== interp $t ==="
  python3 interp.py --tag $t --Rmax 64 2>&1 | grep -vi "glyph\|substituting\|UserWarning\|fig.tight"
done
echo "=== CHAIN DONE ==="
