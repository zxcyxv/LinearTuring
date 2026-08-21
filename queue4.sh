cd /workspace/mnist_model1
for g in 0.02 0.1 0.5 2.0 8.0; do
  t="g${g}"
  python3 train.py --tag $t --epochs 6 --log_every 600 --fix_gamma --gamma $g --rand_R 4,16 \
   && python3 interp.py --tag $t --Rmax 64 2>&1 | grep -vi "glyph\|substituting\|UserWarning\|fig.tight" | tail -4 \
   && python3 attractor.py $t 2>&1 | grep -vi glyph | tail -1
done
echo "=== QUEUE4 DONE ==="
