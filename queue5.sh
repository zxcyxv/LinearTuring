cd /workspace/mnist_model1
for g in 32.0 128.0; do
  t="g${g}"
  python3 train.py --tag $t --epochs 6 --log_every 600 --fix_gamma --gamma $g --rand_R 4,16 \
   && python3 interp.py --tag $t --Rmax 64 2>&1 | grep -vi "glyph\|substituting\|UserWarning\|fig.tight" | tail -3 \
   && python3 attractor.py $t 2>&1 | grep -vi glyph | tail -1
done
python3 robustness.py g0.02 g0.1 g0.5 g2.0 g8.0 g32.0 g128.0 full randR 2>&1 | grep -vi glyph
echo "=== QUEUE5 DONE ==="
