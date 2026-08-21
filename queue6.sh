cd /workspace/mnist_model1
# 게이지 가설의 예측: b 를 제거하면 γ 는 동역학에 아무 영향이 없어야 한다 → λ 가 평평
for g in 0.02 2.0 128.0; do
  t="nb_g${g}"
  python3 train.py --tag $t --epochs 6 --log_every 600 --fix_gamma --gamma $g --rand_R 4,16 --no_bias_v \
   && python3 interp.py --tag $t --Rmax 64 2>&1 | grep -vi "glyph\|substituting\|UserWarning\|fig.tight" | tail -3 \
   && python3 attractor.py $t 2>&1 | grep -vi glyph | tail -1
done
echo "=== QUEUE6 DONE ==="
