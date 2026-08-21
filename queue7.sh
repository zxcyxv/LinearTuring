cd /workspace/mnist_model1
run(){ python3 train.py --tag $1 --epochs 8 --log_every 600 "${@:2}" \
   && python3 interp.py --tag $1 --Rmax 64 2>&1 | grep -vi "glyph\|substituting\|UserWarning\|fig.tight" | tail -3 \
   && python3 attractor.py $1 2>&1 | grep -vi glyph | tail -1; }
# A 를 h^(0) 에서 고정 → h 에 대해 선형 → 카오스 원리적 불가. 입력 비선형성은 유지.
echo "### frozenA : A(h^0) 고정, dt 무작위화"; run frozenA  --freeze_A --rand_R 4,16
echo "### frozenA8: A(h^0) 고정, 고정 R=8"   ; run frozenA8 --freeze_A
python3 robustness.py frozenA frozenA8 randR 2>&1 | grep -vi glyph
echo "=== QUEUE7 DONE ==="
