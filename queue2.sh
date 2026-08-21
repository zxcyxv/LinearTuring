cd /workspace/mnist_model1
while [ ! -f runs/noisy_interp.json ]; do sleep 10; done
run(){ python3 train.py --tag $1 --epochs 8 --log_every 200 "${@:2}" \
   && python3 interp.py --tag $1 --Rmax 64 2>&1 | grep -vi "glyph\|substituting\|UserWarning\|fig.tight" \
   && python3 dt_sweep.py $1 2>&1 | grep -vi glyph; }
echo "### psi0 : ψ≡0 고정 (§2.4 검증)"      ; run psi0  --psi_zero
echo "### kern : 커널 파라미터 학습률 ×30"  ; run kern  --kernel_lr_mult 30
echo "### randR: 매 배치 R~U(4,16), τ=1 고정"; run randR --rand_R 4,16
echo "=== QUEUE2 DONE ==="
