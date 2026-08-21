cd /workspace/mnist_model1
while ! grep -q "QUEUE2 DONE" runs/queue2.log; do sleep 20; done
run(){ python3 train.py --tag $1 --epochs 8 --log_every 400 "${@:2}" \
   && python3 interp.py --tag $1 --Rmax 64 2>&1 | grep -vi "glyph\|substituting\|UserWarning\|fig.tight" \
   && python3 dt_sweep.py $1 2>&1 | grep -vi glyph; }
# ψ=0 대칭 + W_OV 제거 + Λ 제거 + b 제거 = 사양이 "gradient flow" 라 부른 조건 전부
echo "### gradflow : ψ=0, W_OV=I, Λ=0, b=0"; run gradflow --psi_zero --no_ov --lam_mode none --no_bias_v
# 위에서 Λ 하나만 되살림 → Λ 가 진동의 원인인지 격리
echo "### lamonly  : 위 + Λ(full) 복원"     ; run lamonly  --psi_zero --no_ov --lam_mode full --no_bias_v
echo "=== QUEUE3 DONE ==="
