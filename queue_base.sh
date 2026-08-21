cd /workspace/mnist_model1
for rk in "90 3" "90 7" "110 4" "110 8"; do
  set -- $rk
  python3 ca_baseline.py --arch gru --L 2 --rule $1 --k $2 2>&1 | tail -1
  python3 ca_baseline.py --arch cnn --L $2 --rule $1 --k $2 2>&1 | tail -1
  python3 ca_baseline.py --arch cnn --L 2 --rule $1 --k $2 2>&1 | tail -1
done
echo "=== BASELINE DONE ==="
