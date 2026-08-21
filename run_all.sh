set -e
python3 train.py --tag full  --epochs 8 --log_every 200
python3 train.py --tag noov  --epochs 8 --log_every 200 --no_ov --lam_mode diag
python3 train.py --tag noisy --epochs 8 --log_every 200 --noisy
