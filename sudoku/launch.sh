#!/bin/bash
# 새 런 발사.  사용: bash sudoku/launch.sh <RUN_NAME> [추가 hydra 인자...]
#   예) bash sudoku/launch.sh R1B8_bilin_s1 seed=1
#       bash sudoku/launch.sh R1B8_swiglu arch.bilinear=false
# 환경변수: ARCH(lt | minimal) D(폭, 832) DATA(데이터 경로) EPOCHS(50000 = 390,600 step) CKPT AMP
# 1 iter = 250 에폭 = 1,953 step. eval 은 held-out 2048 (truncate_test.py), loops 16.
# 발사 직후 확인: checkpoints/<RUN>/config.yaml 의 arch 플래그, 첫 체크포인트 파라미터 키.
set -e
ROOT=${LT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; export LT_ROOT=$ROOT
RN=${1:?런 이름}; shift
# [2026-08-31] 발사 전 학습 데이터 증강 확인 — NAUG=0 으로 빌드한 분석용 데이터로 학습을 돌린 사고 방지
DATA_DIR=${DATA:-data/sudoku-extreme-1k-aug-1000}
NTRAIN=$(cd "$ROOT/refs/URM" && python3 -c "import numpy as np;print(np.load('$DATA_DIR/train/all__inputs.npy',mmap_mode='r').shape[0])" 2>/dev/null || echo 0)
if [ "${SKIP_DATA_CHECK:-0}" != "1" ] && [ "$NTRAIN" -lt 100000 ]; then
  echo "!!! 학습 데이터 증강 확인 실패: $DATA_DIR/train 예제 $NTRAIN 개 (10만 미만)"
  echo "    증강 빌드: cd refs/URM && PYTHONPATH=. python data/build_sudoku_dataset.py \\"
  echo "               --output-dir $DATA_DIR --subsample-size 1000 --num-aug 1000"
  echo "    의도한 것이면 SKIP_DATA_CHECK=1 로 우회"
  exit 1
fi
echo "학습 데이터 $NTRAIN 예제 확인"
mkdir -p "$ROOT/results/logs"; OUT=$ROOT/results/logs/${RN}.log
cd "$ROOT/refs/URM"
WANDB_MODE=offline OMP_NUM_THREADS=4 nohup torchrun --nproc-per-node 1 --master_port=$((29500+RANDOM%400)) pretrain.py \
  data_path=${DATA:-data/sudoku-extreme-1k-aug-1000} \
  arch=${ARCH:-lt} arch.hidden_size=${D:-832} arch.ckpt=${CKPT:-false} arch.amp=${AMP:-true} \
  "$@" \
  epochs=${EPOCHS:-50000} eval_interval=250 evaluators=[] \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
  +run_name=$RN +checkpoint_path=checkpoints/$RN +ema=True > "$OUT" 2>&1 &
echo "$RN 발사 pid $!  로그: $OUT"
