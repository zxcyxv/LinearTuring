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
