#!/bin/bash
# 체크포인트 재개.  사용: CKPT_PATH=<파일> EPOCHS=<남은 iters×250> bash sudoku/resume.sh <RUN_NAME> [hydra 인자...]
#   예) CKPT_PATH=checkpoints/R1B8_bilin_r2/step_123039.pt EPOCHS=34250 bash sudoku/resume.sh R1B8_bilin_r2
# EPOCHS 산술: 총 200 iters = 390,600 step. 완료 iters = step/1953, EPOCHS = (200 − 완료)×250.
#   lr_min_ratio=1.0 (상수 lr) 이라 EPOCHS 는 스케줄에 영향 없음.
# 함정 (HANDOVER.md §3):
#   - CKPT_PATH 는 파일 경로. 디렉터리 지정은 업스트림 정규식 버그로 불가.
#   - 옵티마이저 없는 EMA-only 체크포인트면 LOAD_OPT=False.
#   - 재개 후 첫 체크포인트의 raw_model_state_dict 가 로드값과 달라야 정상 (같으면 무학습 — assign 패치 확인).
set -e
ROOT=${LT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; export LT_ROOT=$ROOT
RN=${1:?런 이름}; shift
OUT=$ROOT/results/logs/${RN}.log
cd "$ROOT/refs/URM"
WANDB_MODE=offline OMP_NUM_THREADS=4 nohup torchrun --nproc-per-node 1 --master_port=$((29500+RANDOM%400)) pretrain.py \
  data_path=${DATA:-data/sudoku-extreme-1k-aug-1000} \
  arch=${ARCH:-lt} arch.hidden_size=${D:-832} arch.ckpt=${CKPT:-false} arch.amp=${AMP:-true} \
  "$@" \
  epochs=${EPOCHS:?남은 에폭} eval_interval=250 evaluators=[] \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
  +load_checkpoint=${CKPT_PATH:?체크포인트 파일} +load_optimizer_state=${LOAD_OPT:-True} \
  +run_name=$RN +checkpoint_path=checkpoints/$RN +ema=True >> "$OUT" 2>&1 &
echo "$RN 재개 발사 pid $!  로그: $OUT"
