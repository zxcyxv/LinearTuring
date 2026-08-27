#!/bin/bash
# 6일차 재개 — R1B8 쌍선형(활성화 0개) 완주 런.
# EPOCHS 로 "남은 만큼"만 지정한다: 1 iter = 250 에폭 = 1953 step. 원 계획 총량 = 200 iters = 390,600 step.
#   예) step 76,167(=39 iters) 에서 재개 → 남은 161 iters → EPOCHS=40250 → 390,600 에서 종료.
#   예) step 123,039(=63 iters, 6일차 중단점) 에서 재개 → 남은 137 iters → EPOCHS=34250.
# lr_min_ratio=1.0 이라 cosine 항이 죽어 lr 은 상수 — EPOCHS 변경이 스케줄에 영향 없음 (검증됨).
# [사고 2026-08-27] 업스트림 load_checkpoint 의 assign=True 로 재개 시 옵티마이저가 옛 파라미터를 참조 → 무학습. 하네스 패치(assign=False) 후 재발사.
#   판정법: 첫 체크포인트의 raw_model_state_dict 가 로드한 가중치와 달라야 한다 (eval 수치 동일 = 무학습).
# 주의: 하네스의 _resolve_checkpoint_path 정규식이 이중이스케이프 버그(r"step_(\\d+)")라 디렉터리 지정이 안 된다 — 체크포인트 "파일"을 직접 준다.
RN=${RN:-R1B8_bilin_r2}; D=${D:-832}
OUT=/workspace/LinearTuring/sudoku_runs/2026-08-27/${RN}.log
cd /workspace/LinearTuring/refs/URM
WANDB_MODE=offline OMP_NUM_THREADS=4 nohup torchrun --nproc-per-node 1 --master_port=$((29500+RANDOM%400)) pretrain.py \
  data_path=${DATA:-data/sudoku-extreme-1k-aug-1000} \
  arch=lt arch.loops=16 arch.hidden_size=$D arch.num_heads=8 arch.R=1 arch.seg_steps=0 \
  +arch.blocks_per_seg=8 +arch.block_inj=True arch.boundary_mlp=True arch.ckpt=${CKPT:-False} +arch.amp=${AMP:-True} \
  +arch.bilinear=True \
  "$@" \
  epochs=${EPOCHS:-50000} eval_interval=250 evaluators=[] \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
  +load_checkpoint=${CKPT_PATH:?체크포인트 파일 경로 필요} +load_optimizer_state=${LOAD_OPT:-True} \
  +run_name=$RN +checkpoint_path=checkpoints/$RN +ema=True > $OUT 2>&1 &
echo "$RN (d=$D) 재개 발사 pid $!  로그: $OUT"
