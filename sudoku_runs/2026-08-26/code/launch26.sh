#!/bin/bash
# 5일차 발사 스크립트: launch.sh 와 동일 하이퍼 + ckpt=False + bf16 autocast (amp). 추가 arch 플래그는 "$@" 로.
RN=$1; D=$2; shift 2
OUT=/workspace/LinearTuring/sudoku_runs/2026-08-26/${RN}.log
cd /workspace/LinearTuring/refs/URM
WANDB_MODE=offline OMP_NUM_THREADS=4 nohup torchrun --nproc-per-node 1 --master_port=$((29500+RANDOM%400)) pretrain.py \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  arch=lt arch.loops=16 arch.hidden_size=$D arch.num_heads=8 arch.R=1 arch.seg_steps=0 \
  +arch.blocks_per_seg=8 +arch.block_inj=True arch.boundary_mlp=True arch.ckpt=False +arch.amp=True \
  "$@" \
  epochs=50000 eval_interval=250 evaluators=[] \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
  +run_name=$RN +checkpoint_path=checkpoints/$RN +ema=True > $OUT 2>&1 &
echo "$RN (d=$D) 발사 pid $!"
