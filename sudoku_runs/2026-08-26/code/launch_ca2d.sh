#!/bin/bash
# 2D CA (URM 형식) 를 스도쿠와 같은 LT 하네스·R1B8 쌍선형 구성으로 학습. ARC 전환 = data_path 와 grid 만.
RN=$1; DATA=$2; D=${3:-256}; shift 3
OUT=/workspace/LinearTuring/sudoku_runs/2026-08-26/${RN}.log
cd /workspace/LinearTuring/refs/URM
WANDB_MODE=offline OMP_NUM_THREADS=4 nohup torchrun --nproc-per-node 1 --master_port=$((29500+RANDOM%400)) pretrain.py \
  data_path=$DATA \
  arch=lt arch.loops=16 arch.hidden_size=$D arch.num_heads=8 arch.R=1 arch.seg_steps=0 arch.grid=30 \
  +arch.blocks_per_seg=8 +arch.block_inj=True arch.boundary_mlp=True +arch.bilinear=True arch.ckpt=False +arch.amp=True \
  "$@" \
  evaluators=[] lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  +run_name=$RN +checkpoint_path=checkpoints/$RN +ema=True > $OUT 2>&1 &
echo "$RN 발사 pid $!"
