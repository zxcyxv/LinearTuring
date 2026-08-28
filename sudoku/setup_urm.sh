#!/bin/bash
# URM 하네스에 우리 어댑터·패치 설치 + 스도쿠 데이터 빌드.  전제: refs/URM 클론됨 (HANDOVER.md §1)
set -e
ROOT=${LT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT/refs/URM"
cp "$ROOT/sudoku/adam_atan2.py" .
mkdir -p models/lt && cp "$ROOT/sudoku/lt.py" models/lt/ && cp "$ROOT/sudoku/lt.yaml" config/arch/
git apply --check "$ROOT/sudoku/urm_patches.diff" 2>/dev/null && git apply "$ROOT/sudoku/urm_patches.diff" && echo "패치 적용" || echo "패치 이미 적용됨 (또는 충돌 — HANDOVER §1 확인)"
NAUG=${NAUG:-1000}   # 분석만 할 거면 NAUG=0 (테스트셋만 필요, 수 초)
PYTHONPATH=. LT_SEED=${LT_SEED:-0} python data/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug $NAUG
python "$ROOT/sudoku/truncate_test.py" data/sudoku-extreme-1k-aug-1000/test 2048
python "$ROOT/sudoku/cell_depth.py" data/sudoku-extreme-1k-aug-1000/test
mkdir -p checkpoints/R1B8_bilin_r2 && cp "$ROOT/checkpoints/R1B8_bilin_r2_step123039.pt" checkpoints/R1B8_bilin_r2/step_123039.pt
echo "완료. LT_ROOT=$ROOT"
