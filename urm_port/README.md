# URM 하네스 포팅 — 재현 절차

1. `git clone https://github.com/UbiquantAI/URM refs/URM` (arXiv 2512.14693)
2. `cp urm_port/adam_atan2.py refs/URM/` (CUDA 백엔드 없는 순수 torch 심 — 정준형 a=b=1)
3. `mkdir -p refs/URM/models/lt && cp urm_port/lt.py refs/URM/models/lt/`  # 우리 코어 어댑터
4. `cp urm_port/lt.yaml refs/URM/config/arch/`
5. `git -C refs/URM apply ../../urm_port/urm_patches.diff`
   (pretrain.py: 콘솔 로그·eval 출력·wd 그룹 분리·config 저장 버그 우회 / layers.py: SDPA 폴백)
6. 데이터: `PYTHONPATH=. python data/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000`
   → `python urm_port/truncate_test.py <data>/test 2048` (학습 중 eval 용 절단)
7. 발사 명령·하이퍼는 REPORT §11.7 / PLAN.md 참조. **[2026-08-26] 최신 발사 스크립트는 `sudoku_runs/2026-08-26/code/launch26d.sh` (`arch.boundary_mlp=True` 명시, `+arch.bilinear=True` 로 활성화 0개 판). 발사 직후 `checkpoints/<run>/config.yaml` 확인 필수.** 분석: eval_lt.py(전선 분해)·probe_lt.py(세그 곡선·노름비)·cell_depth.py(칸별 전파 깊이)
