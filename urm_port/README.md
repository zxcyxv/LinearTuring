# URM 하네스 포팅 — 재현 절차 (2026-08-27 검증판; 상세·함정은 루트 `HANDOVER.md`)

1. `ln -sfn "$(pwd)" /workspace/LinearTuring` — lt.py·분석 스크립트가 이 경로를 하드코딩
2. `pip install -q einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub numba scipy`
3. `git clone --depth 1 https://github.com/UbiquantAI/URM.git refs/URM` (arXiv 2512.14693; ARC 데이터 동봉 `kaggle/combined/`)
4. `cp urm_port/adam_atan2.py refs/URM/` (CUDA 백엔드 없는 순수 torch 심 — 정준형 a=b=1)
5. `mkdir -p refs/URM/models/lt && cp urm_port/lt.py refs/URM/models/lt/ && cp urm_port/lt.yaml refs/URM/config/arch/`
6. `git -C refs/URM apply ../../urm_port/urm_patches.diff`
   (build_sudoku: 로컬 CSV·LT_SEED / layers: SDPA 폴백 / losses·urm: 소소 / pretrain: 콘솔 로그·eval 출력·wd 그룹 분리·config 저장 우회·raw+EMA shadow 저장·**`assign=False` 재개 버그 패치**)
7. 데이터: `cd refs/URM && PYTHONPATH=. LT_SEED=0 python data/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000`
   → `python /root/LinearTuring/urm_port/truncate_test.py data/sudoku-extreme-1k-aug-1000/test 2048` (학습 중 eval 용 절단). HF 차단 시 ModelScope 사본을 `data/sudoku_csv/{train,test}.csv` 에 두면 패치가 로컬을 우선 읽음.
8. 발사: 새 런은 `sudoku_runs/2026-08-26/code/launch26d.sh` (`arch.boundary_mlp=True` 명시, `+arch.bilinear=True` 로 활성화 0개 판). **재개는 `sudoku_runs/2026-08-27/code/resume27.sh`** (`CKPT_PATH` 는 체크포인트 **파일**, `EPOCHS` 는 남은 iters×250). 발사 직후 `checkpoints/<run>/config.yaml` 과 첫 체크포인트의 파라미터 키·raw 가중치 변화를 확인.
9. 분석: eval_lt.py(전선 분해)·probe_lt.py(세그 곡선·노름비)·cell_depth.py(칸별 전파 깊이) — d832·경로 하드코딩이라 인자화 필요.
