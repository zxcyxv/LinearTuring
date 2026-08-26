# 2026-08-26 변경 사항 (저장소 파일 기준)

## 수정
- `urm_port/lt.py` (= `refs/URM/models/lt/lt.py`, 스냅샷 `code/lt.py`)
  - `LTConfig.bilinear: bool=False` — 경계 게이트 `silu(g)` → `g/2` (쌍선형, 활성화 0개)
  - `LTConfig.gate_quad: bool=False` — 게이트 `g/2 + g²/4` (silu 테일러 2차). 오늘 실험에서 미사용 판정(§7.5: 진폭 구간에서 쌍선형과 동일)
  - `LTConfig.amp: bool=False` — 세그먼트 forward 를 bf16 autocast 로 감싸고 로짓·carry 는 fp32 반환. `forward` → `_forward` 분리
  - `_act(g)` 헬퍼 도입, `_boundary()` 의 silu 호출 2곳을 `_act` 로 치환. `if boundary_mlp` → `elif` (lin_boundaries 분기 뒤). R1B8 경로의 수학은 3일차와 동일
- `urm_port/lt.yaml` (= `refs/URM/config/arch/lt.yaml`, 스냅샷 `code/lt.yaml`): `boundary_mlp: False` → **`True`** (§7.1 사고의 수정). 선형 경계 절제는 `arch.boundary_mlp=False` 명시 필요
- `refs/URM/data/build_sudoku_dataset.py` (gitignore 밖 스냅샷 `code/build_sudoku_dataset.py`): 서브샘플 시드를 `LT_SEED` 환경변수로 (기본 0). 시드 1·2 데이터셋 `data/sudoku-1k-seed{1,2}` 빌드
- `INDEX.md`: 상태줄 갱신, 자기정정 14~17 추가(15·16 은 본 폴더 §7.3·7.4 로 범위 축소 주석)

## 추가 (이 폴더)
- `REPORT.md` — 진단·사고·후기 재진단·튜링/ARC 논의. `CHANGES.md` — 이 파일
- `code/` 진단 스크립트: `diag.py`(클래스·위반 분해) `dyn.py` `dyn2.py` `dyn3.py`(놕아웃·선형 대체·절단) `graph_dep.py`(그래프 입력의존, CA 대조) `sharpen.py`(엔트로피 분업) `gate_swap.py`(게이트 치환) `gate_stats.py`(게이트 분포) `depth_use.py`(깊이 사용도) `prof*.py`(속도) — 결과 `*.json`, `telescope_B16.txt`
- `code/launch26.sh` ~ `launch26d.sh`: 발사 스크립트 (전부 `arch.boundary_mlp=True` 로 수정됨). d 판이 최종: `AMP`/`CKPT`/`DATA` 환경변수, 추가 arch 플래그는 인자
- 로그: `R1B8_bilin_ok.log`(완주 진행), `R1B8_swiglu_ok.log`(4.4k 중단). **무효 런**(경계 플래그 누락): `R1B8_bilin` `R1B8_quad` `R1B8_bilin_fp32` `R1B8_swiglu_ctl` `R1B8_swiglu_day3rep` `R1B8_swiglu_mseed1` `R1B8_swiglu_seed1` — 비잔차 contract 선형 경계의 벽 복제로만 유효

## 환경 (커밋 대상 아님)
- `refs/URM` 재클론 c14e55f + 패치 + 어댑터. 데이터 `data/sudoku-extreme-1k-aug-1000`(시드 0, README 명령), 테스트 2048 절단, `cell_depth.npy`
- 속도: compile+bf16+ckpt=False 로 4.3 → 13.6 it/s (단독). Cayley 는 이득 없음(compile 하 동일) → 미채택
- 체크포인트는 `refs/URM/checkpoints/<run>/` (커밋 제외)

## 추가 수정 (저녁) — 체크포인트 재개 완전성
- `refs/URM/pretrain.py` (→ `urm_port/urm_patches.diff` 재생성, 스냅샷 `code/urm_patches.diff`): EMA 사용 시 저장되던 `model_state_dict` 는 **EMA 가중치**였고 원시(raw) 학습 가중치·EMA shadow 는 저장되지 않았다.
  → `raw_model_state_dict`, `ema_shadow` 키 추가 저장, 로드 시 raw 로 재개 + shadow 복원. 기존 키는 그대로(평가·호환). **현재 실행 중인 완주 런에는 미적용**(프로세스가 옛 코드) — 그 체크포인트로 재개하면 EMA 가중치에서 이어짐(옵티마이저 모멘트는 정상 복원, step·lr·rng 복원).
- 데이터 로더 위치(에폭 내 순서)는 하네스가 원래 저장하지 않음 — 재개 시 새 셔플. 완주 재현엔 무해, 비트 단위 재개는 불가.

## 저녁 — 완주 런 중단·체크포인트
- `R1B8_bilin_ok` 76,167 step 에서 중단 (eval 0.800·835/2048). `checkpoints/R1B8_bilin_ok_step76167.pt` = model_state_dict(EMA)+step 만 (GitHub 100MB 한도). 옵티마이저 상태 포함 151MB 전체본은 `refs/URM/checkpoints/R1B8_bilin_ok/`(gitignore). `checkpoints/R1B8_bilin_ok_config.yaml` = 하네스 저장 설정.
- 무효 런 체크포인트 전부 삭제, 현재 런은 최신 2개만 보존.
