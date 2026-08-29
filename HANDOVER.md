# 인수인계 — 재구축 · 재개 · 분석

GPU 인스턴스가 매번 바뀐다. 이 문서대로 하면 클론 → 분석 5분, 학습 재개 15분.

## 1. 재구축

```bash
git clone -b clean <repo> LinearTuring && cd LinearTuring
pip install -q torch einops tqdm coolname pydantic argdantic omegaconf hydra-core huggingface_hub matplotlib wandb scipy
pip install -q flash-attn --no-build-isolation        # URM layers.py 가 import (우리 모델은 안 씀). 실패 시 sudoku/urm_patches.diff 의 SDPA 폴백이 대신 잡음
git clone --depth 1 https://github.com/UbiquantAI/URM.git refs/URM
NAUG=1000 bash sudoku/setup_urm.sh                    # 어댑터·패치·데이터(≈4분)·테스트 2048·전파깊이·체크포인트 배치. 분석만이면 NAUG=0 (수 초)
```
- URM 은 c14e55f 계열에서 검증됨. 업스트림이 바뀌어 패치가 안 붙으면 `git -C refs/URM checkout c14e55f`.
- 데이터는 HF `sapientinc/sudoku-extreme` 에서 받는다. 차단되면 CSV 를 `refs/URM/data/sudoku_csv/{train,test}.csv` 에 두면 패치가 로컬을 우선 읽음.
- 스크립트는 저장소 루트를 `LT_ROOT` 로 잡는다(기본: 파일 위치 기준). URM 위치·테스트셋 위치는 `URM_DIR`·`SUDOKU_TEST` 로 덮어쓸 수 있다.
- **`pkill -f pretrain.py` 는 자기 셸도 죽인다.** `pids=$(pgrep -f "[p]retrain"); kill $pids` 를 그 문자열이 없는 명령에서 단독 실행.

## 2. 재개

```bash
CKPT_PATH=checkpoints/R1B8_bilin_r2/step_123039.pt EPOCHS=34250 LOAD_OPT=False bash sudoku/resume.sh R1B8_bilin_r2
```
- 남은 학습: 계획 390,600 step 중 267,561 = 137 iters → `EPOCHS = 137 × 250 = 34250`. 일반식: `EPOCHS = (200 − step/1953) × 250`.
- 저장소 체크포인트는 **EMA 가중치만**(54MB). 옵티마이저 없이 재개해도 무해함을 실측 (AdamATan2 β₂=0.95, ~20 step 에 적응 복원; 76,167 → 78,120 에서 완답 835 → 865).
- **재개 검증 (필수, 2.5분 뒤)**: 첫 체크포인트(재개+1,953 step)의 `raw_model_state_dict` 가 로드값과 달라야 한다:
  ```python
  import torch; k='_orig_mod.model.inner.core.lam'
  a=torch.load('checkpoints/R1B8_bilin_r2/step_123039.pt',map_location='cpu',weights_only=False)['model_state_dict'][k]
  b=torch.load('<첫 체크포인트>',map_location='cpu',weights_only=False)['raw_model_state_dict'][k]
  print('무학습' if torch.equal(a,b) else '정상')
  ```
  eval 이 이전과 넷째 자리까지 같으면 무학습이다.

## 3. 함정 (전부 겪은 것)

| 증상 | 원인 | 조치 |
|---|---|---|
| 재개 후 eval 이 매번 동일, 옵티마이저 상태 0 | 업스트림 `load_state_dict(assign=True)` 가 Parameter 객체를 새로 만들어 옵티마이저가 옛 객체를 참조 → grad None → 학습 0 | `urm_patches.diff` 에 `assign=False` 포함. 패치 적용 확인 |
| `load_checkpoint` 에 디렉터리 주면 못 찾음 | 업스트림 정규식 `r"step_(\\d+)"` 이중 이스케이프 | 체크포인트 **파일** 경로 |
| hydra `Could not override` | `load_checkpoint`·`load_optimizer_state`·`run_name` 등은 cfg 에 없는 키 | `+키=값` (스크립트가 처리) |
| eval 이 전혀 안 찍힘 | 기본 `evaluators: [arc@ARC]` 가 스도쿠 경로에서 예외 → eval_loader 까지 None | `evaluators=[]` (스크립트에 포함) |
| 변형 간 곡선이 넷째 자리까지 동일 | 변형 플래그가 죽어 있음 (예: yaml 기본값이 실험 플래그를 덮음) | 발사 직후 `refs/URM/checkpoints/<run>/config.yaml` 의 arch 와 첫 체크포인트 파라미터 키 확인 |
| 로그의 `[EVAL]` 이 첫 eval 만 두 줄 | loops 16 / 24 두 번 평가 (`loop_deltas` 기본) | 무시. 둘 다 EMA |
| 재개 로그의 train `acc 0.0000` | 재개 경로의 콘솔 로깅 결함 (loss·eval 은 정상) | 미수정. train acc 필요하면 pretrain.py 패치의 `[LT]` 프린트 확인 |
| 체크포인트가 eval 마다 169MB | raw + EMA shadow + 옵티마이저 | 최신 1~2개만 남길 것. 완주 시 ~34GB |
| 학습 13.9 → 5.8 it/s | 같은 GPU 에 분석 스크립트 | 분석은 학습 GPU 와 분리 |
| `epochs=50000` 으로 재개 | 재개 시점부터 390,600 을 **더** 돈다 | `EPOCHS` 는 남은 만큼만 |

## 4. 분석

`analysis/*.py` 는 전부 학습 불필요·기본 인자로 `checkpoints/R1B8_bilin_r2_step123039.pt` 를 읽는다. 목록과 재생산 명령은 `RESULTS.md` 끝.
minimal 계열 체크포인트(`checkpoints/R1B8_min_stdp1_step228501.pt`, 설정 `checkpoints/R1B8_min_stdp1_config.yaml`)는 `sudoku/lt.py` 가 아니라 `core/minimal.py` 의 `LT` 로 로드한다 — `analysis/stdp_eval.py: load(path, stdp)` 참고(`stdp=True, stdp_target=product, stdp_lam_fixed=1.0`).
공용 로더 `sudoku/lt.py: load_lt(ckpt, **overrides)` — `_orig_mod.model.` 접두 정리·strict 로드. SwiGLU 판은 `bilinear=0`.
수동 전개는 `analysis/common.py: rollout(m, batch, act=, hook=)` — 게이트 치환과 단계별 훅 지원.

`sudoku/lt.py` 는 원판(`main` 의 `urm_port/lt.py`, 실험 플래그 15개)에서 R1B8 경로만 남긴 것이다. 두 체크포인트 모두에서 16 세그먼트 로짓·carry 가 원판과 **비트 동일**함을 확인했다(2026-08-28).

## 4.5 7일차 추가 (2026-08-29)
- 모델: `core/minimal.py` (URM 에는 `refs/URM/models/lt/minimal.py` 로 복사, arch yaml `sudoku/minimal.yaml` → `refs/URM/config/arch/minimal.yaml`). 플래그: `psi_zero`(2단계), `addr_dim`(3단계, 1248 에서 416), `stdp` + `stdp_target ∈ {addr, value, product}`(4단계; product 가 정합).
- 발사: `ARCH=minimal D=1248 bash sudoku/launch.sh <RUN> arch.psi_zero=true arch.addr_dim=416 arch.stdp=true arch.stdp_target=product`. `resume.sh` 도 `ARCH` 지원. 웜스타트용 병합 체크포인트 예: `refs/URM/checkpoints/warm/step_15624.pt` (3단계 가중치 + η·λ·β 초기값; 재구축 시 사라짐).
- 함정 추가: (i) `pkill -f`/`pgrep -f` 에 `pretrain.py` 문자열이 든 명령은 자기 셸을 죽인다 → `pgrep -f "[p]retrain"` 또는 `run_name=<RUN>` 의 마지막 글자를 `[x]` 로. (ii) 가소성 파라미터(`eta_raw·lam_raw·beta`)는 wd 제외 목록에 있어야 한다(패치 적용됨; 빠지면 sigmoid(0)=0.5 로 끌려감). (iii) `puzzle_emb` 는 batch_size 고정 — 부분 배치 추론은 batch_size=1 모델을 따로 만들 것. (iv) `wandb`·`scipy` 설치 필요.
- 분석 스크립트(7일차): `quadrant_view.py`(칸 범주·궤적) · `early_commit.py`(조기 여유 핀 루프) · `selfgrad.py`(정착 gradient 흔들기) · `stdp_infer.py`(결합 기억 추론) · `stdp_eval.py`(4단계 유용성) · `graph_transplant.py`(학습 그래프 이식) · `dist_sanity.py`/`phase_vs_dist.py`(위상 vs 분포) · `selfcorrect.py`(자기교정 루프). 결과 JSON `results/json/`, 곡선 `results/curves/R1B8_min*_{train,eval}.csv`, `R1B8_min_psi0_split_stdpp_*.csv`.
- 기록 문서: `STDP.md`(7일차 전체) · `STAGES.md`(단계별 필요성) · `PLAN.md`(장기 계획) · `ARC.md`(ARC 설계).

## 5. 다음 목표 — ARC Prize 2026 (2026-08-27 조사)

- 마감 2026-11-02 (재확인 요). 4×L4 · 12h · 오프라인. semi-private 120 + private 120, pass@2 완전일치. 논문 트랙 $450K.
- ARC-2 eval 실측 (`refs/URM/kaggle/combined/`): 120 과제 · test 입력 172 · 격자 중앙값 16×16 · 입출력 같은 크기 69% · 배경 0 인 과제 38/120 · test 에 train 에 없던 색 71/172.
  출력 크기는 규칙 3개(같음/상수/정수비)로 172 중 121 적중. "같은 색 연결성분 = 객체" 가 같은 크기 과제의 54%(eval)·69%(training) 에서 성립.
- 규모: URM ARC-1 프로토콜은 288 H100-시간 (Kaggle 예산의 ~30배). 과제당 학습(NCA 식, 과제당 수 분)은 예산 안.
- 코어 속도 (d256·H8·8스텝·배치 8·fwd+bwd): T=121 53ms/0.16GB · T=256 47ms/0.45GB · T=900 141ms/4.05GB.
- 착수 후보: 과제당 학습으로 ARC-1 training 30과제(같은 크기·≤15×15·새 색 없음) → 실패 양상을 보고 객체 편향·출력 격자 분리·인코더 중 무엇이 필요한지 결정.
- 참고 논문: TRM (2510.04871) · URM (2512.14693) · Easy-to-Hard (2106.04537) · Recursive Algorithmic Reasoning (2307.00337) · NCA for ARC (2506.15746). PDF 는 `main` 브랜치 `papers/`.
