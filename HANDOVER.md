# 인수인계서 — 새 인스턴스에서 바로 이어가기 (2026-08-27 작성)

> 매일 GPU 인스턴스가 바뀐다. 이 문서 하나로 재구축 → 스도쿠 재개 → 다음 실험 착수가 되어야 한다.
> 오늘(6일차) 실제로 걸린 함정을 전부 적었다. 절차는 위에서 아래로 그대로.

## 0. 현재 상태 (2026-08-27 정오 중단 시점)

- 스도쿠 완주 런 `R1B8_bilin_r2` (활성화 0개 쌍선형 경계, d832·H8·R1·블록8·블록주입·bf16): 어제 76,167 → 오늘 **123,039 에서 중단**.
- eval(held-out 2048, loops16): **최고 exact 959/2048 @121,086** (셀 0.8156), **최고 셀 0.8176 @123,039** (exact 944). 어제 최고 859 → +100.
- 궤적(exact): 78k 865 → 84k 812(저점) → 90k 912 → 98k 920 → 105k 940 → 111k 952 → 121k 959 → 123k 944. 완만한 상승에 ±30 변동.
- 체크포인트: `sudoku_runs/2026-08-27/checkpoints/R1B8_bilin_r2_step123039.pt` (**EMA 가중치만**, 54MB) + `R1B8_bilin_r2_config.yaml`.
  raw 가중치·EMA shadow·옵티마이저는 로컬 `refs/`에만 있었고 인스턴스와 함께 사라진다 — EMA-only 재개가 무해함은 오늘 실측(§3-5).
- 남은 학습: 원 계획 390,600 중 267,561 step = 137 iters → 재개 시 `EPOCHS=34250`.
- 논문 5편 `papers/` (tracked). URM 저장소는 `refs/URM`(gitignore) — 매번 재클론. ARC 데이터는 그 저장소에 동봉(`kaggle/combined/*.json`).

## 1. 재구축 절차 (오늘 검증된 순서, ~15분)

```bash
cd /root/LinearTuring                                   # 클론 위치
ln -sfn "$(pwd)" /workspace/LinearTuring                # lt.py·분석 스크립트가 /workspace/LinearTuring 을 하드코딩
pip install -q einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub numba scipy pypdf
mkdir -p refs && git clone --depth 1 https://github.com/UbiquantAI/URM.git refs/URM
cp urm_port/adam_atan2.py refs/URM/
mkdir -p refs/URM/models/lt && cp urm_port/lt.py refs/URM/models/lt/ && cp urm_port/lt.yaml refs/URM/config/arch/
git -C refs/URM apply ../../urm_port/urm_patches.diff   # 5파일: build_sudoku·layers·losses·urm·pretrain (assign=False 패치 포함)
cd refs/URM
PYTHONPATH=. LT_SEED=0 python data/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000   # ~4분
python /root/LinearTuring/urm_port/truncate_test.py data/sudoku-extreme-1k-aug-1000/test 2048
mkdir -p checkpoints/R1B8_bilin_r2 && cp /root/LinearTuring/sudoku_runs/2026-08-27/checkpoints/R1B8_bilin_r2_step123039.pt checkpoints/R1B8_bilin_r2/step_123039.pt
```

걸린 것:
- 데이터 빌드는 `PYTHONPATH=.` 없으면 `No module named 'data'`.
- torch 2.8+cu128 은 인스턴스에 있었음. scipy 없음(객체 분석용). HF 는 오늘 접속됨(3일차엔 차단 → ModelScope 사본, `urm_port/README.md`).
- **`pkill -f pretrain.py` 는 자기 셸도 죽인다** — 명령줄에 그 문자열이 있으면 매치됨(`[p]retrain` 패턴도 같은 명령에 `refs/URM/pretrain.py` 경로가 있으면 마찬가지). 안전: `pids=$(pgrep -f "[p]retrain"); kill $pids` 를 그 문자열이 **없는** 명령에서 단독 실행. 오늘 세 번 당함.

## 2. 재개 절차

```bash
cd /root/LinearTuring
RN=R1B8_bilin_r2 D=832 EPOCHS=34250 LOAD_OPT=False CKPT_PATH=checkpoints/R1B8_bilin_r2/step_123039.pt bash sudoku_runs/2026-08-27/code/resume27.sh
```
- EPOCHS 산술: 1 iter = 250 에폭 = 1,953 step, 총 200 iters = 390,600. `재개 step / 1953` = 완료 iters, `(200 − 완료) × 250` = EPOCHS. lr 은 `lr_min_ratio=1.0` 상수라 EPOCHS 가 스케줄에 영향 없음(검증).
- **재개 검증(필수, 2.5분 뒤)**: 첫 체크포인트(재개+1,953 step)의 `raw_model_state_dict` 가 로드한 가중치와 **달라야** 한다. eval 이 이전과 소수점 넷째 자리까지 같으면 무학습(§3-3).

## 3. 오늘 걸린 재개 함정 (전부 해결, 재발 방지용)

1. hydra: `load_checkpoint`·`load_optimizer_state` 는 cfg 에 없는 키 → `+load_checkpoint=...` 처럼 `+` 접두 필요.
2. 업스트림 `_resolve_checkpoint_path` 정규식이 `r"step_(\\d+)"` 이중 이스케이프 → **디렉터리 지정이 원리적으로 안 됨**. 체크포인트 **파일** 경로를 직접 준다.
3. **업스트림 `load_state_dict(..., assign=True)` (제일 큰 사고)**: Parameter 객체를 새로 만들어 교체 → 그 전에 생성된 옵티마이저는 옛 객체를 참조 → `.grad` 가 `None` → 전 파라미터 스킵 → **학습 0**. 증상: eval 이 매번 동일(0.7998·825 반복), 체크포인트 `raw_model_state_dict` == 로드값, `optimizer_states` 항목 0. 오늘 두 번의 재개(약 40분 GPU)가 이걸로 무효. 패치 `assign=False` 는 `urm_patches.diff` 에 포함. 진단 스니펫:
   ```python
   import torch; k='_orig_mod.model.inner.core.lam'
   a=torch.load('checkpoints/R1B8_bilin_r2/step_123039.pt',map_location='cpu',weights_only=False)['model_state_dict'][k]
   b=torch.load('<재개 후 첫 체크포인트>',map_location='cpu',weights_only=False)['raw_model_state_dict'][k]
   print('무학습' if torch.equal(a,b) else '정상', (a-b).abs().max().item())
   ```
4. `[EVAL]` 이 첫 eval 에만 두 줄인데 이는 raw/EMA 가 아니라 **loops=16 / loops=24** (`loop_deltas` 기본 `[0,8]`, 이후 `[0]`). 전부 EMA 가중치.
5. EMA-only 체크포인트 재개(옵티마이저 모멘트 0, shadow=로드값)는 무해: 76,167(835) → 78,120 에 865/901. AdamATan2 β₂=0.95 라 ~20 step 에 적응 복원, atan2 갱신은 스텝당 `lr·π/2` 로 유계.
6. 체크포인트가 eval(1,953 step)마다 **169MB** 씩 쌓인다(raw+shadow+옵티마이저). 완주면 ~34GB. 최신 1~2개만 남길 것.
7. `epochs=50000` 그대로 재개하면 재개 시점부터 390,600 을 **더** 돈다(진행바 총량이 그렇게 뜸). 남은 만큼만 EPOCHS 로.
8. 같은 GPU 에 벤치마크 등 부업을 올리면 학습이 13.9 → 5.8 it/s 로 떨어진다. T=900·배치32 역전파는 17GB.

## 4. 오늘의 결정·측정 요약

- **목표 전환**: 논문 → ARC Prize 2026. 마감 2026-11-02(검색 결과 기준, 재확인 요), 4×L4·12h·오프라인, semi-private 120 + private 120 = 240문제, pass@2 완전일치. 논문 트랙 $450K(Accuracy·Universality·Progress·Theory·Completeness·Novelty 동일가중, 코드 점수 불필요).
- **ARC-2 데이터 실측** (`refs/URM/kaggle/combined/`): eval 120과제·test 입력 172개, 격자 중앙값 16×16(256칸, ARC-1 의 2.5배), 입출력 같은 크기 69%, 배경색 0 인 과제 38/120 뿐, test 입력에 train 에 없던 색 71/172. 출력 크기는 규칙 3개(같음/상수/정수비)로 test 172 중 121 적중. "같은 색 연결성분 = 객체" 사전분포는 같은 크기 과제의 54%(eval2)·69%(training2)에서 정확히 성립(객체 단위 작용), 색 무시 연결은 19%.
- **규모**: URM ARC-1 프로토콜 = 960과제·441만 시퀀스(900토큰)·200k 에폭·예제당 깊이 768 → 288 H100-시간(TRM 보고). Kaggle 예산 ≈ 2 L4-일 → 약 30배 초과. 과제당 학습(NCA 논문: 과제당 수 분)은 예산 안.
- **코어 속도 실측** (d256·H8·8스텝·배치8·fwd+bwd, 스도쿠 런과 GPU 공유): T=121 52.8ms/0.16GB · T=256 47.0ms/0.45GB · T=900 141ms/4.05GB. 메모리 25배, 시간은 소배치라 2.7배(지연 한계) — 포화 배치 처리량은 GPU 빌 때 재측정.
- **병렬 스캔 질문 정리**: 감쇠 커널은 분리 가능한 지수라 위치 축 스캔 가능(선형 어텐션+감쇠 계열) — T≤256 에선 명시 행렬이 더 빠름. 순차 축은 깊이(A(h) 재계산)이고 이건 얼리면(`freeze_A`) 4배 빨라지지만 순차 과제에서 0.000(REPORT §8.7). Mamba-3 류는 A 가 입력 의존 클래스라 상태 의존을 우회하지 않음(REPORT §9 문헌표 (a)+(d) 행). R1B8 은 이미 블록당 A 1회 재계산 = 하한.
- **논문 정독**: TRM·URM(기숙지) + Easy-to-Hard·Recursive Algorithmic Reasoning·NCA-for-ARC → `PAPERS.md`. 진짜 귀한 것: E2H 의 "반복 연장 외삽" 현상+프로토콜, NCA 의 범위·비용 회계(가능 과제 172/400, 13.4%, 과제당 수 분), RAR 의 "스택 없어도 OOD 유지 — 감독 정렬이 관건" 절제. **주의**: `PAPERS.md` §A~D 의 [우리] 대응 서술은 프로젝트에 끼워맞춘 것이라는 지적을 받음 — 논문 자체 요약이 필요하면 재작성.
- **미결정 다음 착수 후보**: 과제당 학습(NCA 프로토콜, 하네스·캔버스·과제벡터 없이 코어만)으로 ARC-1 training 30과제(같은 크기·≤15×15·새 색 없음) 실험 → 실패 양상을 보고 객체 편향(sheaf 유형별 제한사상)·출력 격자 분리·z 인코더 중 무엇이 필요한지 결정. 이론 논의(sheaf 정준 분해·유형별 단면·유형별 차수=객체 통계)는 세션 기록에 있고 THEORY.md 명제 27 후보.

## 5. 파일

- `sudoku_runs/2026-08-27/`: `R1B8_bilin_r2.log`(유효 런 76,167→123,039), `R1B8_bilin_r2_a.log`·`R1B8_bilin_r2_b_noop.log`(무학습 런 2개, 증거), `code/resume27.sh`, `code/urm_patches.diff`, `checkpoints/`.
- `urm_port/urm_patches.diff` 갱신(assign 패치), `urm_port/README.md` 절차 갱신, `papers/` 5편, `PAPERS.md`.
