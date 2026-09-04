# 2026-09-04

플래그 2개 추가, 게이지 상수화, 물리 레이어 스택 도입, 연산 병목 실측,
캐글 단독 학습 코드와 4×L4 처리량 수정.

| 문서 | 내용 |
|---|---|
| [`DESIGN.md`](DESIGN.md) | **구조 변경 3건** — 블록 순서 `pre`→`post`, 스케일 게이지 상수화, 물리 레이어 스택. 관례 정합성과 수치 |
| [`CHANGES.md`](CHANGES.md) | 오늘의 코드 변경 전체 — `core/minimal.py`·`sudoku/minimal.yaml`(원위치, 기록만), `kaggle/`(신규), 삭제 항목 |
| [`RUNS.md`](RUNS.md) | 오늘 돌린 런의 설정과 곡선 — STDP 창·블록 순서·게이지·깊이·경계 폭 절제 |
| [`KAGGLE_MULTIGPU.md`](KAGGLE_MULTIGPU.md) | 캐글 4×L4 가 1×L4 보다 느렸던 원인과 수정 (step 634 → 104 ms) |
| [`KERNEL.md`](KERNEL.md) | 커널 수준 최적화 — MFU 18.9%, MAC 지분과 시간 지분의 9.1배 괴리, 표적 순위 |

## 분석 스크립트

| 파일 | 내용 |
|---|---|
| `analysis/bench_train.py` | 학습 1스텝(fwd+bwd+opt) 실측. `--profile`(aten 연산만), `--compile`, `--layers`, `--patch-emb` |
| `analysis/bench_urm.py` | URM 표준 스도쿠 프로토콜 동일 조건 벤치 |
| `analysis/cost_model.py` | 블록 1회의 MAC·활성 바이트 해석 모델 (GPU 불필요) |
| `analysis/compile_inspect.py` | dynamo 그래프 브레이크 · inductor 커널 구성 |
| `analysis/kernel_cpu_profile.py` | CPU/CUDA 동시 프로파일 + 할당자 통계. Triton 커널 포함 순위 |
| `analysis/inductor_fix_bench.py` | Inductor 설정별(`nopersist`/`coord`/기본) · 배치별 · `--exp` 별 step 시간 |
| `analysis/smoke_block_order.py` | `block_order` 스모크 4종 (회귀·연산순서·판독지점·값검증) |
| `analysis/flag_audit.py` | 플래그 상호작용 감사 — 조용히 무효가 되는 조합 찾기 |
| `analysis/readout_norm.py` | 판독 지점 ‖h‖ 대 흡수 반지름 |
| `analysis/scale_anatomy.py` | 스케일 해부 (γ·주입·임베딩) |
| `analysis/amplitude_channel.py` | 진폭 발언권 채널 |
| `analysis/train_sim.py` | 학습 속도 시뮬레이션 |

## 속도 실측 요약 (로컬 L4, 배치 128)

| | ms/step | it/s | peak GB | MFU |
|---:|---:|---:|---:|---:|
| 우리 (compile, psi, d832 N=1) | 174.4 | 5.73 | 3.85 | 18.9% |
| URM 표준 (compile) | 353.2 | 2.83 | 6.36 | 16.3% |

같은 L4·배치 128·같은 데이터에서 2.03배. 세그먼트당 층 적용은 48 대 8,
backward 를 도는 것은 24 대 8 이고, 적용 1회당 비용은 우리가 2.30배다.

## 플래그 감사

- `stdp_window` 는 `stdp_target ∈ {product, value}` 나 `stdp=False` 에서 무효
  (그 target 들은 `kcb` 를 안 쓴다)
- `stdp_lam_init` 은 `stdp_lam_fixed ≥ 0` 에서 무효. `R1B8_min_stdp1` 이 그
  설정이라 그 런의 `stdp_lam_init: 0.25` 는 기록에만 있고 효과가 없었다
- `block_order` 는 어떤 조합에서도 죽지 않는다
- `faithful+psi` 는 `product` 와 비트 동일. `pre` 는 플래그 도입 전 코드와 비트 동일

## 캐글

`kaggle/train_kaggle.py` — URM 참조 0개 단독 학습 코드 (1,739줄).
`prep_dataset.py` 가 만드는 업로드 데이터는 179.8 KB (증강은 학습 중 생성).
`예외사항.md` 에 12시간 방어 장치와 사고 기록.
