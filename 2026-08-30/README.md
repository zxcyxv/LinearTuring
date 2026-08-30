# 2026-08-30 (8일차) — 추론 시점 자기교정: 시도와 반증, 그리고 퍼즐 57 인과 추적

모든 실험은 `checkpoints/R1B8_min_stdp1_step228501.pt` (7.16M, minimal+STDP product, λ=1) 로 수행.
분석 스크립트는 전부 `2026-08-30/analysis/` (공용 로더 `analysis/common.py` 는 저장소 루트의 것을 씀). 결론은 `STDP.md §6.13`(오전·오후) 과 `§6.14`(밤).

## 실패한 개입 (전부 이유 확인됨)
| 파일 | 스크립트 | 내용 |
|---|---|---|
| `results/json/ctds/edge_growth_*.json` | `analysis/ctds.py` | 위반 간선 결합의 지수 성장 (ρ·상한·감쇠 스윕). 대칭 반발이 맞는 쪽을 밀어냄 |
| `results/json/ctds/uncommit*.json` | `ctds.py --uncommit` | 칸 굳힘 해제 ± 이웃 확산. 붕괴/동결 |
| `results/logs/integ3.log` | `analysis/integrator.py` | 전달장 적분 / 블록·세그먼트 변위 모멘텀. 전부 기준 이하 |
| — | `analysis/residual_descent.py` | 잔차 ½‖F(h)−h‖² 하강·뉴턴. 0 아닌 오답 최소로 감 / 발산 |
| `R1B8_gate.log` | `core/minimal.py: gate` | 굳힘 게이트 학습. 전제가 반증되어 중단 (아래) |
| `R1B8_causal.log` | `core/minimal.py: stdp_target=causal` | 인과 STDP. 구현 결함 3개로 시험 자체가 무효, 중단 |

## 이긴 것
| 파일 | 스크립트 | 512 퍼즐 완답 |
|---|---|---|
| `results/logs/retract.log` | `analysis/guess_retract.py` | 블록1 추측 되돌리기 깊이 1 → **297** (기준 218) |
| `results/logs/retract2.log` | `analysis/guess_retract2.py` | 깊이 2 탐욕 + clean 정지 채택 → **327** |
| `results/logs/sched.log` | `analysis/consistency_memory.py` | 제시 간 일관성 결합 기억 (η·λ 스케줄 비교) → 64제시 **372** |
| — | `analysis/selfpin_oracle.py` | 오라클 상한(맞은 칸만 되받기) → **511** |

## 진단 측정
| 파일 | 스크립트 | 내용 |
|---|---|---|
| `results/json/extrap_stdp1_228501.json` | `analysis/extrap.py` | 반복 외삽 16→128, 원판/누적/세그먼트초기화 3변형 |
| `results/json/memory_horizon.json` | `analysis/memory_horizon.py` | η 스윕(기억 시평). 기억 제거 시 240→20 붕괴, 균일 치환은 전부 악화 |
| `results/json/orbit_period.json` | `analysis/orbit_period.py` | Brent 정확 주기 검출 1024 세그먼트 → **주기 0/128**, "영원히 못 푼다" 증명 불가 |
| `results/json/why_112.json` | `analysis/why_112.py` | 퍼즐 57 칸 21 여유의 블록별 항 분해 |
| `results/logs/ctds_trace_puzzle57_rho0.05.txt` | `analysis/ctds_trace.py` | 간선 성장 아래 퍼즐 57 추적 |
| `results/puzzle57_escape.html` | — | 퍼즐 57 거시 동역학 페이지(겹침·에너지/컬·DMD·결합) |

## 그 외 스크립트 (출력은 콘솔)
`analysis/edge_signal.py`(위반 탐지기 정밀도), `analysis/failure_structure.py`(오답 구조·결정 시각), `root_{rules,causal,blame,restart}.py`(뿌리 찾기, 전부 실패), `analysis/phase_order.py`(위상 반대칭이 결정 순서를 담음), `analysis/instability.py`, `analysis/msg_decompose.py`(메시지 분해), `analysis/one_puzzle_long.py`, `analysis/dyn_macro.py`, `analysis/decline_anatomy.py`, `analysis/flip_anatomy.py`, `analysis/jacobian_spectrum.py`, `analysis/pin_oracle.py`, `analysis/edge_diag.py`, `analysis/ctds_trace.py`.

`.npz`/`.npy` 중간 산출물은 git 무시. `dyn_macro.py 57 120` 으로 재생성.
