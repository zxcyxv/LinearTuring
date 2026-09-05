# 참고 논문 (2026-08-30 내려받음)

| 파일 | 논문 | 이 프로젝트와의 관계 |
|---|---|---|
| `arXiv-1804.02464.pdf` | Miconi, Stanley, Clune, **Differentiable plasticity** (ICML 2018) | 연결 = 고정 가중치 `w_ij` + 가소성 성분 `α_ij·Hebb_ij`. `α` 가 **연결별 학습 파라미터**인 것이 균일 가소성 대비 우위의 원인이라고 주장 |
| `arXiv-2002.10585.pdf` | Miconi et al., **Backpropamine** (ICLR 2019) | 위에 **신경조절** M(t) 추가. M 은 네트워크 자신의 출력 스칼라(tanh, 부호 자유). 적격 흔적판: `E ← (1−η)E + η·pre·post`, `Hebb ← Clip(Hebb + M·E)`. **세 번째 인자를 손으로 고르지 않고 학습시킨다**는 것이 핵심 |
| `arXiv-1801.05219.pdf` | Gerstner, Lehmann, Liakoni, Corneil, Brea, **Eligibility traces and plasticity on behavioral time scales** | 3-인자 규칙의 신경과학 근거. 적격 흔적이 기록 시각과 판단 시각을 분리해 시평 문제를 푸는 구조 |
| `arXiv-1901.09049.pdf` | Bellec et al., **e-prop** (Nature Comms 2020) | 순환망에서 BPTT 없이 적격 흔적 × 학습 신호로 경사를 근사. 우리 하네스가 세그먼트마다 detach 하는 문제와 같은 자리 |
| `arXiv-1208.0370.pdf` | Ercsey-Ravasz & Toroczkai, **The Chaos Within Sudoku** (Sci. Rep. 2012) | 결정론적 연속 동역학의 어트랙터가 해뿐이 되게 만드는 느린 변수(절 가중치 지수 성장). 우리 §6.12c 의 이론적 출처이며, 왜 유계 EMA 로는 그 정리가 성립하지 않는지도 여기서 나옴 |

읽은 결과와 우리 구조와의 대조는 `STDP.md §6.16`.

## 추가 (2026-09-01) — 반복 추론 모델 계열

| 파일 | 논문 | 이 프로젝트와의 관계 |
|---|---|---|
| `arXiv-2506.21734.pdf` | Wang et al., **Hierarchical Reasoning Model (HRM)** | 우리 하네스의 원류. Sudoku-Extreme 1k 55.0%. Fig 5(c) 의 추론시간 스케일링은 **Sudoku-Extreme-Full(3.83M)** 에서 잰 것이고 1k 에서는 보고하지 않았다 — 우리가 그들 공개 체크포인트로 직접 쟀다 (Δ=+8.59%p) |
| `arXiv-2510.04871.pdf` | Jolicoeur-Martineau, **Less is More: TRM** | 2층 7M 재귀 모델. 자체 보고 Sudoku 87.4% (URM 은 통일 아키텍처로 재현 시 66.8% 라고 반박) |
| `arXiv-2512.14693.pdf` | Gao et al., **Universal Reasoning Model (URM)** | 우리가 쓰는 하네스. Sudoku pass@1 77.6%. **외삽(test-time scaling) 실험 없음** — 비교에서 명시적으로 제외 |
| `arXiv-2512.11847.pdf` | **TRM on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute** | 독립 분석. TRM 의 재귀 외삽 Δ=**0.00** (Table 3), 투표 파이프라인이 Pass@1 의 11%p 기여, 평가 성능이 ID 토큰에 강하게 묶임 |
| `arXiv-2601.10679.pdf` | **Are Your Reasoning Models Reasoning or Guessing?** — HRM 기계론적 분석 | 미독 |

## 추가 (2026-09-05) — EqProp · 위상 진동자 STDP

| 파일 | 논문 | 이 프로젝트와의 관계 |
|---|---|---|
| `arxiv_1808.04873.pdf` | Scellier, Goyal, Binas, Mesnard, Bengio, **Generalization of Equilibrium Propagation to Vector Field Dynamics** (2018) | 비대칭(경사가 아닌) 동역학에서의 EqProp. 정리 1: 대리 경사 ν = (∂C/∂s)(∂μ/∂s)^{-T}(∂μ/∂θ) 와 진짜 경사의 정렬은 야코비안 대칭성에 달림. 우리 182k 고정점에서 코사인 0.80, w 항 0.97 (`2026-09-05/analysis/theorem1_check.py`). STDP 형 규칙 pre×Δpost 의 출처 |
| `arxiv_2103.04992.pdf` | López-Pastor, Marquardt, **Self-learning machines based on Hamiltonian echo backpropagation** (2023) | 시간가역 해밀토니안 계 전용 → 우리 추론(소산적)에는 해당 없음 |
| `seliger_nlin0110044.pdf` | Seliger, Young, Tsimring, **Plasticity and learning in a network of coupled phase oscillators** (PRE 65, 041906, 2002) | 적응 결합 쿠라모토의 원조. K̇ = ε(α cos Δφ − K). ε≪1 에서 쌍안정(기억), 군집 내 배위상 쿠라모토, 외부 구동으로 위상 패턴 저장. 합의 규칙 w ← w + δ(a − w) 의 문헌상 원형 |
| `berner_multicluster_1809.00573.pdf` | Berner, Schöll, Yanchuk, **Multi-clusters in networks of adaptively coupled phase oscillators** (2018) | Aoki–Aoyagi 2009 모델의 해석. β 가 규칙 종류(헤비안/인과 STDP/반헤비안)를 정하고, 위상잠금 해는 splay·antipodal·double antipodal 뿐. "같은 숫자 = 같은 위상, 단위 = splay" 대응의 출처 |
| `gushchin_plastic_1602.07012.pdf` | Gushchin, Mallada, Tang, **Phase-coupled oscillators with plastic coupling: synchronization and stability** (2016) | 정리 1: K 대칭·f 홀함수면 (φ, K) 전체가 퍼텐셜 V = −ΣK F(Δφ) − Σq K + ½ΣK²/α 의 경사계. 위상 동역학과 STDP 갱신이 같은 에너지의 경사 → STDP–EqProp 화해의 형식적 근거 |

읽은 결과와 우리 구조와의 대조는 `2026-09-05/문헌정리.md`.
