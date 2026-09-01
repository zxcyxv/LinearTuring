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
