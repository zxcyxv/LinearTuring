# LinearTuring

활성화 함수 없이 **부호 있는 어텐션 동역학**만으로 순차 계산을 하는 재귀 아키텍처.
현재 결과: Sudoku-Extreme(1k 퍼즐 + 증강) held-out 2048 에서 **셀 0.818 · 완답 947/2048 (46%)**, 활성화 함수 0개, 14.1M 파라미터.

| 문서 | 내용 |
|---|---|
| `ARCHITECTURE.md` | 모델 정의 (코어 수식 + 스도쿠 판 구조) 와 각 부품이 그렇게 생긴 이유 |
| `THEORY.md` | 증명된 성질만 (유계성 · 동차성 · γ 게이지 · 조건부 선형성 · 진폭 정착) |
| `RESULTS.md` | 남긴 실험 4건 — 계단식 학습 곡선 · 게이트 미포화와 쌍선형 치환 · 스도쿠 해석·시각화 · 선형 경계 대조 |
| `HANDOVER.md` | 새 인스턴스에서 재구축 → 재개 → 분석까지의 절차와 함정 |
| `PLAN.md` | **장기 계획** — 한 문장 주장 · 현재 구조 · 장점 · 확인된 것/가설 · 입증 실험과 기준 · ARC Prize 전략(규정·두 트랙·논문 골격) · 9주 일정 · 위험 |
| `STAGES.md` | 7일차 단계(minimal → ψ=0 → 주소/값 분할 → STDP) 각각의 이론적 필요성과 실측된 대가 |
| `ARC.md` | ARC 계획 — v0 과제 임베딩을 수송·커널에(gradient) → v1 문맥 내 델타 규칙 제한 사상(순전파) · 지그소 스도쿠 중간 단계 |
| `STDP.md` | 7일차 — 오답의 실물(맞바꿈 순환) · 결합 기억(위상 STDP, 학습 없이 947→999) · sheaf 확산과 STDP 의 연결 |

이 브랜치(`clean`)는 6일간의 탐색 기록(`main`)에서 **모델·이론·재현 가능한 결과만** 추려 재구성한 것이다.
MNIST·1D CA·n-hop·반응-확산 해석·깊이 축 설계 메모 등은 `main` 의 git 이력에 있다.

## 구조

```
core/model1.py        코어: 복소 주소 투영 · 감쇠 Gabor 커널 · 부호 어텐션 · 4차 소산 (활성화 0개)
sudoku/lt.py          URM 학습 하네스 어댑터 — R1B8 구조 (경계 쌍선형 → 주입 → dt=1 스텝) × 8 블록
sudoku/*.sh, *.diff   발사·재개 스크립트, URM 패치, 데이터 빌드
analysis/             체크포인트 진단·게이트 치환·해석·시각화 (학습 불필요)
tasks/                코어 단독 검증: MNIST (train.py), 1D CA (ca_task.py)
results/              곡선 CSV · 진단 JSON · 그림 · 원 로그
checkpoints/          R1B8_bilin_r2 @123,039 (현재 최고, EMA) · R1B8_swiglu @9,765 (SwiGLU 대조) · R1B8_min_stdp1 @228,501 (minimal+STDP 충실형 7.16M, 완답 974; `core/minimal.py`)
```

## 빠른 시작

```bash
# 환경 (HANDOVER.md §1)
pip install torch einops tqdm coolname pydantic argdantic omegaconf hydra-core huggingface_hub flash-attn matplotlib
git clone --depth 1 https://github.com/UbiquantAI/URM.git refs/URM && bash sudoku/setup_urm.sh

# 체크포인트 진단 (held-out 2048)
python analysis/diag.py                       # 기본: checkpoints/R1B8_bilin_r2_step123039.pt
python analysis/gate_swap.py --ckpt checkpoints/R1B8_bilin_r2_step123039.pt --bilinear 1

# 학습 재개 / 새 런
CKPT_PATH=checkpoints/R1B8_bilin_r2/step_123039.pt EPOCHS=34250 LOAD_OPT=False bash sudoku/resume.sh R1B8_bilin_r2
bash sudoku/launch.sh R1B8_bilin_s1 seed=1
```

## 상태 (2026-08-29)

- 원판 `R1B8_bilin_r2` 123,039 중단(재개 가능). 7일차: minimal→ψ=0→분할→STDP 단계 학습(STDP.md §6), 손실은 ψ=0 하나(2×2), 자기교정 루프 947→1198(원판+결합 기억+위반 초기화). 장기 계획은 `PLAN.md`.
- `R1B8_min_stdp1`(minimal+STDP 충실형, 7.16M) 236,313 에서 정지. 완답 974@228,501(보존)·980@236,313(삭제), 원판 14.1M 의 959@123,039 와 같은 수준을 절반 파라미터·2배 스텝으로 (STDP.md §6.11).
- 8일차(08-30): 자기교정 구조 규명 — 모델 = 강한 전파기 + 블록 1 추측(되돌리지 못함); 검증은 정지(풀린 상태 = 정확한 고정점, 오답은 절대 정지 안 함); 라벨 없이 되는 것: 추측 되돌리기 297, 제시 일관성 결합 기억 372, 오라클 상한 511 (STDP.md §6.13).
- 8일차 밤: 굳힘 게이트·인과 STDP 반증(§6.14), 퍼즐 57 인과 추적 — 정답 신호는 상수로 와 있었고 학습된 비피어 끌림이 그것을 정확히 상쇄. 거시 구조: "정답 말고는 멈출 수 없는 흐름"(정답 반경 0.42, 오답 |λ|=1.33). 오늘 산출물 전부 `2026-08-30/`.
- 다음 목표: ARC Prize 2026. 스도쿠 판 구조 그대로 2D 격자 과제로 이식 예정.
