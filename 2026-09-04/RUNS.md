# 2026-09-04 런 기록

전부 Sudoku-Extreme 1k(증강 1000), 전역 배치 128, lr 1e-4, wd 1.0,
`lr_min_ratio=1.0`(warmup 2000 후 상수 LR), EMA 0.999, `loops=16`,
평가 집합 2048 퍼즐. 로그는 `results/logs/<이름>.log`.

측정 노이즈 (앞선 세션에서 확정): 같은 설정 재실행 ±0.002,
런 내 점간 ±0.02.

---

## 1. STDP 창 · 블록 순서

| 런 | 설정 | 4000 step train acc |
|---|---|---|
| FAITH_psi_pre | `stdp_window=psi`, `block_order=pre` | 0.7012 |
| FAITH_beta_pre | `stdp_window=beta`, `block_order=pre` | 0.7047 |
| FAITH_psi_post | `stdp_window=psi`, `block_order=post` | 0.7180 |

`FAITH_psi_post` 는 9000 step 까지 진행: eval 7812 에서 acc 0.6571, exact 48/2048.

## 2. 게이지 정렬 (γ=1/d, 주입 √d 상수, embed init_std=1/√d)

| 런 | 2000 step train acc | 4000 step |
|---|---|---|
| SQRTD_post | 0.5913 | 0.6727 |
| GAUGE_post | 0.7341 | 0.7357 |

`GAUGE_post` 는 초기화 시 logit std 0.061 → 0.516.

## 3. 깊이 (`num_layers`)

총 블록 적용 = `blocks_per_seg × num_layers`. 반복 깊이와 물리 깊이 두 축이
곱해진다 (설계는 `DESIGN.md §3`).

- `DEEP2_d832` — 물리 깊이 2축 (d832, N=2, blocks 8). 1600 step 에서 중단
- `DEPTH16_d832` — 반복 깊이만 2배 (d832, N=1, blocks 16). 200 step 에서 중단

파라미터를 맞춘 비교는 d576 에서 했다.

| 런 | d | num_layers | mlp_expansion | params |
|---|---|---|---|---|
| GAUGE_post | 832 | 1 | 4.0 | 7.16 M |
| DEEP2_d576 | 576 | 2 | 4.0 | 6.65 M |

`DEEP2_d576` eval 11718: acc 0.6704, exact 172/2048.

## 4. 쌍선형 경계 폭 절제

`DEEP2_d576`(exp 4.0, 6.65 M) 대 `BND256_d576`(exp 0.5, inter=256,
2.23 M = 0.335×). 그 외 설정 동일 (d576, num_layers=2, blocks_per_seg=8,
`block_order=post`, `stdp_window=psi`, `stdp_eta_init=0.05`).

### eval (2048 퍼즐)

| eval step | DEEP2 acc | DEEP2 exact | BND256 acc | BND256 exact |
|---|---|---|---|---|
| 1953 | 0.5048 | 0/2048 | 0.4784 | 0/2048 |
| 3906 | 0.6339 | 3/2048 | 0.6063 | 0/2048 |
| 5859 | 0.6502 | 58/2048 | 0.6433 | 46/2048 |
| 7812 | 0.6556 | 100/2048 | 0.6326 | 47/2048 |
| 9765 | 0.6659 | 153/2048 | 0.6567 | 97/2048 |
| 11718 | 0.6704 | 172/2048 | 0.6543 | 107/2048 |
| 13671 | — | — | 0.6466 | 35/2048 |

`DEEP2_d576` 은 12200 step 에서 종료, `BND256_d576` 은 계속 진행 중.
BND256 의 13671 exact 35 는 직전 11718 의 107 에서 내려간 값이다.

### train cell acc

| train step | DEEP2 | BND256 | 차 |
|---|---|---|---|
| 2000 | 0.6668 | 0.6583 | −0.0085 |
| 4000 | 0.7949 | 0.7678 | −0.0271 |
| 6000 | 0.8288 | 0.7867 | −0.0421 |
| 8000 | 0.8125 | 0.7706 | −0.0419 |
| 10000 | 0.7963 | 0.7888 | −0.0075 |
| 12000 | 0.8089 | 0.7602 | −0.0487 |

### 경계 폭 대 step 시간 (로컬 L4, bs32, d832·N=2, compiled + nopersist)

| mlp_expansion | inter | params | step | 속도비 |
|---|---|---|---|---|
| 4.0 | 3328 | 14.29 M | 95.3 ms | 1.00× |
| 2.0 | 1664 | 9.18 M | 79.1 ms | 1.21× |
| 1.0 | 832 | 6.62 M | 71.4 ms | 1.34× |
| 0.5 | 416 | 5.35 M | 66.3 ms | 1.44× |

파라미터 2.67배 감소에 step 시간 1.44배 감소.

### 미측정

`mlp_expansion` 1.0 과 2.0 의 학습 곡선. exp 0.5 의 12000 step 이후 곡선.

---

## 5. 속도 참조

| 런 | 내용 |
|---|---|
| SPEED_psi | 우리 모델 속도 측정용, 200 step |
| URM_speed | URM 표준 프로토콜 속도 측정용, 600 step |

상세는 `KERNEL.md`.

---

## 6. 캐글 본 런 (진행 중)

d832, `num_layers=2`, `blocks_per_seg=8`, `mlp_expansion=4.0`,
`block_order=post`, `stdp_window=psi`, `stdp_eta_init=0.05`,
params 14,292,171. 4×L4, 전역 배치 128, `max_hours=6.0`.

step 2400 시점: `lm_loss 0.5086`, train acc 0.8430, `fwd+bwd 84ms ·
allreduce 20ms`, 8.96 it/s. 6시간 기준 예상 종료 지점 약 190k step.
