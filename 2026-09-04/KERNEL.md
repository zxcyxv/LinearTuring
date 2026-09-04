# 커널 수준 최적화 — 어디를 쳐야 하는가 (2026-09-04)

> 대상: `core/minimal.py` 스도쿠 판 (d832·H8·블록8·loops16·STDP faithful), 배치 128, T=81, L4 1장.
> 단위는 `2026-09-01/COST.md` 와 같다: **1 step = 세그먼트 1개 fwd + bwd + optimizer.step()**.
> 모든 수치는 실측이거나 실측에서 유도한 것이다. 추정은 **추정이라고 적었다.**

## 0. 결론 먼저

1. **이 모델은 L4 peak 의 18.9% 로 돈다.** 그런데 URM 기준선도 16.3% 다 — **우리만의 결함이 아니라 T=81·소형 재귀 모델 전체의 체제**다.
2. **연산량 지분과 시간 지분이 9.1배 어긋나는 곳이 있다.** T×T 경로(어텐션·STDP)는 MAC 의 0.8%인데 시간의 7.6%를 먹는다. 실효 효율 **peak 의 1.9%**.
3. 그래서 커널 작업의 표적은 큰 GEMM 이 아니라 **① T×T 경로 융합 ② 경계 GEMM 의 에필로그 융합** 둘이다. 큰 GEMM 은 이미 cuBLAS 가 잘 돌린다.
4. **현실적 상한은 1.3~1.6배** (174 → 110~130 ms, 완주 19시간 → 12~14시간). 2배 이상은 수치형식(FP8)이나 구조 변경 없이는 어렵다.
5. 구조를 건드릴 수 있다면 **`inter=2304` 축소가 커널 작업 전체보다 크다** (블록 MAC 의 71.6%). 둘은 곱해진다.

---

## 1. 실측 (배치 128 · T=81 · L4 · `2026-09-04/analysis/bench_train.py`)

| 설정 | ms/step | it/s | peak GB | MFU |
|---|---:|---:|---:|---:|
| eager, STDP 끔 | 283.1 | 3.53 | 3.97 | — |
| eager, window=psi | 369.7 | 2.70 | 5.32 | 8.9% |
| eager, window=beta | 438.2 | 2.28 | 5.88 | — |
| **compile, window=psi** | **174.4** | **5.73** | **3.85** | **18.9%** |
| compile, window=beta | 188.8 | 5.30 | 4.33 | 17.5% |
| compile, STDP 끔 | 145.3 | 6.88 | 2.82 | 22.7% |
| compile + 그래프브레이크 제거 | 173.3 | 5.77 | 3.85 | 19.0% |
| **URM 표준 프로토콜 (compile)** | **353.2** | **2.83** | **6.36** | **16.3%** |
| URM 표준 (eager) | 687.3 | 1.45 | 7.70 | — |

- 실제 학습(`pretrain.py`)에서 5.65 it/s 로 재현됐다 → **하네스에 숨은 오버헤드는 없다.**
- forward 만(no_grad, compile, psi) **57.3 ms** = 스텝의 32.8%.
- MFU 계산: 세그먼트 forward 666 G-MAC = 1.332 TFLOP, backward ≈ 2× → 스텝 **4.00 TFLOP**. L4 BF16 dense peak 121 TFLOPS. FLOP 한계는 **33.0 ms**.

### 1.1 `torch.compile` 은 배치에 종속이다 — 캐글에서 중요

`2026-09-01/COST.md §3` 은 배치 32 에서 compile 이 **느리다**고 적었다(9×9 STDP 켬: eager 84.0 → compile 97.9 ms). 그 관찰은 유효하고, 배치 128 에서는 뒤집힌다(369.7 → 174.4, **2.12배**). 즉 그 결론은 배치 32 한정이다.

**캐글 4×L4 에 전역 배치 128 을 쓰면 프로세스당 32 다 — 정확히 compile 이 손해였던 구간이다.** 4-GPU 발사 전에 배치 32 에서 compile on/off 를 반드시 다시 재야 한다. 손해로 나오면 선택지는 (a) 전역 배치를 올려 프로세스당 128 을 유지하거나(프로토콜 변경) (b) compile 을 끄는 것이다.

> **[2026-09-04 추가] 측정 완료 — 원인이 특정됐고 세 번째 선택지가 있었다.** d832·N=2 에서 compiled bs32 = 544.1 ms, eager bs32 = 148.5 ms, compiled bs48 = 137.5 ms 로 절벽이 bs32 한 점에만 있었다. 원인은 Gabor 커널 역전파의 Inductor persistent reduction 단 하나(호출당 bs48 3.32 ms → bs32 228.39 ms). `torch._inductor.config.triton.persistent_reductions = False` 로 bs32 가 94.25 ms 가 되고 bs64/128 에는 회귀가 없다. 전역 배치도 lr 도 안 바꿨다. 상세: [`KAGGLE_MULTIGPU.md`](KAGGLE_MULTIGPU.md).

---

## 2. 시간이 어디로 가는가

### 2.1 연산량 분해 (블록 1회, forward MAC, window=psi)

| 항목 | G-MAC | 지분 |
|---|---:|---:|
| 경계 `gate_up` (mm) | 39.75 | 47.7% |
| 경계 `down` (mm) | 19.87 | 23.9% |
| 주소 사영 `W_C` | 7.18 | 8.6% |
| 값 사영 `W` | 7.18 | 8.6% |
| 출력 `Wᵀ` | 7.18 | 8.6% |
| 어텐션 `a` | 0.70 | 0.8% |
| `agree` | 0.70 | 0.8% |
| 메시지 `o` | 0.70 | 0.8% |
| **합** | **83.25** | 100% |

경계가 **71.6%**, 사영이 25.9%, T×T 가 2.4%.

### 2.2 결정적 증거 — MAC 지분과 시간 지분의 괴리

`stdp_window` 플래그는 **블록당 `attn_xy` 를 정확히 한 번 더 부르는가**만 바꾼다. 그 차분이 순수한 T×T 경로 비용이다.

```
beta − psi = 188.8 − 174.4 = 14.4 ms/step
그 연산량      = 33.5 GFLOP
실효 성능       = 2.33 TFLOPS = peak 의 1.9%
MAC 지분 0.8%  vs  시간 지분 7.6%   →  9.1배 괴리
```

inductor 덤프(`2026-09-04/results/inductor_code.txt`)가 이유를 보여준다. T×T 경로는 전부 **batch 1024 의 초소형 bmm** 으로 나간다:

```
extern_kernels.bmm(..., (1024, 81,  52))  ×32     ← attn_xy 의 qx·kx, qy·ky
extern_kernels.bmm(..., (1024, 81, 104))  ×32     ← agree, o
extern_kernels.bmm(..., (1024, 81,  81))  ×64     ← 위들의 backward
```
81×52×81 = 0.34 M-MAC 짜리 GEMM 을 1024개 배치로 던지는 것이라 텐서코어가 거의 놀고, 대신 `[B,H,81,81]` 텐서(13.4 MB)를 계속 쓰고 읽는다.

### 2.3 정정 — 큰 GEMM 은 이미 잘 돌고 있다

구두로 "헤드별 einsum 이 N=52·104 짜리 스키니 GEMM 을 만든다, 병합하면 이득"이라고 말했는데 **틀렸다.** inductor 덤프에서 사영은 전부 헤드가 합쳐진 단일 `mm` 으로 나간다:

```
extern_kernels.mm(reinterpret_tensor(buf16,(10368, 832)), buf17)   ← 주소 A (N=416)
extern_kernels.mm(reinterpret_tensor(buf16,(10368, 832)), buf19)   ← 주소 B (N=416)
extern_kernels.mm(reinterpret_tensor(buf16,(10368, 832)), buf30)   ← 값 (N=832)
mm 모양 분포: (10368,832)×64 · (10368,4608)×14 · (10368,2304)×16 · (4608,10368)×16 · (832,10368)×16
```
M=10368 의 큰 GEMM 이라 cuBLAS 가 제 성능을 낸다. **헤드 병합은 할 일이 없다.**

### 2.4 시간 배분 추정

측정된 차분으로 나누면:

| 구간 | ms/step | 근거 |
|---|---:|---|
| 기반(경계 + 사영 + Φ + 손실 + 옵티마이저) | 145.3 | STDP 끈 실측 |
| STDP 본체(`agree`·`Γ`·`w` EMA·`a_eff`) | **29.1** | 174.4 − 145.3 |
| (`window=beta` 면 `a_β` 추가) | +14.4 | 188.8 − 174.4 |

기반 145.3 안에서는 **추정**이다: 경계 GEMM 을 65% 효율로 잡으면 fwd 12.1 ms, 사영 4.4 ms, 나머지 T×T(`a`·`o`)와 elementwise 가 fwd 57.3 중 약 40 ms.

**정리하면 T×T 계열(`a` + `o` + STDP 전부)이 스텝의 약 30% 를 먹는데 연산량은 2.4% 다.** 구두로는 "스도쿠에서는 T×T 융합이 2.4%짜리라 값을 못 한다"고 했는데 **그것도 틀렸다.** MAC 지분으로 판단한 오류다. 실제로는 가장 큰 표적이다.

---

## 3. 최적화 목록

### Tier 0 — 이미 켜져 있음 (확인됨)

| | 상태 | 근거 |
|---|---|---|
| `torch.compile` | ON | `minimal.yaml` 에 `profile` 키가 없어 `pretrain.py:183` 비활성 조건에 안 걸림. eager 대비 2.12배 |
| bf16 autocast | ON | `arch.amp=true` |
| gradient checkpointing | OFF(=빠름) | `arch.ckpt=false`. minimal 은 구현 자체가 없다 |
| 블록 가중치 공유 | — | 7.16M(14 MB)이 L2(48 MB)에 상주 → 가중치 트래픽 무시 가능 |

### Tier 1 — 코드 변경 최소, 커널 안 짬

| # | 항목 | 근거 | 추정 이득 | 위험 |
|---|---|---|---|---|
| 1 | **`torch.compile(mode="max-autotune")`** | 지금은 GEMM 이 cuBLAS 로 나가 **에필로그 융합이 불가**하다. Triton GEMM 템플릿을 쓰면 `gate_up` 직후 `0.5·g·u` 를 에필로그에 붙일 수 있고, 초소형 bmm 도 자동 튜닝 대상이 된다 | 경계 트래픽 절약분만으로 fwd 5.1 ms(스텝 약 15 ms, **8%**). bmm 개선분은 미지 | 컴파일 시간 수 분↑. 수치 동일성 확인 필요 |
| 2 | `qx·kx + qy·ky` 를 **한 bmm 으로** | 지금은 K=52 짜리 bmm 두 개를 더한다. `[qx|qy]·[kx|ky]ᵀ` 로 쓰면 K=104 하나 | 런치 절반, K 두 배 | cat 트래픽이 늘 수 있음 — 융합되는지 확인 필요 |
| 3 | `mode="reduce-overhead"` (CUDA graphs) | 스텝당 extern 488 + triton 커널 다수 ≈ 700~900 런치. 5 µs/런치면 3.5~4.5 ms | **2~3%** | `w`·carry 가 스텝 간 상태라 캡처 가능 여부 확인 필요 |
| 4 | 그래프 브레이크 제거 | `CastedSparseEmbedding.forward` 의 `if torch.any(...)` 가 데이터 의존 분기 → 그래프 10개/브레이크 9개. 제거하면 1개/0개 | **0.6%** (174.4 → 173.3, 실측) | 없음. 이미 `kaggle/train_kaggle.py` 에는 반영됨 |

Tier 1 합계 추정 **10~13%** (1·3·4 실측 근거, 2 는 추정).

### Tier 2 — 커스텀 커널 (진짜 커널 연구)

**T2-A. 융합 STDP-어텐션 — 가장 큰 표적**

지금 블록마다 `[B,H,81,81]` 텐서가 다섯 개 생겼다 사라진다: `a`, `agree`, `Γ`, `w`, `a_eff`. 각 13.4 MB(bf16). 그리고 그것들을 만드는 GEMM 은 전부 batch 1024 의 초소형 bmm 이다(§2.2).

FlashAttention 방식으로 타일 안에서 `a` 를 만들고, `agree` 를 곱하고, `w` 를 갱신하고, `a_eff` 를 만들고, 곧바로 `v` 와 축약하면 **`w` 하나만 메모리에 남는다.** `w` 는 블록을 넘어 사는 상태라 없앨 수 없지만(읽기+쓰기 8×2×13.4 MB = 214 MB/step, 0.7 ms — 무시 가능), 나머지 넷은 전부 레지스터/공유메모리 안에서 끝난다.

- **표준 flash 커널에 없는 요소**: T×T 상태를 하나 이고 가는 것. 그래서 backward 도 flash 식 재계산 + `w` 의 블록별 스냅샷 관리가 필요하다.
- 표적 시간: `a`(14.4 ms 실측) + STDP 본체(29.1 ms 실측) + `o` ≈ **스텝의 30%**.
- **추정 이득 15~25%** (표적의 절반~2/3 제거).
- T=900(ARC)에서는 `2026-09-01/COST.md §5` 측정으로 이 경로가 활성 메모리의 **80%** 다. 스도쿠에서 만들어 두면 ARC 진입 비용이 그대로 회수된다.

**T2-B. 융합 쌍선형 경계**

`gate_up` 이 `[B,T,4608]`(95.6 MB)을 메모리에 쓰고, chunk 해서 곱하고, 다시 읽어 `down` 에 넣는다. 블록당 fwd 트래픽 약 290 MB, 8블록 2.26 GB → peak BW 에서 8.1 ms.
타일 단위로 `g,u` 를 레지스터에서 만들어 바로 `down` 에 누적하면(fused SwiGLU 와 같은 수법) 왕복 1.42 GB 가 사라진다 → **fwd 5.1 ms, 스텝 약 15 ms(8%)**.
Tier 1-1(max-autotune)이 같은 것을 부분적으로 해 준다. **먼저 max-autotune 을 재고, 남는 만큼만 손으로 짜는 게 맞다.**

### Tier 3 — 수치형식·구조 (커널 밖)

| 항목 | 근거 | 판단 |
|---|---|---|
| FP8 (Ada 텐서코어) | GEMM 이 MAC 의 72% | **지금은 아니다.** MFU 18.9% 라 FLOP 이 병목이 아니다. 대역폭을 먼저 고친 뒤에야 값을 한다. 정확도 위험 + transformer-engine 도입 비용도 크다 |
| **`inter` 축소** | 경계가 블록 MAC 의 71.6%. `inter=2304` 는 트랜스포머 SwiGLU 관례(`4d × 2/3`)에서 온 상속값이고 이 구조에서 검토된 적이 없다 | **커널 작업 전체보다 크다.** 2304→768 이면 블록 83.25→43.5 G-MAC. 다만 모델 변경이라 성능을 다시 봐야 한다 |
| 배치 확대 | 대역폭 병목이라 배치가 클수록 상각된다. TRM 은 같은 과제를 배치 768 로 돈다 | 스텝 수·유효 lr·비교 기준이 전부 바뀐다. 순수 최적화가 아니다 |
| `w` 저계수화 | `2026-09-01 §7` 측정: `w` 는 대각 에너지 81.7%, 비대각 안정랭크 6~7.6 | O(T²) → O(T·r). **ARC 로 갈 때 필수**, 스도쿠에서는 이득 작음 |

---

## 4. 상한

```
현재            174.4 ms/step   MFU 18.9%
FLOP 한계(peak)  33.0 ms        MFU 100%  ← 도달 불가
```

부분별 상한을 더한 현실적 목표:

| 시나리오 | 절약 | ms/step | 배수 | 완주(390,625 step) |
|---|---|---:|---:|---|
| 현재 | — | 174.4 | 1.00 | 19h 10m |
| Tier 1 전부 | −18 ms | ~156 | 1.12 | 17h |
| + T2-A(보수적 절반) | −27 ms | ~129 | 1.35 | 14h |
| + T2-B(max-autotune 잔여) | −15 ms | ~114 | **1.53** | **12.6h** |
| 낙관 (T2-A 2/3 제거) | −60 ms | ~100 | 1.74 | 11h |

**커널 작업만으로는 1.3~1.6배가 현실이고 2배는 낙관이다.** MFU 로는 19% → 29~33%. 소형 재귀 모델이 T=81 에서 그 이상을 내는 예는 흔치 않다(URM 기준선이 16.3%인 것과 같은 이유).

구조를 건드리면 곱해진다: `inter` 2304→768 은 그 자체로 약 1.5배이고, 커널 1.5배와 합치면 2.2배(174 → 80 ms, 완주 8.7h)가 된다.

---

## 5. 하지 말아야 할 것 (근거 있음)

- **헤드별 einsum 을 손으로 병합** — inductor 가 이미 단일 `mm` 으로 낸다(§2.3). 할 일 없음.
- **그래디언트 체크포인팅** — peak 3.85 GB / 23 GB 라 메모리가 병목이 아니다. 켜면 연산만 는다.
- **MAC 지분으로 표적 고르기** — T×T 경로가 0.8% MAC / 7.6% 시간이다. **9.1배 어긋난다.** 반드시 시간으로 골라야 한다.
- **`linalg_qr` 최적화** — eager 4.79 ms(1.3%)이고 이미 세그먼트당 1회다.
- **작은 배치에서 compile 을 당연시** — 배치 32 에서는 손해였다(§1.1).

---

## 6. 검증 방법 (측정 없이 결론 내지 말 것)

```bash
# 기본 벤치 (GPU 를 다른 작업과 공유하면 값이 오염된다 — HANDOVER §3)
python 2026-09-04/analysis/bench_train.py --bs 128 --window psi --compile
python 2026-09-04/analysis/bench_train.py --bs 128 --window psi --compile --profile --fwd-only-split

# 캐글 4-GPU 대비: 프로세스당 배치에서 compile 이 이득인지
python 2026-09-04/analysis/bench_train.py --bs 32 --window psi
python 2026-09-04/analysis/bench_train.py --bs 32 --window psi --compile

# 배치 스케일링 (예제당 비용)
for b in 128 256 512; do python 2026-09-04/analysis/bench_train.py --bs $b --window psi --compile; done

# 커널 구성·그래프 브레이크
python 2026-09-04/analysis/compile_inspect.py --bs 128 --window psi
TORCH_LOGS=output_code python 2026-09-04/analysis/bench_train.py --bs 128 --window psi --compile --warmup 2 --iters 2 2> code.txt

# 해석 모델 (GPU 불필요)
python 2026-09-04/analysis/cost_model.py --window psi
python 2026-09-04/analysis/cost_model.py --window psi --bs 8 --grid 30   # ARC 크기
```

## 7. 미측정 (이 문서에서 추정으로 남은 것)

- `max-autotune` 실제 이득
- `reduce-overhead`(CUDA graphs)가 `w` 상태와 양립하는지
- ~~배치 32 에서 compile 의 부호 (캐글 4-GPU 의 전제)~~ → 해결, `KAGGLE_MULTIGPU.md`
- 배치 256/512 의 예제당 비용
- T2-A/T2-B 커널의 실제 이득 (설계만 있고 구현 없음)
- 기반 145.3 ms 안의 세부 배분(경계 GEMM 65% 효율 가정에 의존)
