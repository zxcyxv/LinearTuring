# 캐글 4×L4 가 1×L4 보다 느렸던 건 — 진단과 수정

## 관측

캐글 4×L4, 전역 배치 128 (프로세스당 32), d832·N=2, `torch.compile`:

```
1%|  | 400/390625 [04:27<69:31:16, 1.56it/s]
[LT] step 400  lm_loss 1.3349  [fwd+bwd 614ms · allreduce 19ms]
```

같은 모델 1×L4 배치 128 은 2.79 it/s (358 ms/step). 4장이 순손실이었다.

## 배제한 가설

### 통신
`all_reduce` 실측 **18~20 ms**. 이론 상한도 작다:

| 그래디언트 57.2 MB (fp32), ring 전송 85.8 MB/GPU | |
|---|---|
| PCIe4 x16 P2P (20 GB/s) | 4.3 ms |
| host 경유 (6 GB/s) | 14.3 ms |
| P2P off, 느린 토폴로지 (1.5 GB/s) | 57.2 ms |

최악을 잡아도 57 ms 라 614 ms 를 설명하지 못한다.
(진단 전에 파라미터별 `all_reduce` 25회를 `_flatten_dense_tensors` 로
1회에 합치는 수정을 먼저 넣었으나, 이는 원인이 아니었다. 수정 자체는 유지.)

## 배치 스케일링 실측 (로컬 L4 단독, d832·N=2, window=psi)

```
        eager      compiled
bs128  513→ —     347.0 ms
bs96   513.2 ms      —
bs64   316.4 ms   182.5 ms
bs48   226.1 ms   137.5 ms
bs32   148.5 ms   544.1 ms   ←
```

- eager 는 배치에 선형 (32/64 = 0.47)
- compiled 는 bs48 까지 정상, **bs32 한 점에서만** 무너짐
- bs48 에서 배치 비례로 외삽한 bs32 기대값은 91.7 ms
- 재현성: 독립 3회 측정에서 544.1 / 543.6 / 544.7 ms
- GPU 를 다른 프로세스와 공유해도 bs32 는 545.3 → 544.1 ms 로 변화 없음
  (같은 조건에서 bs128 은 347 → 725 ms)

## 원인

`torch.profiler` 커널별 self CUDA 시간:

```
triton_per_fused__to_copy_add_cos_div_mul_neg_sin_sum
   bs48:  6 calls,   19.9 ms 합  →   3.32 ms/call   (전체 CUDA 의 5.0%)
   bs32:  6 calls, 1370.0 ms 합  → 228.39 ms/call   (전체 CUDA 의 84.4%)
```

같은 커널이 호출당 69배 느리다. step 당 2회 호출 = 457 ms.
`aten::mm`(289회)·`aten::bmm`(196회)의 호출 수와 시간은 배치에 비례해 정상
(mm 33.8 vs 51.5 ms = 0.656 ≈ 32/48).

연산 조합(`cos·sin·neg·sum`)으로 보아 감쇠 Gabor 커널
`a_tn = Σ_j e^{−α‖Δ‖₁} cos(Δφ_j − θ_j·Δ − ψ_j)` 의 역전파에서
ψ·θ 에 대한 j-축 축약이다. `triton_per_` 접두사는 Inductor 의
persistent reduction 을 뜻한다.

> 주의: `bench_train.py --profile` 은 `aten::` 연산만 집계하므로 Triton 커널이
> 빠진다. 그 표만 보면 bs32 의 커널 합이 54.6 ms 로 나와 "GPU 유휴"로
> 잘못 읽힌다. 실제 self CUDA 총합은 540 ms/step 으로 step 시간과 거의 같다.

## 수정

```python
import torch._inductor.config as _ic
_ic.triton.persistent_reductions = False
```

로컬 L4 실측 (d832·N=2, compiled):

| bs | 기본 | persistent_reductions=False | |
|---|---|---|---|
| 32 | 544.66 ms | **94.25 ms** | −83% |
| 48 | 137.52 ms | 133.85 ms | −3% |
| 64 | 182.49 ms | 173.45 ms | −5% |
| 128 | 347.00 ms | 345.12 ms | −0.5% |

`coordinate_descent_tuning=True` 도 bs32 에서 94.58 ms 로 같은 지점에
도달하나 컴파일 시간이 길다.

## 적용 후 캐글 실측

```
[LT] inductor: persistent_reductions=False (소배치 축약 병리 회피)
[LT] world_size=4 local_bs=32 params=14,292,171
1%| | 2400/390625 [03:50<12:02:20, 8.96it/s]
[LT] step 2400  lm_loss 0.5086  [fwd+bwd 84ms · allreduce 20ms]  acc 0.8430
```

step 634 ms → 104 ms. **6.1배.**

## `train_kaggle.py` 변경 3건

| 위치 | 내용 |
|---|---|
| L129 | `CFG["inductor_no_persist"]=True` 신설 |
| L1505–1507 | compile 직전 `persistent_reductions=False` 설정 + 로그 |
| L141 | `max_hours` 9.0 → 6.0 |
| L1463 | `/kaggle/input` 재개 구조 키에 `num_layers`, `mlp_expansion` 추가 |

마지막 건은 `num_layers` 를 이번에 도입하면서 생긴 구멍이다. 두 키 모두
파라미터 shape 을 바꾸므로 대조 목록에 없으면 `load_checkpoint` 가 shape
오류로 죽는다. `out_dir` 쪽 재개 경로에는 이 대조가 여전히 없다.

## quota 산수 (참고)

주당 30시간, 4×L4 는 2배 차감. 수정 후 4×L4 는 시간당 약 32k step.

| | step 시간 | 시간당 step | quota-시간당 step |
|---|---|---|---|
| 1×L4 (1배 차감) | 345 ms | 10.4k | 10.4k |
| 4×L4 (2배 차감) | 104 ms | 34.6k | 17.3k |
