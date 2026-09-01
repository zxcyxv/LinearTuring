# ARC-AGI 프로토콜 구성 — 스도쿠와 무엇이 다른가

`refs/URM` 코드를 직접 읽어 확인한 것. 추정 아님, 줄 번호를 붙였다.

## 1. ARC 과제가 데이터셋이 되는 과정

ARC 과제 하나는 이렇게 생겼다.

```
과제 007bbfb7:
  시연 1:  입력 A → 출력 A'      ← 정답 공개
  시연 2:  입력 B → 출력 B'      ← 정답 공개
  시연 3:  입력 C → 출력 C'      ← 정답 공개
  문제  :  입력 D → ???          ← 맞혀야 하는 것
```

`data/build_arc_dataset.py` 는 이 격자쌍들을 **각각 독립적인 seq2seq 예제로 풀어헤친다**.

```
예제 1:  A(900토큰) → A'(900토큰)
예제 2:  B → B'
예제 3:  C → C'
문제  :  D → D'
```

**모델은 한 번에 격자 하나만 본다.** 시퀀스 안에 "시연이 이랬으니 규칙은 이거다" 같은 맥락이 없다.
in-context 방식(LLM 계열 ARC 접근)과 근본적으로 다르다.

## 2. 그러면 규칙은 어디에 있는가 — 과제 임베딩

같은 과제의 네 예제 전부에 같은 `puzzle_identifier` 가 붙고, 그 번호가 임베딩 벡터가 되어 입력에 섞인다.

```python
# 우리 (core/minimal.py:257)   — 모든 위치에 더함
inj = embed(inputs) + pad(puzzle_emb(puzzle_identifiers))

# URM (models/urm/urm.py:119)  — 앞에 토큰으로 붙임
embedding = cat([puzzle_emb(ids).view(-1, puzzle_emb_len, hidden), embed(inputs)], dim=-2)
```

즉 모델이 하는 일은 **"격자 D와 번호 k를 받았다. k번 과제의 규칙대로 변환하라"** 이고,
**k번 과제의 규칙은 오직 `e_k` 벡터 안에만 있다.**

## 3. 핵심 — 평가 과제의 시연쌍이 학습셋에 들어간다

`data/build_arc_dataset.py:169-172, 218`:

```python
train_examples_dest = ("train", "all")                                   # 모든 과제의 시연쌍
test_examples_map = {config.test_set_name: [(1.0, ("test", "all"))],     # 평가 과제의 문제만
                     "_default":           [(1.0, ("train", "all"))]}
...
convert_single_arc_puzzle(results, name, puzzle, config.num_aug,
                          {"train": train_examples_dest, "test": test_examples_dest})
```

| 평가 과제 k | 정답이 주어지나 | 어디로 |
|---|---|---|
| **시연쌍** (A→A', B→B', C→C') | ARC가 원래 공개 | **학습 분할** |
| **문제** (D→?) | 숨겨진 정답 | **테스트 분할에만** |

**정답 유출은 없다.** 학습에 들어가는 것은 시연쌍이지 맞혀야 할 문제가 아니다.

그러나 결과적으로 **`e_k` 는 평가 과제 k의 시연쌍에 대한 경사하강으로 학습된다.**
이름만 사전학습이지, 실제로 일어나는 일은 *"테스트 과제의 시연쌍으로 그 과제 전용 파라미터를 학습하는 것"* 이다.
`puzzle_emb_lr` 이 스도쿠 1e-4 → ARC 1e-2 로 **100배** 뛰는 것이 이 통로를 세게 쓴다는 표시다.

ARC 규칙상 시연쌍 사용은 허용되므로 반칙이 아니다. 다만 **평가가 zero-shot 이 아니고**,
성능 중 얼마가 모델의 추론이고 얼마가 `e_k` 암기인지 분리되지 않는다.

**함의**: 이 모델들은 "새 규칙을 알아내는 능력"이 아니라 "임베딩에 이미 새겨진 규칙을 새 입력에 적용하는
능력"을 시험받는다. 진짜 새 과제에는 임베딩을 새로 학습시켜야 한다.

## 4. 증강마다 별도 식별자

`data/build_arc_dataset.py:105`:

```python
name_with_aug_repr = f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(...)}"
```

색 치환 + 이면체 변환으로 1000가지 증강을 만드는데 **각 변형본이 다른 이름 → 다른 식별자**를 받는다
(`:234-241` 의 `identifier_map`). 같은 과제가 1001개의 정체성을 갖는다.

그 증강은 **ARC 과제 자신의 대칭군**이다 (우리 스도쿠 증강과 같은 성격 — 같은 규칙을 이름만 바꿔 다시 쓴 것).
같은 규칙에 다른 정체성을 부여하면 모델이 (과제, 변형) 조합 1001개를 각각 외울 수 있다.

독립 분석 `papers/arXiv-2512.11847.pdf` 가 정면으로 지적한 지점:
> *"the ability to solve a particular evaluation puzzle is tightly linked to the associated ID token"*
> *"the 1000-sample voting pipeline improves Pass@1 by about 11 percentage points over single-pass canonical inference"*

## 5. 평가 프로토콜

`config/cfg_pretrain.yaml:13` → `evaluators: [arc@ARC]`
`evaluators/arc.py:44` → `pass_Ks=(1,2,5,10,100,1000)`, `aggregated_voting=True`

테스트 입력을 1000가지로 증강 → 각각 예측 → 역변환 → **다수결 투표** → pass@K.

## 6. 스도쿠 대 ARC 총정리

| | 스도쿠 | ARC-AGI |
|---|---|---|
| `seq_len` | 81 | **900** (30×30 고정 패딩, `:28 ARCMaxGridSize=30`) |
| 아키텍처 설정 | `loops16 H2 L6 layers4` | **완전히 동일** (`scripts/URM_*.sh`) |
| `epochs` | 50,000 | 200,000 |
| `eval_interval` | 2,000 | 10 |
| `weight_decay` | 1.0 | 0.1 |
| **`puzzle_emb_lr`** | 1e-4 | **1e-2** |
| `num_puzzle_identifiers` | **1** (blank 하나) | 과제 × 증강마다 하나 |
| 평가 과제 시연쌍 | 해당 없음 | **학습 분할에 포함** |
| `evaluators` | `[]` | `arc@ARC` |
| 평가 | 단일 패스, 완전일치 | 1000증강 + 다수결, pass@1~1000 |
| 병진 증강 | 없음 | train 분할에만 (`:250`) |

**우리 스도쿠 설정은 `num_puzzle_identifiers=1` — 모든 퍼즐이 벡터 하나를 공유하므로 과제 정체성 정보가 0이다.
2026-09-01 의 77.49% 는 이 통로 없이 낸 숫자다.**

## 7. 과제 임베딩의 메모리 (예상 못 한 병목)

`models/sparse_embedding.py`:
- `weights` 는 `nn.Buffer` — 파라미터가 아니라 **DDP 동기화 대상이 아니고, 샤딩도 없다 (GPU마다 전체 사본)**
- 순전파는 배치 행만 `local_weights [batch, dim]` 로 복사 — 활성 메모리는 배치 크기만큼
- 옵티마이저 `CastedSparseEmbeddingSignSGD_Distributed` — **SignSGD 라 상태가 없다** (Adam이면 3배 추가)
- `ema_shadow` 에 안 들어간다 (버퍼는 EMA 대상 아님)

ARC-AGI-1 = 학습 400 + 평가 400 과제 × 1001 증강 ≈ **801,000 식별자**

| | 차원 | 표 크기 (fp32, GPU당) |
|---|---:|---:|
| URM | 512 | **1.64 GB** |
| 우리 | 832 | **2.67 GB** |

L4 24GB 에서 11% 상시 점유. 우리는 배치가 이미 빠듯한 쪽이라 아프다.

**그런데 이 비용은 과제가 요구하는 것이 아니라 프로토콜 설계에서 나온다:**

| 선택 | 식별자 수 | 표 크기 | 부작용 |
|---|---:|---:|---|
| 그들 방식 (증강마다) | 801,000 | 2.67 GB | 암기 통로 열림 |
| **과제마다 하나** | **800** | **2.7 MB** | 증강을 가로질러 일반화해야 함 |
| 과제마다 + `puzzle_emb_ndim=128` | 800 | 0.4 MB | 위와 같음 |

`puzzle_emb_ndim` 은 어차피 hidden 까지 0 패딩되므로(`core/minimal.py:258`) 832 일 필요가 없다.

**과제당 하나로 바꾸면 1000배 줄고, 동시에 "임베딩 암기 덕분 아니냐"는 반론도 막힌다.**
ARC 이식 시 이것을 따를지가 별도 결정이며, 따르지 않으면 발표 수치들과 직접 비교가 안 되고
따르면 우리 구조의 기여가 임베딩에 가려진다.
