# 2026-09-04 변경 기록

구조 변경 3건(블록 순서 `pre`→`post`, 스케일 게이지 상수화, 물리 레이어 스택)의
채택 근거와 수치는 [`DESIGN.md`](DESIGN.md) 에 따로 있다.

기존 루트 파일에 가한 변경은 **여기에 기록만** 한다 (파일 자체는 원위치).
오늘 새로 만든 산출물은 이 폴더와 `kaggle/` 에 있다.

---

## 1. `core/minimal.py` (수정)

### 1-1. `stdp_window: str = "beta"` 신설
`faithful` STDP 의 Γ 가 쓰는 창을 고른다.

- `beta` (기존 동작): `kcb = self.kernel(L, L.beta)` — β 로 창을 **다시 계산**
- `psi`: `kcb = None` → Γ 가 전달용 커널 `a`(ψ 가 들어간 원본)를 그대로 씀

### 1-2. `block_order: str = "pre"` 신설
블록 내부 순서.

- `pre` (기존): `경계 → 주입 → 어텐션+수송(Φ 포함)`
- `post`: `주입 → 어텐션+수송 → 경계 → Φ`

구현은 `step()` 에 `apply_phi` 인자를 추가해, `post` 일 때 Φ 를 스텝이 아니라
블록 끝(경계 뒤)에서 한 번 건다. v1 에서 Φ 를 아예 건너뛰어 판독 지점의
‖h‖ 가 29.7 (흡수 반지름 3.24) 까지 커진 버그가 있었고, Φ 를 블록 끝으로
옮겨 수정했다.

### 1-3. `inj_gate` 파라미터 삭제 → 상수 `√d`
URM/TRM 규약을 따른다.

```
self.embed_scale = math.sqrt(d)
self.embed = nn.Embedding(vocab, d)
trunc_normal_init_(self.embed.weight, std=1.0 / self.embed_scale)
```

배경: 310k 스텝 체크포인트의 `inj_gate` 실측값이 7.602 였다
(`gamma_init`/`inj_gate_init` 기본값 0.25 에서 출발). 임베딩에 wd=1.0 이
걸려 있고 `inj_gate` 는 no-decay 그룹이라, `β·‖E‖` 가 보존되는 방향으로
게이트가 커지고 임베딩이 줄어드는 게이지 미끄러짐이 있었다.

### 1-4. `gamma_raw` 학습 파라미터 삭제 → 상수 `1/d`
```
self.gamma = 1.0 / d       # 흡수 반지름 √d
```
`gamma_init` 설정 키는 폐기 목록에 넣어 사용 시 에러를 낸다.

### 1-5. `num_layers` 신설 + `LTLayer` 도입
물리 레이어 스택. 레이어당 파라미터를 `LTLayer` 로 묶고
`LT_Inner.layers = nn.ModuleList([...])` 로 보유한다.

레이어별 비공유: `wc_raw, psi, theta, alpha_raw, w_sh, eta_raw, lam_raw,
gain_raw, beta, (mu), (gate_s_raw), b_gate_up, b_down`
상태로서 스택을 관통하는 것: `h`, STDP 결합 `w`

총 블록 적용 횟수 = `blocks_per_seg × num_layers` (URM 규약).
`num_layers > 1` 이면 레이어 0 의 state_dict 를 복제해 초기화한다.

주입은 **블록마다** 유지했다. 저장소 주석(`block_inj`, "URM 의 cycle-주입과
동형; 밀도가 K 에 불변")대로 URM 의 L-cycle 이 우리 **블록**에 대응하며,
스택 반복 1회당 1번으로 바꾼 시험에서 4000 스텝 train acc 0.4541 → 0.4169
로 낮아져 되돌렸다.

### 1-6. `LTConfig` 검증기
알 수 없는 키와 폐기된 키(`gamma_init`, `inj_gate_init`)를 거부한다.

---

## 2. `sudoku/minimal.yaml` (수정)

```
+ blocks_per_seg: 8      # 스택 반복 횟수
+ num_layers: 1          # 물리 레이어 수. 총 블록 적용 = blocks_per_seg × num_layers
+ block_order: pre
+ stdp_window: beta
- inj_gate_init: 0.25
- gamma_init: 0.1
```

---

## 3. `kaggle/` (신규 디렉터리)

경로는 사용자가 지정한 `kaggle/` 를 유지한다 (붙여넣기 대상 파일이라
날짜 폴더로 옮기면 매번 경로가 바뀐다).

| 파일 | 내용 |
|---|---|
| `train_kaggle.py` | 자기완결 캐글 학습 스크립트 (1,739줄). 모델부는 `core/minimal.py` 를 그대로 옮긴 것 |
| `prep_dataset.py` | 업로드용 npz 빌더 |
| `upload/sudoku_lt_1k.npz` | 업로드 데이터 (증강은 코드에서 생성) |
| `upload/dataset_meta.json` | 데이터셋 메타 |
| `예외사항.md` | 코드로 해결 불가한 항목 + 사고 기록 |

`train_kaggle.py` 의 오늘 변경은 `KAGGLE_MULTIGPU.md` 참조.

---

## 4. 삭제

- `checkpoints/*` — 274 MB, 전부 삭제
- `torch_compile_debug/` — 43 MB, `compile_inspect.py` 부산물
- `__pycache__`, `.ipynb_checkpoints`

---

## 5. 로그

오늘 런의 로그는 기존 규약대로 `results/logs/` 에 있다 (3~5일차 로그와 같은 위치).
런별 수치는 `RUNS.md` 참조.
