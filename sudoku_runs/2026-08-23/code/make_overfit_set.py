"""과적합 진단용 소수 데이터셋 생성 (URM 하네스 포맷).

sudoku_csv/train.csv 앞쪽에서 N개 퍼즐을 뽑아 train/test 를 **동일하게** 만든다.
  - test = train 이므로 eval 지표가 곧 "학습한 바로 그 예제를 맞히는가" = 암기 여부
  - 증강 없음 (증강은 일반화 테스트가 되어 목적과 어긋남)
  - num_puzzle_identifiers=1 유지 → 퍼즐 임베딩으로 외우는 지름길 없음
"""
import argparse, csv, json, os
import numpy as np
from data.common import PuzzleDatasetMetadata

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--out", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--src", default="data/sudoku_csv/train.csv")
a = ap.parse_args()
out = a.out or f"data/sudoku-of{a.n}"

rows = []
with open(a.src, newline="") as f:
    rd = csv.reader(f); next(rd)
    for i, (src, q, ans, rating) in enumerate(rd):
        rows.append((q, ans, int(rating)))
        if len(rows) >= a.n * 200: break          # 앞쪽 일부만 읽고 그 안에서 표본
rng = np.random.default_rng(a.seed)
idx = rng.choice(len(rows), size=a.n, replace=False)
sel = [rows[i] for i in idx]

inp = np.stack([np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8) - ord('0') for q, _, _ in sel])
lab = np.stack([np.frombuffer(ans.encode(), dtype=np.uint8) - ord('0') for _, ans, _ in sel])
assert inp.shape == (a.n, 81) and lab.shape == (a.n, 81)
assert inp.min() >= 0 and inp.max() <= 9 and lab.min() >= 1 and lab.max() <= 9
inp, lab = inp + 1, lab + 1                        # 빌더와 동일한 +1 (0=PAD)

data = dict(
    inputs=inp, labels=lab,
    group_indices=np.arange(a.n + 1, dtype=np.int32),
    puzzle_indices=np.arange(a.n + 1, dtype=np.int32),
    puzzle_identifiers=np.zeros(a.n, dtype=np.int32),
)
meta = PuzzleDatasetMetadata(
    seq_len=81, vocab_size=11, pad_id=0, ignore_label_id=0, blank_identifier_id=0,
    num_puzzle_identifiers=1, total_groups=a.n, mean_puzzle_examples=1, sets=["all"])

for split in ("train", "test"):                     # test = train (암기 측정)
    d = os.path.join(out, split); os.makedirs(d, exist_ok=True)
    json.dump(meta.model_dump(), open(os.path.join(d, "dataset.json"), "w"))
    for k, v in data.items(): np.save(os.path.join(d, f"all__{k}.npy"), v)
json.dump(["<blank>"], open(os.path.join(out, "identifiers.json"), "w"))
givens = (inp > 1).sum(1)
print(f"{out}: {a.n} 퍼즐 · givens {givens.min()}~{givens.max()} (평균 {givens.mean():.1f}) · train=test")
