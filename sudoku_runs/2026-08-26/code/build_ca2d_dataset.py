"""2D 셀룰러 오토마타 데이터셋 — URM/ARC 형식 그대로 (30×30 캔버스, PAD=0, EOS=1, 색=2.., 과제 id = 규칙).
   ARC 로 옮길 때 데이터 빌더만 바꾸면 되도록 build_arc_dataset.py 의 인코딩·그룹 구조를 복제.
   규칙: Life-like B/S 표기 (외부총합). 격자 크기 무작위(ARC 처럼 가변), 경계 고정 0, k 스텝 또는 고정점까지.
사용: python build_ca2d_dataset.py --output-dir data/ca2d-k4 --k 4 --n-train 2000 --n-test 256
"""
import os, json, argparse, numpy as np
RULES = {  # 이름: (B, S, Wolfram 클래스 대략)
    "vote":       ({5,6,7,8}, {4,5,6,7,8}, 2),
    "flakes":     ({3}, {0,1,2,3,4,5,6,7,8}, 2),
    "life":       ({3}, {2,3}, 4),
    "daynight":   ({3,6,7,8}, {3,4,6,7,8}, 4),
    "seeds":      ({2}, set(), 3),
    "replicator": ({1,3,5,7}, {1,3,5,7}, 3),
}
def step2d(g, B, S):
    n = sum(np.roll(np.roll(g, dr, 0), dc, 1) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0))
    # 고정 경계: roll 이 감싸지 않도록 가장자리 이웃 계산을 패딩으로
    p = np.pad(g, 1); n = sum(p[1+dr:1+dr+g.shape[0], 1+dc:1+dc+g.shape[1]] for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0))
    born = (g==0) & np.isin(n, list(B)); surv = (g==1) & np.isin(n, list(S))
    return (born | surv).astype(np.uint8)
def encode(grid, pad_r, pad_c, M=30):
    h, w = grid.shape; out = np.zeros((M, M), np.uint8)
    out[pad_r:pad_r+h, pad_c:pad_c+w] = grid + 2
    if pad_r+h < M: out[pad_r+h, pad_c:pad_c+w] = 1
    if pad_c+w < M: out[pad_r:pad_r+h, pad_c+w] = 1
    return out.flatten()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fixpoint", action="store_true", help="k 대신 고정점까지(최대 k) 반복"); ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=256); ap.add_argument("--min-size", type=int, default=8); ap.add_argument("--max-size", type=int, default=20)
    ap.add_argument("--density", type=float, default=0.35); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rules", default=",".join(RULES))
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); rules = a.rules.split(",")
    for split, n in (("train", a.n_train), ("test", a.n_test)):
        R = {k: [] for k in ["inputs","labels","puzzle_identifiers","puzzle_indices","group_indices"]}; R["puzzle_indices"].append(0); R["group_indices"].append(0)
        ex = pz = 0
        for ri, name in enumerate(rules):
            B, S, _ = RULES[name]
            for _ in range(n):
                h, w = rng.integers(a.min_size, a.max_size+1, 2); g = (rng.random((h, w)) < a.density).astype(np.uint8); y = g.copy()
                for _ in range(a.k):
                    y2 = step2d(y, B, S)
                    if a.fixpoint and (y2 == y).all(): break
                    y = y2
                pr, pc = (rng.integers(0, 30-h+1), rng.integers(0, 30-w+1)) if split == "train" else (0, 0)
                R["inputs"].append(encode(g, pr, pc)); R["labels"].append(encode(y, pr, pc)); ex += 1
                R["puzzle_indices"].append(ex); R["puzzle_identifiers"].append(ri + 1); pz += 1   # 0 = blank id
                R["group_indices"].append(pz)   # 그룹 = 예제 1개 (증강 없음)
        os.makedirs(f"{a.output_dir}/{split}", exist_ok=True)
        for k, v in R.items(): np.save(f"{a.output_dir}/{split}/all__{k}.npy", np.stack(v,0) if k in ("inputs","labels") else np.array(v, np.int32))
        meta = dict(pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=12, seq_len=900, num_puzzle_identifiers=len(rules)+1,
                    total_groups=pz, mean_puzzle_examples=1.0, sets=["all"])
        json.dump(meta, open(f"{a.output_dir}/{split}/dataset.json","w")); print(split, "예제", ex, "규칙", rules)
    json.dump(["<blank>"]+rules, open(f"{a.output_dir}/identifiers.json","w"))
    json.dump({n: {"B": sorted(RULES[n][0]), "S": sorted(RULES[n][1]), "class": RULES[n][2]} for n in rules}, open(f"{a.output_dir}/rules.json","w"), indent=1)
if __name__ == "__main__": main()
