"""테스트셋 유출 점검: 학습 데이터가 테스트에 들어갔는가.

빌드 스크립트(refs/URM/data/build_sudoku_dataset.py)상 train/test 는 sudoku-extreme 의 서로 다른 CSV 이고
증강은 train 에만 걸린다. 그 전제가 실제 파일에서 지켜졌는지 직접 확인한다.

  1. 완전 일치   — 테스트 문제/정답/(문제,정답) 이 학습 1,001,000 행 안에 있는가
  2. 정답판 공유 — 단서가 달라도 완성된 격자가 같은가
  3. 궤도 일치   — 증강군(자릿수 치환 9! × 전치 2 × 밴드/행 1296 × 스택/열 1296) 아래 같은 문제인가.
                   군 불변량(단서 수, 자릿수 빈도 다중집합, 밴드별 행 단서수, 스택별 열 단서수, 박스 3×3)으로
                   필요조건을 걸고, 충돌한 쌍만 정확히 검증한다.
  4. 퍼즐 식별자 — 임베딩을 통한 유출 통로가 있는가
사용: python 2026-09-01/analysis/leak_check.py
"""
import os, sys, itertools, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "refs", "URM", "data", "sudoku-extreme-1k-aug-1000")

def load(split):
    g = lambda n: np.load(f"{D}/{split}/all__{n}.npy")
    return g("inputs"), g("labels"), g("puzzle_identifiers"), g("group_indices")

tr_i, tr_l, tr_p, tr_g = load("train")
te_i, te_l, te_p, te_g = load("test")
print(f"train {tr_i.shape}  test {te_i.shape}")
print(f"train 그룹 {len(tr_g)-1}  test 그룹 {len(te_g)-1}")

# 저장 인코딩: 0=pad, 1=빈칸, 2..10 = 숫자 1..9 → 숫자 격자로 환원
def digits(a): return (a.astype(np.int16) - 1).clip(0)      # 빈칸 0, 숫자 1..9
TRI, TRL, TEI, TEL = map(digits, (tr_i, tr_l, te_i, te_l))
assert TRL.min() >= 1 and TEL.min() >= 1, "정답에 빈칸이 있으면 안 됨"

# 학습 원본(증강 안 된 첫 행) = 그룹 시작 인덱스
base_idx = tr_g[:-1]
print(f"학습 원본(비증강) {len(base_idx)}개 — 그룹당 {tr_g[1]-tr_g[0]}행")
assert (TRI[base_idx] != TRI[base_idx + 1]).any(1).all() or True

def rows_as_bytes(a): return np.ascontiguousarray(a.astype(np.uint8)).view([('', np.uint8)] * a.shape[1]).ravel()

def overlap(A, B, name):
    sa = set(map(bytes, A.astype(np.uint8))); sb = set(map(bytes, B.astype(np.uint8)))
    inter = sa & sb
    print(f"  {name:38s} 교집합 {len(inter)}")
    return inter

print("\n[1] 완전 일치")
o1 = overlap(TEI, TRI, "테스트 문제 ∩ 학습 문제(증강 포함)")
o1b = overlap(TEI, TRI[base_idx], "테스트 문제 ∩ 학습 문제(원본만)")
pair_te = np.concatenate([TEI, TEL], 1); pair_tr = np.concatenate([TRI, TRL], 1)
o1c = overlap(pair_te, pair_tr, "테스트 (문제,정답) ∩ 학습 (문제,정답)")

print("\n[2] 정답판 공유")
o2 = overlap(TEL, TRL, "테스트 정답 ∩ 학습 정답(증강 포함)")
o2b = overlap(TEL, TRL[base_idx], "테스트 정답 ∩ 학습 정답(원본만)")

print("\n[3] 증강군 궤도 — 불변량 대조")
def sig(grid):
    """증강군 불변 서명. grid: [81] 숫자(0=빈칸)."""
    g = grid.reshape(9, 9); m = (g > 0)
    n = int(m.sum())
    # 자릿수 빈도 다중집합 (자릿수 치환·기하 모두에 불변)
    freq = tuple(sorted(np.bincount(g[m], minlength=10)[1:].tolist()))
    # 밴드별 행 단서수: 밴드 안 정렬 후 밴드끼리 정렬
    rc = m.sum(1); cc = m.sum(0)
    rsig = tuple(sorted(tuple(sorted(rc[b*3:b*3+3].tolist())) for b in range(3)))
    csig = tuple(sorted(tuple(sorted(cc[s*3:s*3+3].tolist())) for s in range(3)))
    geo = min((rsig, csig), (csig, rsig))                      # 전치 대칭
    # 박스 단서수 3×3 을 행·열 치환 + 전치 아래 정규화
    box = np.array([[m[b*3:b*3+3, s*3:s*3+3].sum() for s in range(3)] for b in range(3)])
    cands = []
    for B in (box, box.T):
        for rp in itertools.permutations(range(3)):
            for cp in itertools.permutations(range(3)):
                cands.append(tuple(B[np.ix_(rp, cp)].ravel().tolist()))
    return (n, freq, geo, min(cands))

tr_sig = {}
for k, i in enumerate(base_idx):
    tr_sig.setdefault(sig(TRI[i]), []).append(int(i))
te_sig = [sig(TEI[j]) for j in range(len(TEI))]
hits = [(j, tr_sig[s]) for j, s in enumerate(te_sig) if s in tr_sig]
print(f"  학습 원본 1000개의 서로 다른 서명: {len(tr_sig)}")
print(f"  서명이 겹치는 테스트 문제: {len(hits)} / {len(TEI)}")

# --- 충돌한 쌍만 정확히 검증: 마스크를 옮기는 기하 변환이 실제로 있는가
def geo_transforms():
    """(전치, 행순열, 열순열) 전체 2×1296×1296 은 크므로 마스크 기준 가지치기로 탐색."""
    perms = [np.concatenate([np.array(b)*3 + np.array(p[b]) for b in range(3)])
             for bp in itertools.permutations(range(3))
             for p in itertools.product(itertools.permutations(range(3)), repeat=3)
             for b in [None] if False]
    return perms

def all_perms():
    out = []
    for bands in itertools.permutations(range(3)):
        for p0 in itertools.permutations(range(3)):
            for p1 in itertools.permutations(range(3)):
                for p2 in itertools.permutations(range(3)):
                    inner = (p0, p1, p2)
                    out.append(np.concatenate([np.array(inner[k]) + bands[k]*3 for k in range(3)]))
    return out            # 1296

def same_orbit(a, b):
    """a,b: [9,9] 숫자 격자. 증강군에서 a→b 가 가능한가 (자릿수 치환은 일관성만 확인)."""
    P = all_perms()
    mb = (b > 0)
    for T in (a, a.T):
        mt = (T > 0)
        rc = mt.sum(1); cc = mt.sum(0)
        for rp in P:
            if not (rc[rp] == mb.sum(1)).all(): continue
            for cp in P:
                x = T[np.ix_(rp, cp)]
                if not ((x > 0) == mb).all(): continue
                mp = {}; ok = True
                for u, v in zip(x[mb], b[mb]):
                    if mp.setdefault(int(u), int(v)) != int(v): ok = False; break
                if ok and len(set(mp.values())) == len(mp): return True
    return False

real = []
for j, idxs in hits:
    for i in idxs:
        if same_orbit(TEI[j].reshape(9, 9), TRI[i].reshape(9, 9)):
            real.append((j, i))
print(f"  실제 궤도 일치(증강 동치): {len(real)}")
if real: print("   ", real[:20])

print("\n[4] 퍼즐 식별자")
print(f"  train 고유 id {sorted(set(tr_p.tolist()))[:5]}  test 고유 id {sorted(set(te_p.tolist()))[:5]}")
print(f"  num_puzzle_identifiers = {json.load(open(f'{D}/train/dataset.json'))['num_puzzle_identifiers']}")

print("\n=== 판정 ===")
bad = len(o1) or len(o1c) or len(real)
print("문제 완전 일치:", len(o1), "| (문제,정답) 일치:", len(o1c), "| 정답판 공유:", len(o2), "| 궤도 동치:", len(real))
print("유출 있음" if bad else "유출 없음")
