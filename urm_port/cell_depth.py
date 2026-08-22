"""칸별 전파 깊이 — naked/hidden single 반복으로 각 칸이 확정되는 라운드 수.
   깊이 0 = 주어진 칸, k = k라운드 전파로 확정, -1 = 단순 전파로 미확정 (탐색 필요)."""
import numpy as np

def prop_depth(board):                       # board: [81] 0=빈칸
    g = board.reshape(9, 9).copy()
    depth = np.where(g > 0, 0, -1)
    UNITS = []
    for i in range(9):
        UNITS.append([(i, j) for j in range(9)])
        UNITS.append([(j, i) for j in range(9)])
    for bi in range(3):
        for bj in range(3):
            UNITS.append([(bi*3+r, bj*3+c) for r in range(3) for c in range(3)])
    rnd = 0
    while True:
        rnd += 1
        cand = {}
        for i in range(9):
            for j in range(9):
                if g[i, j] == 0:
                    used = set(g[i, :]) | set(g[:, j]) | set(g[i//3*3:i//3*3+3, j//3*3:j//3*3+3].flatten())
                    cand[(i, j)] = set(range(1, 10)) - used
        newly = []
        for (i, j), c in cand.items():       # naked single
            if len(c) == 1: newly.append((i, j, c.pop()))
        for unit in UNITS:                    # hidden single
            empt = [(p, cand[p]) for p in unit if p in cand]
            for v in range(1, 10):
                places = [p for p, c in empt if v in c]
                if len(places) == 1 and g[places[0]] == 0:
                    newly.append((*places[0], v))
        newly = {(i, j): v for i, j, v in newly}
        if not newly: break
        for (i, j), v in newly.items():
            if g[i, j] == 0: g[i, j] = v; depth[i, j] = rnd
    return depth.flatten(), int((g == 0).sum())

if __name__ == "__main__":
    import sys
    D = sys.argv[1]
    inp = np.load(f"{D}/all__inputs.npy") - 1          # 토큰→값 (1=빈칸→0)
    N = len(inp)
    depths = np.zeros((N, 81), dtype=np.int8); unsolved = np.zeros(N, dtype=np.int32)
    for n in range(N):
        depths[n], unsolved[n] = prop_depth(inp[n].astype(np.int64))
    np.save(f"{D}/cell_depth.npy", depths)
    d = depths[depths >= 0]
    print(f"{N}개 퍼즐:  단순 전파로 완전 해결 {int((unsolved==0).sum())}개 ({(unsolved==0).mean()*100:.1f}%)")
    print(f"칸 분포:  주어짐 {(depths==0).mean()*100:.1f}%  전파확정 {((depths>0).mean())*100:.1f}%  탐색필요 {(depths==-1).mean()*100:.1f}%")
    for k in range(1, 9):
        print(f"  깊이 {k}: {(depths==k).mean()*100:5.2f}%", end="")
        if k % 4 == 0: print()
    print(f"\n최대 전파 깊이: {depths.max()}")
