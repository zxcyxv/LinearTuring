"""제약전파 솔버: 각 칸이 몇 번째 라운드에 naked/hidden single 로 확정되는지 기록."""
import numpy as np
def peers(i):
    r, c = divmod(i, 9); br, bc = r//3*3, c//3*3
    s = set()
    for k in range(9):
        s.add(r*9+k); s.add(k*9+c)
    for a in range(br, br+3):
        for b in range(bc, bc+3): s.add(a*9+b)
    s.discard(i); return sorted(s)
PEERS = [peers(i) for i in range(81)]
UNITS = ([[r*9+c for c in range(9)] for r in range(9)]
       + [[r*9+c for r in range(9)] for c in range(9)]
       + [[(br+a)*9+(bc+b) for a in range(3) for b in range(3)] for br in (0,3,6) for bc in (0,3,6)])

def propagate(grid):
    """grid: 81 int (0=빈칸). 반환: (round_of_cell[81], method[81], 최종 grid, 막혔는지)"""
    g = grid.copy()
    cand = [set(range(1,10)) if g[i]==0 else {g[i]} for i in range(81)]
    for i in range(81):
        if g[i]: 
            for p in PEERS[i]: cand[p].discard(g[i])
    rnd = np.full(81, -1); meth = [""]*81
    rnd[g>0] = 0; 
    for i in range(81):
        if g[i]: meth[i] = "clue"
    r = 0
    while (g==0).any():
        r += 1; found = []
        for i in range(81):                                   # naked single
            if g[i]==0 and len(cand[i])==1:
                found.append((i, next(iter(cand[i])), "naked"))
        for U in UNITS:                                       # hidden single
            for d in range(1,10):
                spots = [i for i in U if g[i]==0 and d in cand[i]]
                if len(spots)==1 and not any(f[0]==spots[0] for f in found):
                    found.append((spots[0], d, "hidden"))
        if not found: return rnd, meth, g, True               # 막힘 → 고급 기법 필요
        for i, d, mth in found:
            if g[i]: continue
            g[i] = d; cand[i] = {d}; rnd[i] = r; meth[i] = mth
            for p in PEERS[i]: cand[p].discard(d)
    return rnd, meth, g, False

if __name__ == "__main__":
    d = np.load("v1_traj.npz"); X, Y = d["X"], d["Y"]
    for pid in (115, 295, 460):
        rnd, meth, g, stuck = propagate(X[pid].astype(int))
        solved_ok = (g == Y[pid]).all()
        nb = (X[pid]==0).sum()
        print(f"#{pid}  단서 {81-nb}  빈칸 {nb}   전파만으로 {'막힘' if stuck else '완주'}   정답일치 {solved_ok}")
        if not stuck:
            import collections
            c = collections.Counter(rnd[rnd>0])
            print(f"     라운드별 확정 칸수: {dict(sorted(c.items()))}   최대깊이 {rnd.max()}")
        else:
            print(f"     전파로 확정된 칸 {(rnd>0).sum()}/{nb},  남은 {(g==0).sum()}칸은 고급기법/탐색 필요")
