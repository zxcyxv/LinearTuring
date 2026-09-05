"""세 군(해결 / 미해결·진동 / 미해결·고정점)에서 샘플 하나씩 뽑아 퍼즐 특징을 그대로 보여준다.
캐글 182k, 테스트 앞 256 퍼즐, 256 세그. 진동 = 마지막 32 세그 동안 답이 한 번이라도 바뀐 칸."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False)
N, SEGS, TAIL = 256, 256, 32
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=N, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
X = z["test_inputs"][:N].reshape(N, 81).astype(int); Y = z["test_labels"][:N].reshape(N, 81).astype(int)
x = torch.from_numpy(X.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device="cuda"))
with torch.device("cuda"): carry = m.initial_carry(batch)
preds = []; t0 = time.time()
for si in range(SEGS):
    carry, o = m(carry, batch); preds.append(o["logits"].argmax(-1).cpu().numpy() - 1)
    if (si + 1) % 64 == 0: print(f"  seg {si+1}/{SEGS}  {time.time()-t0:.0f}s  완답 {int((preds[-1] == Y).all(-1).sum())}/{N}", flush=True)
P = np.stack(preds)                                    # [seg, N, 81]
final = P[-1]; solved = (final == Y).all(-1)
osc_cells = (np.diff(P[-TAIL:], axis=0) != 0).any(0)   # [N,81]
osc = osc_cells.any(-1)
grp = np.where(solved, 0, np.where(osc, 1, 2))          # 0 해결, 1 진동, 2 고정점
print(f"\n군 크기: 해결 {int((grp==0).sum())}  미해결·진동 {int((grp==1).sum())}  미해결·고정점 {int((grp==2).sum())}\n")
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (bx[:, None] == bx[None])) & ~np.eye(81, dtype=bool)
def singles_fill(g):
    """naked/hidden single 만으로 채우는 칸 수 (난이도 대리: 많을수록 쉬움)."""
    g = g.copy(); filled = 0
    while True:
        cand = [set(range(1, 10)) - set(g[peer[i]]) if g[i] == 0 else set() for i in range(81)]
        prog = False
        for i in range(81):
            if g[i] == 0 and len(cand[i]) == 1: g[i] = cand[i].pop(); filled += 1; prog = True
        for i in range(81):
            if g[i]: continue
            for v in list(cand[i]):
                units = [np.where(r == r[i])[0], np.where(c == c[i])[0], np.where(bx == bx[i])[0]]
                if any(all((g[j] != 0 or v not in cand[j]) for j in u if j != i) for u in units):
                    g[i] = v; filled += 1; prog = True; break
        if not prog: return filled, int((g == 0).sum())
def grid(vals, mark=None, mark2=None):
    rows = []
    for i in range(9):
        cells = []
        for j in range(9):
            k = i * 9 + j; v = vals[k]; ch = "." if v == 0 else str(v)
            if mark is not None and mark[k]: ch = ch + "*"
            elif mark2 is not None and mark2[k]: ch = ch + "~"
            else: ch = ch + " "
            cells.append(ch)
            if j in (2, 5): cells.append("|")
        rows.append(" ".join(cells))
        if i in (2, 5): rows.append("-" * 29)
    return "\n".join(rows)
for g, name in [(0, "해결"), (1, "미해결·진동"), (2, "미해결·고정점")]:
    idx = np.where(grp == g)[0]
    if len(idx) == 0: print(f"=== {name}: 없음"); continue
    i = int(idx[0]); givens = int((X[i] != 0).sum()); sf, left = singles_fill(X[i])
    wrong = final[i] != Y[i]; viol = int(((final[i][:, None] == final[i][None]) & peer).sum() // 2)
    first_solved = next((k + 1 for k in range(SEGS) if (P[k, i] == Y[i]).all()), None)
    n_wrong_hist = [(int((P[k, i] != Y[i]).sum())) for k in (0, 7, 15, 31, 63, 127, 255)]
    print(f"=== {name}  (퍼즐 #{i}, 군 내 {len(idx)}개 중 첫 번째)")
    print(f"  주어진 칸 {givens}  |  싱글 전파로 채워지는 칸 {sf}/{81-givens} (남는 빈칸 {left})  |  끝 틀린 칸 {int(wrong.sum())}  위반 쌍 {viol}  진동 칸 {int(osc_cells[i].sum())}")
    print(f"  틀린 칸 수 추이 (seg 1,8,16,32,64,128,256): {n_wrong_hist}" + (f"   최초 완답 seg {first_solved}" if first_solved else ""))
    if g == 1:
        cyc = [tuple(P[k, i][osc_cells[i]]) for k in range(SEGS - 8, SEGS)]
        print(f"  진동 칸 값 (마지막 8세그, 진동칸 순서대로): " + " / ".join("".join(map(str, t)) for t in cyc))
    print("  입력 (주어진 칸)              모델 끝 답 (*=틀림, ~=진동)          정답")
    a, b, d = grid(X[i]).split("\n"), grid(final[i], wrong, osc_cells[i]).split("\n"), grid(Y[i]).split("\n")
    for l1, l2, l3 in zip(a, b, d): print(f"  {l1:30s} {l2:30s} {l3}")
    print()
