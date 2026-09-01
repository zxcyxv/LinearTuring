"""w 의 대각 w_tt 는 칸마다 균일한가(감쇠기) 아니면 칸을 고르는가(선택기).

물음: 자기결합이 정답을 맞힌 칸에서 더 크면 그것은 선택기이고, 균일하면 관성(제동)이다.
구조상 a_tt = Σ_j r_tj²·cos(ψ_j) 이고 Σ_j r_tj² = 1 (주소 정규화) 이므로 거의 상수여야 한다 — 확인한다.
분할: 단서칸 / 빈칸-최종정답 / 빈칸-최종오답.
사용: python 2026-09-01/analysis/w_diag.py [--n 256] [--segs 192]
"""
import argparse, json, os, sys, importlib.util
import numpy as np, torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URM = os.path.join(ROOT, "refs", "URM"); TEST = os.path.join(URM, "data", "sudoku-extreme-1k-aug-1000", "test")
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256); ap.add_argument("--segs", type=int, default=192)
ap.add_argument("--marks", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 192])
ap.add_argument("--out", default=os.path.join(ROOT, "2026-09-01", "results", "json", "w_diag.json"))
args = ap.parse_args()
N = args.n
inp = torch.from_numpy(np.load(f"{TEST}/all__inputs.npy")).to(torch.int32)[:N].cuda()
lab = torch.from_numpy(np.load(f"{TEST}/all__labels.npy")).to(torch.int32)[:N].cuda().long()
pid = torch.from_numpy(np.load(f"{TEST}/all__puzzle_identifiers.npy")).to(torch.int32)[:N].cuda()
cfg = yaml.safe_load(open(f"{ROOT}/checkpoints/R1B8_min_faith_config.yaml"))["arch"]
cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
cfg.update(batch_size=N, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, loops=args.segs + 1)
ck = torch.load(f"{ROOT}/checkpoints/R1B8_min_faith_step310527.pt", map_location="cpu", weights_only=False)
full = dict(ck["raw_model_state_dict"]); full.update(ck["model_state_dict"])
m = mod.LT(cfg).cuda().eval()
m.load_state_dict({k.replace("_orig_mod.model.", ""): v for k, v in full.items()}, strict=True)
inner = m.inner; AB = inner.W_C(); kc = inner.kernel()

batch = dict(inputs=inp, labels=lab, puzzle_identifiers=pid)
carry = mod.LTCarry(current_hidden=torch.empty(N, 81, cfg["hidden_size"], device="cuda"),
                    steps=torch.zeros(N, dtype=torch.int32, device="cuda"),
                    halted=torch.ones(N, dtype=torch.bool, device="cuda"),
                    current_data={k: torch.empty_like(v) for k, v in batch.items()})
clue = (inp != 1)                                  # 단서칸 (빈칸 토큰 = 1)
snap = {}
for s in range(args.segs):
    carry, out = m(carry, batch)
    if s + 1 in args.marks:
        snap[s + 1] = (torch.diagonal(carry.coupling.float(), dim1=-2, dim2=-1).clone(),   # [N,H,81]
                       out["logits"].argmax(-1).clone())
pred_fin = snap[args.segs][1]
ok_fin = (pred_fin == lab)                          # [N,81] 최종 정답 여부
a_diag = torch.diagonal(inner.attn(carry.current_hidden.float(), AB, kc).float(), dim1=-2, dim2=-1)

def stats(d, mask):
    """d [N,H,81], mask [N,81] → 해당 칸들의 평균."""
    mm = mask.unsqueeze(1).expand_as(d)
    return float(d[mm].mean()) if mm.any() else float("nan")

print(f"step={ck['step']}  N={N}  최종 완답 {100*ok_fin.all(-1).float().mean():.2f}%")
print(f"\n[대각의 균일성]  칸 간 표준편차 / 평균 = 변동계수")
print(f"{'seg':>5} | {'평균':>8} {'칸간 표준편차':>12} {'변동계수':>9} | {'헤드간 표준편차':>13}")
rows = []
for s in args.marks:
    d = snap[s][0]                                   # [N,H,81]
    mu = d.mean(); sd_cell = d.std(-1).mean(); sd_head = d.mean(-1).std(-1).mean()
    print(f"{s:5d} | {float(mu):8.4f} {float(sd_cell):12.4f} {float(sd_cell/mu.abs()):9.4f} | {float(sd_head):13.4f}")
    rows.append(dict(seg=s, mean=float(mu), sd_cell=float(sd_cell), cv=float(sd_cell/mu.abs())))

print(f"\n[칸 범주별 대각 평균]  선택기라면 '빈칸-정답' 이 '빈칸-오답' 보다 커야 한다")
print(f"{'seg':>5} | {'단서칸':>9} {'빈칸-정답':>10} {'빈칸-오답':>10} | {'정답−오답':>10} {'(표준편차 대비)':>14}")
blank = ~clue
for s in args.marks:
    d = snap[s][0]
    g1 = stats(d, clue); g2 = stats(d, blank & ok_fin); g3 = stats(d, blank & ~ok_fin)
    sd = float(d.std(-1).mean())
    print(f"{s:5d} | {g1:9.4f} {g2:10.4f} {g3:10.4f} | {g2-g3:+10.4f} {(g2-g3)/sd:+14.3f}σ")
    rows[args.marks.index(s)].update(clue=g1, blank_ok=g2, blank_bad=g3)

print(f"\n[참고] 커널 대각 a_tt: 평균 {float(a_diag.mean()):.4f}  칸간 표준편차 {float(a_diag.std(-1).mean()):.4f}  "
      f"변동계수 {float(a_diag.std(-1).mean()/a_diag.mean().abs()):.4f}")
d = snap[args.segs][0]
r = torch.stack([torch.corrcoef(torch.stack([d[:, h].flatten().double(),
                 ok_fin.flatten().double()]))[0, 1] for h in range(d.shape[1])])
print(f"[상관] w_tt 와 최종 정답 여부의 헤드별 피어슨 r: " + " ".join(f"{x:+.3f}" for x in r.tolist()))
print(f"       |r| 최대 {float(r.abs().max()):.3f}")
json.dump(dict(step=int(ck['step']), n=N, rows=rows, corr=r.tolist()), open(args.out, "w"), indent=1)
