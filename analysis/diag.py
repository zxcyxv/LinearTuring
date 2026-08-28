"""체크포인트 진단 (held-out 2048): 셀/완답 · 전파깊이 클래스별 정확도 · 오답의 행/열/박스 위반 분해 · 세그별 정확도.
사용: python diag.py [--ckpt PATH --bilinear 0|1] [--out JSON]
전파깊이 파일(cell_depth.npy)은 sudoku/cell_depth.py <test_dir> 로 생성.
원 결과: results/json/diag_*.json"""
import argparse, json, numpy as np, torch
from common import load_lt, load_test, make_batch, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.bs)
inp, lab, depth = load_test(); N = len(inp); L = m.config.loops
preds = torch.zeros(N, 81, dtype=torch.int32); segacc = np.zeros(L)
for i in range(0, N, args.bs):
    b = slice(i, min(i + args.bs, N)); n = b.stop - b.start
    bi = torch.zeros(args.bs, 81, dtype=torch.int32, device="cuda"); bl = bi.clone()
    bi[:n] = inp[b]; bl[:n] = lab[b]; batch = make_batch(bi, bl)
    with torch.device("cuda"): carry = m.initial_carry(batch)
    for s in range(L):
        carry, o = m(carry, batch)
        p = o["logits"][:n].argmax(-1).to(torch.int32)
        segacc[s] += (p == lab[b]).float().sum().item()
    preds[b] = p.cpu()
segacc /= N * 81
P, Lb, I = preds.numpy(), lab.cpu().numpy(), inp.cpu().numpy()
ok = P == Lb; res = {"cell": float(ok.mean()), "exact": int(ok.all(1).sum()), "N": N, "seg_cell": segacc.round(4).tolist()}
giv = I > 1; res["given"] = float(ok[giv].mean())
if depth is not None:
    for lo, hi, nm in ((1, 3, "prop1-2"), (3, 6, "prop3-5"), (6, 20, "prop6+")):
        mk = (depth >= lo) & (depth < hi); res[nm] = float(ok[mk].mean())
    res["search"] = float(ok[depth == -1].mean())
r = np.arange(81) // 9; c = np.arange(81) % 9; bx = (r // 3) * 3 + c // 3
def dup(units):
    out = np.zeros_like(P, dtype=bool)
    for u in range(9):
        idx = np.where(units == u)[0]; v = P[:, idx]
        same = v[:, :, None] == v[:, None, :]; np.einsum('nii->ni', same)[:] = False
        out[:, idx] = same.any(-1)
    return out
vr, vc, vb = dup(r), dup(c), dup(bx)
wrong = ~ok & ~giv; tot = wrong.sum(); res["wrong_cells"] = int(tot)
res["viol_row"] = float((wrong & vr).sum() / tot); res["viol_col"] = float((wrong & vc).sum() / tot); res["viol_box"] = float((wrong & vb).sum() / tot)
res["viol_none"] = float((wrong & ~vr & ~vc & ~vb).sum() / tot)   # 제약 위반 없이 틀림 (자기일관 오답)
print(json.dumps(res, indent=1, ensure_ascii=False))
if args.out: json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
