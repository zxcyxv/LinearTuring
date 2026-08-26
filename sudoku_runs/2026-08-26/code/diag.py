"""체크포인트 진단 (학습 불필요): 셀/exact · 전파깊이 클래스 분해 · 오답의 행/열/박스 위반 분해 · 세그별 정확도.
사용: python diag.py <ckpt.pt> <cfg_json> [out_json]
  cfg_json 예: '{"R":8,"blocks_per_seg":1}' (나머지는 d832 H8 boundary_mlp 기본)."""
import sys, json, numpy as np, torch
sys.path.insert(0, '/workspace/LinearTuring/refs/URM')
from models.lt.lt import LT
torch.set_grad_enabled(False)
ck, over = sys.argv[1], json.loads(sys.argv[2]); out = sys.argv[3] if len(sys.argv) > 3 else None
D = "/workspace/LinearTuring/refs/URM/data/sudoku-extreme-1k-aug-1000/test"
st = torch.load(ck, map_location='cuda', weights_only=False)
sd = st["model_state_dict"] if "model_state_dict" in st else st
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith(("model.", "_orig_mod.")) else k): v for k, v in sd.items()}
BS = 128
cfg = dict(batch_size=BS, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832,
           hidden_size=832, num_heads=8, R=8, loops=16, causal=False, boundary_mlp=True, ckpt=False,
           blocks_per_seg=1, block_inj=False)
cfg.update(over)
with torch.device("cuda"): m = LT(cfg)
miss, unexp = m.load_state_dict(sd, strict=False); print("missing", miss, "unexpected", unexp); m.eval()
inp = torch.from_numpy(np.load(f"{D}/all__inputs.npy")).to(torch.int32)
lab = torch.from_numpy(np.load(f"{D}/all__labels.npy")).to(torch.int32)
depth = np.load(f"{D}/cell_depth.npy"); N = len(inp)
preds = torch.zeros(N, 81, dtype=torch.int32); segacc = np.zeros(16)
for i in range(0, N, BS):
    b = slice(i, min(i+BS, N)); n = b.stop - b.start
    batch = dict(inputs=torch.zeros(BS,81,dtype=torch.int32), labels=torch.zeros(BS,81,dtype=torch.int32), puzzle_identifiers=torch.zeros(BS,dtype=torch.int32))
    batch["inputs"][:n] = inp[b]; batch["labels"][:n] = lab[b]; batch = {k: v.cuda() for k, v in batch.items()}
    with torch.device("cuda"): carry = m.initial_carry(batch)
    for s in range(16):
        carry, o = m(carry, batch)
        p = o["logits"][:n].argmax(-1).to(torch.int32).cpu()
        segacc[s] += (p == lab[b]).float().sum().item()
    preds[b] = p
segacc /= (N*81)
P, L, I = preds.numpy(), lab.numpy(), inp.numpy()
ok = P == L; res = {"cell": float(ok.mean()), "exact": int(ok.all(1).sum()), "N": N, "seg_cell": segacc.round(4).tolist()}
giv = I > 1; res["given"] = float(ok[giv].mean())
for lo, hi, nm in ((1,3,"prop1-2"), (3,6,"prop3-5"), (6,20,"prop6+")):
    mk = (depth >= lo) & (depth < hi); res[nm] = float(ok[mk].mean())
res["search"] = float(ok[depth == -1].mean())
# --- 제약 위반 분해 (예측 격자 기준) ---
r = np.arange(81)//9; c = np.arange(81)%9; bx = (r//3)*3 + c//3
def dup_mask(units):  # 각 칸이 자기 단위 안에서 같은 값과 충돌하는가 [N,81]
    out = np.zeros_like(P, dtype=bool)
    for u in range(9):
        idx = np.where(units == u)[0]; v = P[:, idx]              # [N,9]
        same = (v[:, :, None] == v[:, None, :]); np.einsum('nii->ni', same)[:] = False
        out[:, idx] = same.any(-1)
    return out
vr, vc, vb = dup_mask(r), dup_mask(c), dup_mask(bx)
wrong = ~ok & ~giv
tot = wrong.sum()
res["wrong_cells"] = int(tot)
res["viol_row"] = float((wrong & vr).sum()/tot); res["viol_col"] = float((wrong & vc).sum()/tot); res["viol_box"] = float((wrong & vb).sum()/tot)
res["viol_box_only"] = float((wrong & vb & ~vr & ~vc).sum()/tot); res["viol_rowcol_only"] = float((wrong & (vr|vc) & ~vb).sum()/tot)
res["viol_none"] = float((wrong & ~vr & ~vc & ~vb).sum()/tot)  # 위반 없이 틀림 (유일해 아닌 자기일관 오답)
res["grid_valid_but_wrong"] = int(((~(vr|vc|vb)).all(1) & ~ok.all(1)).sum())  # 격자 전체가 유효한데 정답 아님
# 오답 값의 출처: 주어진 피어가 이미 가진 값(즉시 배제 가능)을 썼는가
peers_given = np.zeros_like(P, dtype=bool)
for t in range(81):
    pm = ((r == r[t]) | (c == c[t]) | (bx == bx[t])); pm[t] = False
    peers_given[:, t] = ((I[:, pm] == P[:, t:t+1]) & (I[:, pm] > 1)).any(-1)
res["wrong_uses_given_peer_value"] = float((wrong & peers_given).sum()/tot)
print(json.dumps(res, indent=1, ensure_ascii=False))
if out: json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
