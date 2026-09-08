import os,sys,json,numpy as np
ROOT="/workspace/LinearTuring"; NB=512
z=np.load(os.path.join(ROOT,"kaggle/upload/sudoku_lt_1k.npz"))
X=z["test_inputs"].reshape(-1,81).astype(np.int32)[:NB]; Y=z["test_labels"].reshape(-1,81).astype(np.int32)[:NB]
clue=(X>0); xt=(X+1).astype(np.int8)
B=np.load("runs/base256.npz"); be=B["exact256"]; bp=B["pred256"]
bad=~be                                   # 기준선 오답 125
ovw_cell = clue & (bp!=xt)                # 기준선이 덮어쓴 단서칸
has_ov = bad & (ovw_cell.sum(1)>0)        # 124
print(f"기준선 오답 {bad.sum()}, 그중 단서 덮어씀 {has_ov.sum()}, 덮어쓴 단서칸 총 {ovw_cell[bad].sum()}")
print("tag\tfixed\tnotfixed_restored\t미해결군_단서복원(칸)\t미해결군_전체단서유지")
for tag in sys.argv[1:]:
    f="runs/"+tag+".npz"
    if not os.path.exists(f): continue
    R=np.load(f); e=R["exact256"]; p=R["pred256"]
    fixed = bad & e
    nf = bad & ~e                          # 여전히 틀린 것
    # 미해결군에서 '기준선이 덮어썼던 단서칸' 이 복원된 비율 (비자명)
    m = ovw_cell & nf[:,None]
    rest = int(((p==xt) & m).sum()); tot = int(m.sum())
    # 미해결군의 전체 단서 유지율
    keep = float(((p==xt)&clue)[nf].sum() / clue[nf].sum())
    bkeep = float(((bp==xt)&clue)[nf].sum() / clue[nf].sum())
    print(f"{tag}\t{int(fixed.sum())}\t{rest}/{tot} ({rest/max(tot,1):.1%})\t\t단서유지 {keep:.4f} (기준선 동일집합 {bkeep:.4f})")
