import os, sys, json, numpy as np
ROOT="/workspace/LinearTuring"
OUT="/tmp/claude-0/-workspace-LinearTuring/8092b54b-030b-4f2c-b7a7-8107892a6681/scratchpad/runs"
NB=512
z=np.load(os.path.join(ROOT,"kaggle/upload/sudoku_lt_1k.npz"))
X=z["test_inputs"].reshape(-1,81).astype(np.int32)[:NB]; Y=z["test_labels"].reshape(-1,81).astype(np.int32)[:NB]
clue=(X>0); xt=(X+1).astype(np.int8); yt=(Y+1).astype(np.int8)
B=np.load(os.path.join(OUT,"base256.npz")); bm=json.loads(str(B["meta"]))
be128=B["exact128"]; be256=B["exact256"]; bp256=B["pred256"]
# 기준선(256세그)에서 틀린 것 = 목표 집합
bad=~be256; good=be256
# 기준선 오답 중 '단서 1개 덮어씀' 유형
ovw = ((bp256!=xt)&clue).sum(1)
print(f"# baseline: exact128={be128.sum()} exact256={be256.sum()} bad256={bad.sum()}")
print(f"#   그중 덮어쓴 단서 개수 분포:", {int(k):int(v) for k,v in zip(*np.unique(ovw[bad],return_counts=True))})
rows=[]
for tag in sys.argv[1:]:
    f=os.path.join(OUT,tag+".npz")
    if not os.path.exists(f): print("MISSING",tag); continue
    R=np.load(f); mm=json.loads(str(R["meta"]))
    e128,e256,p256=R["exact128"],R["exact256"],R["pred256"]
    fixed = bad & e256; broke = good & ~e256
    # 기제: 고쳐진 퍼즐에서 기준선이 덮어썼던 단서칸이 복원됐나
    bad_ov = bad & (ovw>0)
    restored = 0; tot_ov_cells=0; restored_cells=0
    for i in np.where(fixed & (ovw>0))[0]:
        cells = np.where(clue[i] & (bp256[i]!=xt[i]))[0]
        tot_ov_cells += len(cells)
        restored_cells += int((p256[i][cells]==xt[i][cells]).sum())
        if (p256[i][cells]==xt[i][cells]).all(): restored += 1
    # 개입판 전체 단서유지율
    rows.append(dict(tag=tag,mode=mm["mode"],S0=mm["S0"],rho=mm["rho_max"],
      e128=int(e128.sum()),e256=int(e256.sum()),
      fixed=int(fixed.sum()),broke=int(broke.sum()),net=int(e256.sum())-int(be256.sum()),
      cell256=mm["cell"][ "256"],clue_keep256=mm["clue_keep"]["256"],
      churn=round(mm["churn_128_256"],5),hn256=mm["hnorm"]["256"],hn128=mm["hnorm"]["128"],
      restored_puz=restored, restored_cells=f"{restored_cells}/{tot_ov_cells}"))
hdr=["tag","e128","e256","fixed","broke","net","cell256","clue_keep256","churn","hn128","hn256","restored_puz","restored_cells"]
print("\t".join(hdr))
print("\t".join(["BASE",str(int(be128.sum())),str(int(be256.sum())),"-","-","0",str(bm["cell"]["256"]),str(bm["clue_keep"]["256"] if "clue_keep" in bm else "-"),str(round(bm["churn_128_256"],5)),str(bm["hnorm"]["128"]),str(bm["hnorm"]["256"]),"-","-"]))
for r in rows: print("\t".join(str(r[h]) for h in hdr))
