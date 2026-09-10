"""v1.7 체크포인트별: 세그먼트마다 '무모순 격자'(행·열·박스 전부 1..9 순열) 개수. 정답 여부와 별개.
분류: 정답 / 무모순·단서변경 / 무모순·단서유지·오답 / 모순."""
import os, sys, importlib.util, glob, time, numpy as np, torch
ROOT="/workspace/LinearTuring"; torch.set_grad_enabled(False)
NB, SEG = 512, 128
s=importlib.util.spec_from_file_location("tv", os.path.join(ROOT,"2026-09-09/analysis/train_v17.py"))
tv=importlib.util.module_from_spec(s); s.loader.exec_module(tv)
P=os.environ["LT_CKPT"]
ck=torch.load(P, map_location="cpu", weights_only=False)
cfg=dict(ck["cfg"]); cfg.update(batch_size=NB, seq_len=81, num_puzzle_identifiers=1, loops=SEG+1)
m=tv.LT(cfg).cuda().eval()
full=dict(tv.strip_prefix(ck["model_state_dict"])); full.update(tv.strip_prefix(ck.get("ema_shadow",{})))
full={(k[len("model."):] if k.startswith("model.") else k):v for k,v in full.items()}
m.load_state_dict(full, strict=False)
z=np.load(os.path.join(ROOT,"kaggle/upload/sudoku_lt_1k.npz"))
X=z["test_inputs"].reshape(-1,81).astype(np.int32)[:NB]; Y=z["test_labels"].reshape(-1,81).astype(np.int32)[:NB]
x=torch.from_numpy(X+1).cuda(); y=torch.from_numpy(Y+1).cuda().long()
batch=dict(inputs=x,labels=y,puzzle_identifiers=torch.zeros(NB,dtype=torch.int32,device="cuda"))
UNITS=([[r*9+c for c in range(9)] for r in range(9)]+[[r*9+c for r in range(9)] for c in range(9)]
       +[[(br+a)*9+(bc+b) for a in range(3) for b in range(3)] for br in (0,3,6) for bc in (0,3,6)])
UI=np.array(UNITS)                                   # [27,9]
def valid(D):                                        # D [NB,81] digits (1..9, 다른 값=무효)
    ok=((D>=1)&(D<=9)).all(1)
    U=D[:,UI]                                        # [NB,27,9]
    perm=(np.sort(U,axis=2)==np.arange(1,10)).all(2).all(1)
    return ok&perm
clue=(X!=0)
t0=time.time()
with torch.device("cuda"): carry=m.initial_carry(batch)
rows=[]; PP=np.zeros((SEG,NB,81),np.int8)
for si in range(SEG):
    carry,out=m(carry,batch); p=out["logits"].argmax(-1).cpu().numpy(); PP[si]=p.astype(np.int8)
    D=p-1                                            # 토큰 2..10 → 숫자 1..9
    v=valid(D); correct=(D==Y).all(1)
    clue_kept=(D[clue]==X[clue]).reshape(-1) if False else np.array([(D[b][clue[b]]==X[b][clue[b]]).all() for b in range(NB)])
    rows.append((si+1, int(correct.sum()), int((v&~correct&~clue_kept).sum()), int((v&~correct&clue_kept).sum()), int((~v).sum())))
print(f"# {os.path.basename(P)} step={ck['step']} EMA  ({time.time()-t0:.0f}s)")
print(f"{'seg':>4} {'정답':>5} {'무모순·단서변경':>9} {'무모순·단서유지·오답':>11} {'모순':>5} | {'무모순 합':>7}")
for r in rows:
    if r[0] in (1,2,4,8,16,24,32,48,64,96,128):
        print(f"{r[0]:4d} {r[1]:5d} {r[2]:9d} {r[3]:11d} {r[4]:5d} | {r[1]+r[2]+r[3]:7d}")
bestv=max(rows,key=lambda r:r[1]+r[2]+r[3]); bestc=max(rows,key=lambda r:r[1])
np.savez_compressed(os.environ.get("LT_SAVE","/dev/null"),P=PP,X=X,Y=Y) if os.environ.get("LT_SAVE") else None
print(f"무모순 최대 {bestv[1]+bestv[2]+bestv[3]} @seg{bestv[0]}   정답 최대 {bestc[1]} @seg{bestc[0]}")
