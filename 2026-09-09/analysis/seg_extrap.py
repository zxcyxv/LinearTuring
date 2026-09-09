import os, sys, importlib.util, glob, time, numpy as np, torch
ROOT="/workspace/LinearTuring"; torch.set_grad_enabled(False)
NB = 512
SEG = int(os.environ.get("LT_SEG", "128"))   # 기본 128 = 기존 기록과 비교 가능한 값
s=importlib.util.spec_from_file_location("tv", os.path.join(ROOT,"2026-09-09/analysis/train_v17.py"))
tv=importlib.util.module_from_spec(s); s.loader.exec_module(tv)
P=os.environ.get("LT_CKPT") or sorted(glob.glob(os.path.join(ROOT,"2026-09-09/run_v17/step_*.pt")), key=os.path.getmtime)[-1]
ck=torch.load(P, map_location="cpu", weights_only=False)
cfg=dict(ck["cfg"]); cfg.update(batch_size=NB, seq_len=81, num_puzzle_identifiers=1, loops=SEG+1)
m=tv.LT(cfg).cuda().eval()
full=dict(tv.strip_prefix(ck["model_state_dict"])); full.update(tv.strip_prefix(ck.get("ema_shadow",{})))
full={(k[len("model."):] if k.startswith("model.") else k):v for k,v in full.items()}
miss,unexp = m.load_state_dict(full, strict=False)
print(f"# ckpt {os.path.basename(P)}  step={ck['step']}  weights=ema  missing={list(miss)} unexpected={list(unexp)[:3]}")
z=np.load(os.path.join(ROOT,"kaggle/upload/sudoku_lt_1k.npz"))
X=z["test_inputs"].reshape(-1,81).astype(np.int32)[:NB]; Y=z["test_labels"].reshape(-1,81).astype(np.int32)[:NB]
x=torch.from_numpy(X+1).cuda(); y=torch.from_numpy(Y+1).cuda().long()
batch=dict(inputs=x,labels=y,puzzle_identifiers=torch.zeros(NB,dtype=torch.int32,device="cuda"))
blank=(x==1)
t0=time.time()
with torch.device("cuda"): carry=m.initial_carry(batch)
prev=None; rows=[]
for si in range(SEG):
    carry,out=m(carry,batch); p=out["logits"].argmax(-1)
    acc=(p==y).float().mean().item(); ex=int((p==y).all(-1).sum())
    ch=0.0 if prev is None else ((p!=prev)&blank).float().sum().item()/blank.sum().item()
    prev=p.clone(); rows.append((si+1,acc,ex,ch))
print(f"{'seg':>5} {'acc':>8} {'exact':>6} {'exact%':>8} {'churn':>8}")
for si,acc,ex,ch in rows:
    if si in (1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256) or si==SEG:
        print(f"{si:5d} {acc:8.4f} {ex:6d} {100*ex/NB:8.2f} {ch:8.5f}" + ("  <-train" if si==16 else ""))
e16=rows[15][2]
def summarize(rs, tag):
    eb=max(r[2] for r in rs); sb=[r[0] for r in rs if r[2]==eb][0]
    cap = "  <-상한에 걸림" if sb==rs[-1][0] else ""
    print(f"seg16 {e16}/512 ({100*e16/NB:.2f}%)   best {eb}/512 ({100*eb/NB:.2f}%) @seg{sb}   +{eb-e16}{cap}   [{tag}]")
    return 100*((NB-e16)-(NB-eb))/(NB-e16)
print()
# 기존 기록과 직접 비교 가능한 값은 언제나 seg128 까지로 자른 것
r128 = summarize(rows[:128], "seg128 기준")
print(f"오차 감소율 = {r128:.1f}%   ({time.time()-t0:.0f}s)")
if SEG > 128:
    rS = summarize(rows, f"seg{SEG} 기준")
    print(f"오차 감소율(seg{SEG}) = {rS:.1f}%")
