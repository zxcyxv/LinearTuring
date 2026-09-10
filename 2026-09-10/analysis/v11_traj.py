"""v1.1 마일스톤 체크포인트 → 512 퍼즐 × seg128 예측 저장 (state_an / conv_an 입력)."""
import os, sys, importlib.util, numpy as np, torch
ROOT="/workspace/LinearTuring"; SP="/tmp/claude-0/-workspace-LinearTuring/8092b54b-030b-4f2c-b7a7-8107892a6681/scratchpad"
torch.set_grad_enabled(False); NB,SEG=512,128
P=os.environ["LT_CKPT"]; MOD=os.environ.get("LT_MOD", ROOT+"/2026-09-09/analysis/train_v11.py")
s=importlib.util.spec_from_file_location("tk",MOD); tk=importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck=torch.load(P,map_location="cpu",weights_only=False)
cfg=dict(ck["cfg"]); cfg.update(batch_size=NB,seq_len=81,num_puzzle_identifiers=1,loops=SEG+1)
m=tk.LT(cfg).cuda().eval()
def strip(k): k=k.replace("_orig_mod.",""); return k[len("model."):] if k.startswith("model.") else k
sd={strip(k):v for k,v in ck["model_state_dict"].items()}
if ck.get("ema_shadow"): sd.update({strip(k):v for k,v in ck["ema_shadow"].items()})
miss,unexp=m.load_state_dict(sd,strict=False); print(f"# {os.path.basename(P)} step={ck['step']} EMA  missing={list(miss)[:3]} unexpected={list(unexp)[:3]}")
z=np.load(os.path.join(ROOT,"kaggle/upload/sudoku_lt_1k.npz"))
X=z["test_inputs"].reshape(-1,81).astype(np.int32)[:NB]; Y=z["test_labels"].reshape(-1,81).astype(np.int32)[:NB]
x=torch.from_numpy(X+1).cuda(); y=torch.from_numpy(Y+1).cuda().long()
batch=dict(inputs=x,labels=y,puzzle_identifiers=torch.zeros(NB,dtype=torch.int32,device="cuda"))
with torch.device("cuda"): carry=m.initial_carry(batch)
PP=np.zeros((SEG,NB,81),np.int8)
for si in range(SEG):
    carry,out=m(carry,batch); PP[si]=out["logits"].argmax(-1).cpu().numpy().astype(np.int8)
out=os.environ.get("LT_SAVE", os.path.join(SP, f"v11_traj_{ck['step']}.npz"))
np.savez_compressed(out,P=PP,X=X,Y=Y); print("saved",out)
