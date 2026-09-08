"""1단계: 세 퍼즐에서 세그먼트 끝의 (a) 내용 위상 유사도 Re<ẑ_t,ẑ_n> (위치회전 제외), (b) 실제 커널 a 를 기록."""
import os, sys, importlib.util, time, numpy as np, torch
ROOT="/workspace/LinearTuring"; torch.set_grad_enabled(False)
PIDS=[115,295,460]; NB,SEG,BPS=3,128,8
s=importlib.util.spec_from_file_location("tk",os.path.join(ROOT,"2026-09-06/analysis/train_0901.py"))
tk=importlib.util.module_from_spec(s); s.loader.exec_module(tk)
sys.path.insert(0,os.path.join(ROOT,"checkpoints"))
from ckpt_npz import load as npz_load
cfg=dict(tk.CFG); cfg.update(batch_size=NB,seq_len=81,num_puzzle_identifiers=1,loops=SEG+1)
m=tk.LT(cfg).cuda().eval()
sd,_=npz_load(os.path.join(ROOT,"checkpoints/0901_R1B8_min_faith_step310527.npz"),which="ema")
PER={"wc_raw","psi","theta","alpha_raw","w_sh","eta_raw","lam_raw","gain_raw","beta","b_gate_up","b_down"}
def remap(k):
    k=k.replace("_orig_mod.",""); k=k[len("model."):] if k.startswith("model.") else k
    t=k[len("inner."):]
    if t=="inj_gate": return "inner.embed_scale"
    return f"inner.layers.0.{t}" if t.split(".")[0] in PER else k
m.load_state_dict({remap(k):v for k,v in sd.items()},strict=True); I=m.inner
z=np.load(os.path.join(ROOT,"kaggle/upload/sudoku_lt_1k.npz"))
X=z["test_inputs"].reshape(-1,81).astype(np.int32)[PIDS]; Y=z["test_labels"].reshape(-1,81).astype(np.int32)[PIDS]
x=torch.from_numpy(X+1).cuda(); y=torch.from_numpy(Y+1).cuda().long()
batch=dict(inputs=x,labels=y,puzzle_identifiers=torch.zeros(NB,dtype=torch.int32,device="cuda"))
H,T=8,81
CS=np.zeros((SEG,NB,H,T,T),np.float16)   # Re<ẑ_t,ẑ_n>  위치회전·ψ 없음 = 내용 위상 유사도
A =np.zeros((SEG,NB,H,T,T),np.float16)   # 실제 커널 a (감쇠·θ·ψ 포함)
PR=np.zeros((SEG,NB,T),np.int8)
ctr={"b":0}; orig=I.step
def hooked(L,h,AB,kc,w=None,fresh=None,kcb=None,vprev=None,apply_phi=True):
    b=ctr["b"]; si,bi=divmod(b,BPS); ctr["b"]+=1
    out=orig(L,h,AB,kc,w,fresh,kcb,vprev,apply_phi)
    if bi==BPS-1:
        hout=out[0]
        xx,yy=I.addr(hout,AB)                                   # ẑ [B,T,H,p]
        cs=(torch.einsum('bthj,bnhj->bhtn',xx,xx)+torch.einsum('bthj,bnhj->bhtn',yy,yy))
        CS[si]=cs.float().cpu().numpy().astype(np.float16)
        A[si]=I.attn_xy((xx,yy),kc).float().cpu().numpy().astype(np.float16)
        PR[si]=I.w_cls(hout).argmax(-1).cpu().numpy().astype(np.int8)
    return out
I.step=hooked
t0=time.time()
with torch.device("cuda"): carry=m.initial_carry(batch)
for si in range(SEG): carry,out=m(carry,batch)
np.savez_compressed("/tmp/claude-0/-workspace-LinearTuring/8092b54b-030b-4f2c-b7a7-8107892a6681/scratchpad/v1_phase.npz",
                    CS=CS,A=A,PR=PR,X=X,Y=Y,pids=np.array(PIDS))
print(f"{time.time()-t0:.0f}s  블록 {ctr['b']}")
