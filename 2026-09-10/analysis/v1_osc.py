"""v1(9/1@310527): 칸 단위 위상 진동 세기 vs 칸 난이도의 상관.
진동 세기 = 1 − 원형 집중도.  (a) 칸 자체 위상 e^{iφ_{t,j}(k)} 의 블록 평균 크기,
                              (b) 피어 쌍 위상차 ⟨ẑ_t,ẑ_n⟩ 의 블록 평균 크기 / 평균 절댓값.
두 지평: 학습 지평(블록 1..128 = seg16) 과 전체(블록 1..1024 = seg128).
난이도 = (D1) 제약전파 솔버 라운드, (D2) 모델 커밋 시각(마지막으로 예측이 바뀐 seg), (D3) 최종 정오."""
import os, sys, importlib.util, time, numpy as np, torch
ROOT="/workspace/LinearTuring"; SP="/tmp/claude-0/-workspace-LinearTuring/8092b54b-030b-4f2c-b7a7-8107892a6681/scratchpad"
torch.set_grad_enabled(False)
NB,SEG,BPS=512,128,8; KTRAIN=16*BPS
s=importlib.util.spec_from_file_location("tk",os.path.join(ROOT,"2026-09-06/analysis/train_0901.py"))
tk=importlib.util.module_from_spec(s); s.loader.exec_module(tk)
sys.path.insert(0,os.path.join(ROOT,"checkpoints")); from ckpt_npz import load as npz_load
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
X=z["test_inputs"].reshape(-1,81).astype(np.int32)[:NB]; Y=z["test_labels"].reshape(-1,81).astype(np.int32)[:NB]
x=torch.from_numpy(X+1).cuda(); y=torch.from_numpy(Y+1).cuda().long()
batch=dict(inputs=x,labels=y,puzzle_identifiers=torch.zeros(NB,dtype=torch.int32,device="cuda"))
H,T=8,81
dev="cuda"
SUMU=torch.zeros(NB,T,H,52,dtype=torch.complex64,device=dev)     # Σ_k e^{iφ}
SUMR2=torch.zeros(NB,T,H,52,device=dev)                            # Σ_k r̂²
SUMC=torch.zeros(NB,H,T,T,dtype=torch.complex64,device=dev)       # Σ_k ⟨ẑ_t,ẑ_n⟩_C
SUMA=torch.zeros(NB,H,T,T,device=dev)                              # Σ_k |⟨ẑ_t,ẑ_n⟩_C|
snap={}
ctr={"b":0}; orig=I.step
def hooked(L,h,AB,kc,w=None,fresh=None,kcb=None,vprev=None,apply_phi=True):
    b=ctr["b"]; ctr["b"]+=1
    xx,yy=I.addr(h,AB)                                             # ẑ 실/허 [B,T,H,p]
    zc=torch.complex(xx.float(),yy.float())
    mag=zc.abs()
    SUMU.add_(zc/(mag+1e-9)); SUMR2.add_(mag*mag)
    C=torch.einsum('bthj,bnhj->bhtn',zc.conj(),zc)                # Σ_j conj(ẑ_t)ẑ_n
    SUMC.add_(C); SUMA.add_(C.abs())
    if b+1==KTRAIN:
        snap.update(U=SUMU.clone(),R2=SUMR2.clone(),C=SUMC.clone(),A=SUMA.clone())
    return orig(L,h,AB,kc,w,fresh,kcb,vprev,apply_phi)
I.step=hooked
t0=time.time()
with torch.device(dev): carry=m.initial_carry(batch)
P=np.zeros((SEG,NB,T),np.int8)
for si in range(SEG):
    carry,out=m(carry,batch); P[si]=out["logits"].argmax(-1).cpu().numpy().astype(np.int8)
print(f"forward {time.time()-t0:.0f}s  블록 {ctr['b']}",flush=True)

def cellstats(U,R2,C,A,K):
    Rc=(U.abs()/K)                                                 # [B,T,H,p] 칸 자체 집중도
    r_unw=Rc.mean((2,3))                                           # [B,T]
    wgt=R2/R2.sum(3,keepdim=True).clamp_min(1e-9)
    r_w=(Rc*wgt).sum(3).mean(2)                                    # r̂² 가중
    Rp=C.abs()/A.clamp_min(1e-9)                                   # [B,H,T,T] 쌍 집중도
    return r_unw.cpu().numpy(), r_w.cpu().numpy(), Rp.mean(1).cpu().numpy()   # 쌍은 헤드 평균 [B,T,T]
ru16,rw16,rp16=cellstats(snap["U"],snap["R2"],snap["C"],snap["A"],KTRAIN)
ruF,rwF,rpF=cellstats(SUMU,SUMR2,SUMC,SUMA,SEG*BPS)
np.savez_compressed(os.path.join(SP,"v1_osc.npz"),ru16=ru16,rw16=rw16,rp16=rp16,ruF=ruF,rwF=rwF,rpF=rpF,P=P,X=X,Y=Y)
print("저장 완료",flush=True)
