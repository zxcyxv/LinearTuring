"""v1 의 embed_scale 을 상수로 바꿔 스윕. es=28 이면 주입 크기가 URM/v4 규약과 일치."""
import os, sys, importlib.util, json, time, numpy as np, torch
from dataclasses import replace
ROOT="/workspace/LinearTuring"; torch.set_grad_enabled(False)
NB, SEG = 512, 512
s=importlib.util.spec_from_file_location("tk", os.path.join(ROOT,"2026-09-06/analysis/train_0901.py"))
tk=importlib.util.module_from_spec(s); s.loader.exec_module(tk)
sys.path.insert(0, os.path.join(ROOT,"checkpoints")); from ckpt_npz import load as npz_load
cfg=dict(tk.CFG); cfg.update(batch_size=NB, seq_len=81, num_puzzle_identifiers=1, loops=SEG+1)
sd,_=npz_load(os.path.join(ROOT,"checkpoints/0901_R1B8_min_faith_step310527.npz"), which="ema")
PER={"wc_raw","psi","theta","alpha_raw","w_sh","eta_raw","lam_raw","gain_raw","beta","b_gate_up","b_down"}
def remap(k):
    k=k.replace("_orig_mod.",""); k=k[len("model."):] if k.startswith("model.") else k
    t=k[len("inner."):]
    return "inner.embed_scale" if t=="inj_gate" else (f"inner.layers.0.{t}" if t.split(".")[0] in PER else k)
SD={remap(k):v for k,v in sd.items()}
z=np.load(os.path.join(ROOT,"kaggle/upload/sudoku_lt_1k.npz"))
X=z["test_inputs"].reshape(-1,81).astype(np.int32)[:NB]; Y=z["test_labels"].reshape(-1,81).astype(np.int32)[:NB]
x=torch.from_numpy(X+1).cuda(); y=torch.from_numpy(Y+1).cuda().long()
batch=dict(inputs=x,labels=y,puzzle_identifiers=torch.zeros(NB,dtype=torch.int32,device="cuda"))
clue=torch.from_numpy(X>0).cuda()

def run(es_new, S0):
    m=tk.LT(cfg).cuda().eval(); m.load_state_dict(SD, strict=True)
    shuffle = es_new < 0
    if shuffle:
        es_new = -es_new
        g=torch.Generator().manual_seed(0)
        perm=torch.randperm(m.inner.embed.weight.shape[0], generator=g)
        m.inner.embed.weight.data = m.inner.embed.weight.data[perm].contiguous()
    es0=float(m.inner.embed_scale)
    with torch.device("cuda"): carry=m.initial_carry(batch)
    out_ex={}; out_cell={}; out_ck={}; out_hn={}; P=None
    for si in range(SEG):
        m.inner.embed_scale.data.fill_(es_new if si>=S0 else es0)
        carry,out=m(carry,batch)
        if si+1 in (16,64,128,192,256,384,512):
            p=out["logits"].argmax(-1)
            out_ex[si+1]=int((p==y).all(-1).sum()); out_cell[si+1]=float((p==y).float().mean())
            out_ck[si+1]=float(((p==y)|~clue).float().mean()) if False else float((p[clue]==y[clue]).float().mean())
            out_hn[si+1]=float(carry.current_hidden.norm(dim=-1).mean())
        if si+1==SEG: P=out["logits"].argmax(-1).cpu().numpy()
    return dict(es=es_new, S0=S0, shuffle=bool(shuffle), es0=es0, exact=out_ex, cell=out_cell, clue_keep=out_ck, hnorm=out_hn), P

CFGS=[(7.602,0),(28,16),(20,16),(28,32),(-28,16)]
res=[]; Pbase=None
for es_new,S0 in CFGS:
    t0=time.time(); r,P=run(es_new,S0)
    if Pbase is None: Pbase=P; okb=(P==(Y+1)).all(1)
    ok=(P==(Y+1)).all(1)
    r["fixed"]=int((ok & ~okb).sum()); r["broken"]=int((~ok & okb).sum()); r["net"]=r["fixed"]-r["broken"]
    r["sec"]=round(time.time()-t0)
    res.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(res, open("/tmp/claude-0/-workspace-LinearTuring/8092b54b-030b-4f2c-b7a7-8107892a6681/scratchpad/es512.json","w"), ensure_ascii=False, indent=1)
