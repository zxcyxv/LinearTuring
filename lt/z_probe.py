"""v1.7 흔적 z 와 재주입의 결합: 순전파 노름·DC 지배도·기울기 노름."""
import os, sys, importlib.util, numpy as np, torch, torch.nn.functional as F
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT,"lt"))
from ckpt_npz import load_lt, load_data
P=os.environ.get("LT_CKPT", os.path.join(ROOT,"checkpoints/v17_step_380000.npz"))
NB,SEG=128,32
m,cfg,step=load_lt(P, batch_size=NB, loops=SEG+1); ck={"step":step}; I=m.inner
X,Y,batch=load_data(ROOT,NB)
L=I.layers[0]; rho=torch.sigmoid(L.mu_rho_raw); om=L.mu_omega
print(f"# {os.path.basename(P)} step={ck['step']}  ρ: min {rho.min():.3f} med {rho.median():.3f} max {rho.max():.3f}   ω: |ω| med {om.abs().median():.2f}")
# 상수 입력 이론 증폭  |z*|/|u| = √(1−ρ²)/|1−ρe^{iω}|
amp=torch.sqrt(1-rho**2)/torch.sqrt((1-rho*torch.cos(om))**2+(rho*torch.sin(om))**2)
print(f"이론(상수 입력) |z*|/|u| 성분별: med {amp.median():.2f} max {amp.max():.2f}   (max 성분 ρ={rho.flatten()[amp.argmax()]:.3f}, ω={om.flatten()[amp.argmax()]:.2f})")
# 재주입 주소
with torch.no_grad(), torch.device("cuda"):
    inj=I.injection(batch); h_inj=I.embed_scale*inj
    AB=I.W_C(L)
rec={"zn":[],"un":[],"hn":[],"cos_inj":[],"cos_u_inj":[],"gin":None}
orig=I.trace_step
def hooked(L_,ux,uy,ztr,fresh):
    zx,zy,ztr_new=orig(L_,ux,uy,ztr,fresh)
    with torch.no_grad():
        zn=torch.sqrt((zx**2+zy**2).sum(-1)); un=torch.sqrt((ux**2+uy**2).sum(-1))       # [B,T,H]
        # 재주입 주소와의 정렬 (복소 코사인)
        ix,iy=I.addr_raw(h_inj,AB)
        num=torch.sqrt((zx*ix+zy*iy).sum(-1)**2+(zx*iy-zy*ix).sum(-1)**2)
        den=zn*torch.sqrt((ix**2+iy**2).sum(-1))+1e-9
        rec["zn"].append(zn.float().cpu()); rec["un"].append(un.float().cpu()); rec["cos_inj"].append((num/den).float().cpu())
        numu=torch.sqrt((ux*ix+uy*iy).sum(-1)**2+(ux*iy-uy*ix).sum(-1)**2); denu=un*torch.sqrt((ix**2+iy**2).sum(-1))+1e-9
        rec["cos_u_inj"].append((numu/denu).float().cpu())
    return zx,zy,ztr_new
I.trace_step=hooked
origstep=I.step
def hstep(L_,h,AB_,kc,w=None,fresh=None,kcb=None,ztr=None,apply_phi=True):
    rec["hn"].append(h.norm(dim=-1).float().detach().cpu().mean().item()); return origstep(L_,h,AB_,kc,w,fresh,kcb,ztr,apply_phi)
I.step=hstep
I.eval()
with torch.no_grad(), torch.device("cuda"), torch.autocast("cuda",dtype=torch.bfloat16):
    carry=m.initial_carry(batch)
    for si in range(SEG): carry,out=m(carry,batch)
zn=torch.stack(rec["zn"]); un=torch.stack(rec["un"]); ci=torch.stack(rec["cos_inj"])      # [K,B,T,H]
r=zn/(un+1e-9); K=zn.shape[0]
print(f"\n[순전파, {K} 블록]  ‖h‖ 평균: 블록1 {rec['hn'][0]:.1f}  블록8 {rec['hn'][7]:.1f}  블록{K} {rec['hn'][-1]:.1f}   (√d = {np.sqrt(832):.1f})")
print("블록      :  " + "  ".join(f"{k:>5d}" for k in [1,2,4,8,16,32,64,128,256] if k<=K))
print("‖z‖/‖u‖ 평균: " + "  ".join(f"{r[k-1].mean():5.2f}" for k in [1,2,4,8,16,32,64,128,256] if k<=K))
print("‖z‖/‖u‖ 최대: " + "  ".join(f"{r[k-1].max():5.2f}" for k in [1,2,4,8,16,32,64,128,256] if k<=K))
print("‖z‖ 최소     : " + "  ".join(f"{zn[k-1].min():5.2f}" for k in [1,2,4,8,16,32,64,128,256] if k<=K) + f"    (ε = {I.config.eps})")
print("cos(z,재주입) : " + "  ".join(f"{ci[k-1].mean():5.2f}" for k in [1,2,4,8,16,32,64,128,256] if k<=K))
cu=torch.stack(rec["cos_u_inj"])
print("cos(u,재주입) : " + "  ".join(f"{cu[k-1].mean():5.2f}" for k in [1,2,4,8,16,32,64,128,256] if k<=K))
print(f"‖z‖ 전 구간 최소 {zn.min():.3f}, 하위 0.1% 분위 {torch.quantile(zn.flatten()[:2_000_000],0.001):.3f}   ‖u‖ 전 구간 최소 {un.min():.3f}")
clue=torch.from_numpy(X!=0)
print(f"마지막 블록  cos(z,재주입): 단서 {ci[-1][clue].mean():.3f} 빈칸 {ci[-1][~clue].mean():.3f}   cos(u,재주입): 단서 {cu[-1][clue].mean():.3f} 빈칸 {cu[-1][~clue].mean():.3f}")
# ---- 기울기: 8 세그 무기울기 워밍업 후 1 세그 역전파, 군별 노름 ----
I.trace_step=orig; I.step=origstep; m.train()
for p in m.parameters(): p.requires_grad_(True)
groups={"주소 W_C":["wc_raw"],"흔적 ρ,ω":["mu_rho_raw","mu_omega"],"값 W_sh":["w_sh"],"창 β,ψ,θ,α":["beta","psi","theta","alpha_raw"],"가소성 η,λ,g":["eta_raw","lam_raw","gain_raw"],"경계":["b_gate_up","b_down"],"임베딩·분류":["embed","w_cls"]}
tot=[]; gn={k:[] for k in groups}
for rep,warm in enumerate([1,4,8,16]):
    b=batch
    with torch.device("cuda"):
        carry=m.initial_carry(b)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            for si in range(warm): carry,_=m(carry,b)
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            carry,out=m(carry,b); logits=out["logits"].float()
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]), b["labels"].reshape(-1)); loss.backward()
    g2=0.0
    for n,p in m.named_parameters():
        if p.grad is None: continue
        v=p.grad.float().norm().item()**2; g2+=v
        for k,keys in groups.items():
            if any(kk in n for kk in keys): gn[k].append(v**0.5)
    tot.append(g2**0.5)
print(f"\n[기울기 노름: 워밍업 1/4/8/16 세그 뒤 1 세그 역전파, 배치 128]  전체: "+"  ".join(f"{t:.3f}" for t in tot))
for k in groups: 
    if gn[k]: print(f"   {k:12s}: "+"  ".join(f"{v:.4f}" for v in gn[k]))
