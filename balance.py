"""부호 그래프로서의 A: 구조적 균형과 그 분할의 의미.
   L̄ = D̄ − A_sym,  D̄ = diag(Σ_n|a_tn|)  는 부호에 무관하게 PSD.
   λ_min(L̄) ≈ 0  ⟺  균형(balanced) ⟺ 두 집단으로 나뉘고 흐름이 그 이분 상태로 수렴.
   λ_min 을 평균 차수로 정규화한 것이 좌절도(frustration).
   비교군: 같은 |a| 를 유지한 채 부호만 무작위로 섞은 그래프."""
import sys, json; sys.path.insert(0,'.')
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import koreanize_matplotlib
import matplotlib.pyplot as plt
import interp
plt.rcParams.update({"figure.dpi":130,"font.size":8,"axes.titlesize":8,"axes.unicode_minus":False})

@torch.no_grad()
def analyse(tag, nb=64, at_r=None):
    m,cfg = interp.load(tag); x,y = interp.testset(noisy=cfg["noisy"])
    xb = x[:nb].to(interp.DEV); decay,Theta = m.kernel(); h = m.embed_patches(xb); dt=1/m.R
    at_r = m.R if at_r is None else at_r
    for r in range(at_r):
        h = m.phi(h, dt/2); f,*_ = m.field(h, decay, Theta); h = h + dt*f; h = m.phi(h, dt/2)
    a,*_ = m.attn(h, decay, Theta)                      # [B,H,T,T]
    A = a.mean(1)                                       # 헤드 평균 (전체 혼합 그래프)
    As = 0.5*(A + A.transpose(-1,-2))
    D  = As.abs().sum(-1)
    Di = (D+1e-9).pow(-0.5)
    L  = torch.eye(As.shape[-1],device=As.device) - Di.unsqueeze(-1)*As*Di.unsqueeze(-2)  # 정규화 부호 라플라시안
    ev, evec = torch.linalg.eigh(L)
    frus = ev[:,0].cpu().numpy()                        # 정규화하면 λ_min 자체가 좌절도 (0~1)
    part = evec[:,:,0]
    pr = (1.0/part.pow(4).sum(-1)).cpu().numpy()        # 참여비: 1이면 한 노드 국소화, T면 전역
    dhet = (D.std(-1)/D.mean(-1)).cpu().numpy()         # 차수 이질성
    # 부호만 섞은 비교군
    g = torch.Generator(device=A.device).manual_seed(0)
    sgn = (torch.randint(0,2,As.shape,generator=g,device=A.device)*2-1).float()
    sgn = torch.tril(sgn,-1); sgn = sgn + sgn.transpose(-1,-2) + torch.eye(As.shape[-1],device=A.device)
    Ar = As.abs()*sgn
    Dr = Ar.abs().sum(-1); Dri=(Dr+1e-9).pow(-0.5)
    Lr = torch.eye(As.shape[-1],device=As.device) - Dri.unsqueeze(-1)*Ar*Dri.unsqueeze(-2)
    evr = torch.linalg.eigvalsh(Lr)
    frus_r = evr[:,0].cpu().numpy()
    # 분할이 획/배경과 맞는가 — 입력 패치 밝기와의 상관
    G,P = m.grid, m.patch
    bright = xb.view(nb,1,G,P,G,P).permute(0,2,4,3,5,1).reshape(nb,G*G,P*P).mean(-1)
    cc=[]
    for b in range(nb):
        p_ = part[b].cpu().numpy(); q_ = bright[b].cpu().numpy()
        if p_.std()>1e-9 and q_.std()>1e-9: cc.append(abs(np.corrcoef(p_,q_)[0,1]))
    negfrac = (A<0).float().mean().item()
    return dict(tag=tag, frus=float(frus.mean()), frus_std=float(frus.std()),
                frus_rand=float(frus_r.mean()), negfrac=negfrac,
                corr_bright=float(np.mean(cc)), corr_std=float(np.std(cc)),
                pr=float(pr.mean()), T=int(As.shape[-1]), dhet=float(dhet.mean()),
                part=part[:4].cpu().numpy(), bright=bright[:4].cpu().numpy(), G=G,
                lam=[float(v) for v in ev[0,:6].cpu().numpy()])

tags = sys.argv[1:] or ["noov"]
res=[analyse(t) for t in tags]
fig,ax=plt.subplots(2,len(tags)*2,figsize=(3.0*len(tags)*2,4.4),squeeze=False)
for i,r in enumerate(res):
    print(f"[{r['tag']}] 음수 간선 {r['negfrac']:.3f}")
    print(f"   정규화 좌절도 λ_min/⟨d⟩ = {r['frus']:.4f} ± {r['frus_std']:.4f}   (부호 무작위 비교군 {r['frus_rand']:.4f})")
    print(f"   → 비교군 대비 {r['frus_rand']/max(r['frus'],1e-9):.2f}배 {'낮음(균형쪽)' if r['frus']<r['frus_rand'] else '높음'}")
    print(f"   참여비 PR = {r['pr']:.1f} / T={r['T']}  ({'한 노드 국소화 — 균형 아님' if r['pr']<3 else '퍼진 분할'}),  차수 이질성 {r['dhet']:.3f}")
    print(f"   분할 고유벡터 vs 패치 밝기 |상관| = {r['corr_bright']:.3f} ± {r['corr_std']:.3f}")
    print(f"   L̄ 최소 고유값 6개: " + " ".join(f"{v:.3f}" for v in r['lam']))
    for j in range(2):
        ax[j][2*i].imshow(r['part'][j].reshape(r['G'],r['G']),cmap='RdBu_r'); ax[j][2*i].set_title(f"{r['tag']} 분할 #{j}",fontsize=7)
        ax[j][2*i+1].imshow(r['bright'][j].reshape(r['G'],r['G']),cmap='gray'); ax[j][2*i+1].set_title("패치 밝기",fontsize=7)
        for k in range(2): ax[j][2*i+k].set_xticks([]); ax[j][2*i+k].set_yticks([])
fig.suptitle("부호 그래프의 구조적 균형 — 흐름이 수렴하는 '합의'는 균일이 아니라 분할",y=1.02)
fig.tight_layout(); fig.savefig("figs/balance.png",bbox_inches="tight")
json.dump([{k:v for k,v in r.items() if k not in ('part','bright')} for r in res], open("runs/balance.json","w"))
