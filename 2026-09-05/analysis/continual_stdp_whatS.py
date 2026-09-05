"""공유 시냅스 S 가 무엇을 담았는지: 위치 커널 a 의 스트림 평균과의 코사인, 동료/비동료 분해, 샘플 간 분산."""
import os, numpy as np, torch
src = open("/workspace/LinearTuring/2026-09-05/analysis/continual_stdp.py").read().split("res = {")[0]
os.environ["SEGS"] = "16"; exec(src)
# 64 퍼즐 2청크만: 블록마다 a 와 Γ 를 모아 평균/분산
Sa = [torch.zeros(I.H,81,81,device="cuda") for _ in layers]; SG = [torch.zeros(I.H,81,81,device="cuda") for _ in layers]; SG2 = [torch.zeros(I.H,81,81,device="cuda") for _ in layers]; cnt = 0
for b in range(0, 128, CH):
    x = torch.from_numpy(X[b:b+CH].astype(np.int32)+1).cuda(); y = torch.from_numpy(Y[b:b+CH].astype(np.int32)+1).cuda().long(); nb = len(x)
    batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(nb, dtype=torch.int32, device="cuda"))
    inj = I.injection(batch)*I.embed_scale; ABk = [(I.W_C(L), I.kernel(L)) for L in layers]
    h = I.init_hidden.expand(nb,81,-1).clone(); w=None; fresh=torch.ones(nb,dtype=torch.bool,device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(16):
            for _ in range(I.config.blocks_per_seg):
                for li, L in enumerate(layers):
                    AB, kc = ABk[li]; xy = I.addr(h+inj, AB); a = I.attn_xy(xy, kc).float()
                    hout, w, tgt = step_gamma(L, h+inj, AB, kc, w, fresh, None); h = I.phi(I.boundary(L, hout)); fresh=None
                    Sa[li] += a.mean(0); SG[li] += tgt.float().mean(0); SG2[li] += (tgt.float()**2).mean(0)
            cnt += 1
peer = torch.as_tensor(peer_f).cuda().bool(); eye = torch.eye(81, device="cuda").bool()
for li in range(2):
    A = Sa[li]/cnt; G = SG[li]/cnt; V = SG2[li]/cnt - G**2
    cos = torch.nn.functional.cosine_similarity(A.flatten(), G.flatten(), 0).item()
    print(f"L{li}: cos(⟨Γ⟩, ⟨a⟩) = {cos:.3f}   ⟨Γ⟩ rms: 동료 {G[:, peer].pow(2).mean().sqrt():.4f}  비동료 {G[:, ~peer & ~eye].pow(2).mean().sqrt():.4f}  자기 {G[:, eye].pow(2).mean().sqrt():.4f}")
    print(f"     블록 간 표준편차/|평균| (동료) = {(V[:, peer].sqrt()/G[:, peer].abs().clamp_min(1e-6)).median():.2f}   ⟨Γ⟩ 부호: 동료 양 {(G[:, peer]>0).float().mean():.2f}")
