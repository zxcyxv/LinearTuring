"""3번 검증: (i) θ·Δ 항이 각 칸의 위상에 흡수되는가 — a_tn == D(Δ)·cos(φ'_t − φ'_n + ψ), φ' = φ + θ·pos
            (ii) 거리 감쇠 D(Δ)=e^{−α|Δ|₁} 는 학습 후 얼마나 남아 있는가 (α 헤드별, |Δ| 최대 16)"""
import os, importlib.util, math, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)
torch.set_grad_enabled(False)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=4, seq_len=81, num_puzzle_identifiers=1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).eval(); m.load_state_dict(sd, strict=True); I = m.inner
h = torch.randn(4, 81, 832)
for li, L in enumerate(I.layers):
    AB = I.W_C(L); kc = I.kernel(L)
    a_code = I.attn(h, AB, kc)                                            # [B,H,T,T] 코드 경로
    x, y = I.addr(h, AB)                                                  # ẑ = (x + iy) [B,T,H,p], |ẑ|=1 (헤드별 p채널 합)
    phi = torch.atan2(y, x)                                               # 채널별 위상 φ [B,T,H,p]
    r = torch.sqrt(x*x + y*y)                                             # 채널별 진폭 (헤드 안 정규화)
    ppos = (L.theta[..., 0, None] * I.pos_u + L.theta[..., 1, None] * I.pos_w).permute(2, 0, 1)   # [T,H,p] θ·pos_t
    phi_eff = phi + ppos.unsqueeze(0)                                     # φ' = φ + θ·pos  ← 위치가 위상에 흡수
    # a_tn = D(Δ)·Σ_j r_tj r_nj cos(φ'_tj − φ'_nj + ψ_j)
    d = phi_eff.unsqueeze(2) - phi_eff.unsqueeze(1) + L.psi.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [B,T,T,H,p]
    a_abs = (r.unsqueeze(2) * r.unsqueeze(1) * torch.cos(d)).sum(-1).permute(0, 3, 1, 2) * kc[0].unsqueeze(0)
    err = (a_code - a_abs).abs().max().item()
    alpha = L.alpha.flatten()
    print(f"layer{li}: |a_code − a_absorbed| max = {err:.2e}   (흡수 성립 여부)")
    print(f"   α 헤드별 = {[round(v,3) for v in alpha.tolist()]}")
    print(f"   D(Δ) at |Δ|=1: {[round(v,2) for v in torch.exp(-alpha*1).tolist()]}")
    print(f"   D(Δ) at |Δ|=8: {[round(v,2) for v in torch.exp(-alpha*8).tolist()]}")
    print(f"   D(Δ) at |Δ|=16:{[round(v,2) for v in torch.exp(-alpha*16).tolist()]}")
