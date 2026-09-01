"""체크포인트마다 헤드별 λ=sigmoid(lam_raw), η=sigmoid(eta_raw) 를 찍는다.
   TRAIN_PLAN.md 의 사전 등록 항목: λ 가 0.25 근처에 머무는가 / 1 로 가는가 / 0 으로 가는가.
사용: python 2026-08-31/analysis/track_lam.py [체크포인트 디렉터리]"""
import os, re, sys, glob, torch
d = sys.argv[1] if len(sys.argv) > 1 else "/root/LinearTuring/refs/URM/checkpoints/R1B8_min_lam"
fs = sorted(glob.glob(os.path.join(d, "step_*.pt")), key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
if not fs: sys.exit(f"체크포인트 없음: {d}")
print(f"{'step':>8} | {'λ평균':>6} {'λ최소':>6} {'λ최대':>6} | {'δ평균':>6} | {'G평균':>6} {'G최대':>6} | {'|β|':>6} | λ 헤드별")
for f in fs:
    sd = torch.load(f, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    g = lambda n: [v for k, v in sd.items() if k.endswith(n)]
    lr_, er_ = g("lam_raw"), g("eta_raw")
    if not lr_: continue
    lam = torch.sigmoid(lr_[0].float()).flatten(); eta = torch.sigmoid(er_[0].float()).flatten()
    gn_ = g("gain_raw"); bt_ = g("beta")
    gn = torch.nn.functional.softplus(gn_[0].float()).flatten() if gn_ else torch.zeros(1)
    bt = bt_[0].float().abs().mean() if bt_ else torch.tensor(0.)
    st = int(re.search(r"step_(\d+)", f).group(1))
    print(f"{st:>8} | {lam.mean():>6.3f} {lam.min():>6.3f} {lam.max():>6.3f} | {eta.mean():>6.3f} | "
          f"{gn.mean():>6.3f} {gn.max():>6.3f} | {bt:>6.3f} | " + " ".join(f"{v:.2f}" for v in lam))
