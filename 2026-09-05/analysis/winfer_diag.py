"""진단: (a) η≈0 grad 경로 수치 재현, (b) τ=0, (c) 경사 상승(−η), (d) 같은 크기의 무작위 방향 섭동. N=256, 128세그, K=16."""
import os, sys, numpy as np, torch
os.environ.update(N="256", SEGS="128", K="16")
src = open("/workspace/LinearTuring/2026-09-05/analysis/winfer_exact.py").read().split('if __name__ == "__main__":')[0]
exec(src)
MODE = {"mode": "grad"}
_run = run
def run_mode(eta, tau, mode):
    """mode: grad(하강) | ascent(상승) | random(무작위 방향, 같은 RMS 크기)"""
    MODE["mode"] = mode; return _run(eta, tau)
# run() 안의 갱신 줄을 모드별로 바꾸기 위해 소스 패치 후 재정의
src2 = src.replace("                    u = u - eta * g.to(u.dtype)",
"""                    if MODE["mode"] == "grad": u = u - eta * g.to(u.dtype)
                    elif MODE["mode"] == "ascent": u = u + eta * g.to(u.dtype)
                    else:
                        rnd = torch.randn_like(g); rnd = rnd * (g.float().pow(2).mean().sqrt() / (rnd.pow(2).mean().sqrt() + 1e-12))
                        u = u - eta * rnd.to(u.dtype)""")
exec(src2); _run = run
print("기준선(첫 256퍼즐): seg16 145 / seg128 186", flush=True)
for label, eta, tau, mode in [("(a) η≈0 grad 경로", 1e-9, 0.3, "grad"), ("(b) τ=0 하강", 0.01, 0.0, "grad"), ("(c) τ=0.3 상승(−η)", 0.01, 0.3, "ascent"), ("(d) τ=0.3 무작위 방향", 0.01, 0.3, "random"), ("(e) τ=0.3 하강 재현", 0.01, 0.3, "grad")]:
    R = run_mode(eta, tau, mode)
    print(f"{label:>22}: seg16 {int(R['ex'][15]):3d}  seg64 {int(R['ex'][63]):3d}  seg128 {int(R['ex'][127]):3d}   소프트위반 {R['soft'][-1]:.5f}  argmax위반 {R['hard'][-1]:.2f}  확신도 {R['conf'][-1]:.3f}", flush=True)
