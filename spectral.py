"""야코비안 스펙트럼 반경 — DEQ/IFT 적용 가능성 판정.

DEQ 는 h* = G(h*) 의 안정 고정점을 요구한다. 필요조건은 ρ(∂G/∂h) < 1.
두 가지 사상을 잰다:
  G   : 마이크로스텝 1회  (Strang: Φ ∘ (I+dt·f) ∘ Φ)
  G^R : τ=1 블록 전체 (R 스텝)
비정규계(야코비안 비대칭 0.575 실측)이므로 ρ 와 함께 σ_max 도 본다 —
ρ<1 은 점근 수축만 보장하고, σ_max>1 이면 과도 성장(transient growth)이 있다.
"""
import sys, math
sys.path.insert(0, '/workspace/LinearTuring')
import numpy as np, torch
import interp

DEV = interp.DEV


def step_map(m, decay, Theta, AB, a_fix, dt):
    def G(v):
        h = v.view(1, m.T, m.d)
        h = m.phi(h, dt / 2)
        f, *_ = m.field(h, decay, Theta, a_fix, AB)
        h = h + dt * f
        h = m.phi(h, dt / 2)
        return h.reshape(-1)
    return G


def block_map(m, decay, Theta, AB, a_fix, dt, R):
    def GB(v):
        h = v.view(1, m.T, m.d)
        for _ in range(R):
            h = m.phi(h, dt / 2)
            f, *_ = m.field(h, decay, Theta, a_fix, AB)
            h = h + dt * f
            h = m.phi(h, dt / 2)
        return h.reshape(-1)
    return GB


@torch.no_grad()
def evolve(m, xb, R):
    decay, Theta = m.kernel(); AB = m.W_C(); dt = 1.0 / m.R
    h = m.embed_patches(xb)
    a_fix = m.attn(h, decay, Theta, AB)[0] if m.freeze_A else None
    for _ in range(R):
        h = m.phi(h, dt / 2)
        f, *_ = m.field(h, decay, Theta, a_fix, AB)
        h = h + dt * f; h = m.phi(h, dt / 2)
    return h, decay, Theta, AB, a_fix, dt


def spec(J):
    ev = torch.linalg.eigvals(J)
    rho = float(ev.abs().max())
    s = torch.linalg.svdvals(J)
    return rho, float(s[0]), float(s.mean())


def analyse(tag, nsamp=2):
    m, cfg = interp.load(tag)
    x, _ = interp.testset(noisy=cfg.get("noisy", False))
    out = []
    for i in range(nsamp):
        xb = x[i:i+1].to(DEV)
        h, decay, Theta, AB, a_fix, dt = evolve(m, xb, m.R)   # 끌개 위에서
        v0 = h.reshape(-1).clone()
        af = a_fix[0:1] if a_fix is not None else None
        with torch.enable_grad():
            J1 = torch.autograd.functional.jacobian(
                step_map(m, decay, Theta, AB, af, dt), v0, vectorize=True).detach()
            JB = torch.autograd.functional.jacobian(
                block_map(m, decay, Theta, AB, af, dt, m.R), v0, vectorize=True).detach()
        out.append((spec(J1), spec(JB)))
    r1 = np.array([[*a, *b] for a, b in out]).mean(0)
    return dict(tag=tag, R=m.R, dim=m.T*m.d,
                rho1=r1[0], s1=r1[1], rhoB=r1[3], sB=r1[4])


if __name__ == "__main__":
    tags = sys.argv[1:] or ["full"]
    print(f"{'모델':10s} {'차원':>6s} | {'ρ(G)':>8s} {'σmax(G)':>9s} | {'ρ(G^R)':>10s} {'σmax(G^R)':>11s} | 판정")
    print("-" * 82)
    for t in tags:
        try:
            r = analyse(t)
        except Exception as e:
            print(f"{t:10s}  실패: {type(e).__name__}: {e}"); continue
        # DEQ 가능 = 블록 사상이 수축
        verdict = "DEQ 가능" if r['rhoB'] < 1.0 else ("경계" if r['rhoB'] < 1.2 else "발산성")
        print(f"{r['tag']:10s} {r['dim']:6d} | {r['rho1']:8.4f} {r['s1']:9.4f} | "
              f"{r['rhoB']:10.4f} {r['sB']:11.4f} | {verdict}")
