"""진폭 틀 v2 — 기저 상태를 '끌개 위의 실제 궤적점'으로 옮긴다.

v1 은 균일 고정점에서 선형화했는데, CA 처럼 입력이 균일 상태에서 먼 과제에서는
궤적이 그 근방을 아예 안 지나가 무의미했다 (불안정 모드 0/2048).
여기서는 샘플별로 r=R 의 실제 상태에서 선형화하고, 그 샘플의 궤적을 그 기저로 분해한다.
샘플마다 기저가 다르므로 평균 대신 샘플별 통계를 보고한다.
"""
import sys, math
sys.path.insert(0, '/workspace/LinearTuring')
import numpy as np, torch
import interp, dispersion as D, modes as M
from model1 import Model1

DEV = interp.DEV
SCR = '/tmp/claude-0/-workspace-LinearTuring/ac978981-cfca-4f33-817b-3ad8423e9e35/scratchpad'
torch.set_grad_enabled(False)


def jac_at(m, hstate, decay, Theta, AB, a_fix, dt):
    """주어진 상태 [T,d] 에서 마이크로스텝 사상의 야코비안."""
    def G(v):
        h = v.view(1, m.T, m.d)
        h = m.phi(h, dt / 2)
        f, *_ = m.field(h, decay, Theta, a_fix, AB)
        h = h + dt * f
        return m.phi(h, dt / 2).reshape(-1)
    with torch.enable_grad():
        return torch.autograd.functional.jacobian(G, hstate.reshape(-1).clone(),
                                                  vectorize=True).detach()


def run(tag, ca=False, nsamp=6, topk=12):
    if ca:
        m, cfg = M.load_ca(tag); xb, yb = M.ca_batch(cfg, 32)
        tau = cfg.get('tau', 1)
    else:
        m, cfg = interp.load(tag)
        x, y = interp.testset(noisy=cfg.get("noisy", False))
        xb, yb = x[:32].to(DEV), y[:32].to(DEV); tau = 1
    Rread = m.R * tau
    steps = Rread * 2
    traj, decay, Theta, AB, a_fix, dt = M.trajectory(m, xb, steps)   # [steps+1,B,T,d]
    Wc = m.w_cls.weight; Cn = Wc.shape[0]
    print(f"[{tag}] {'CA' if ca else 'MNIST'}  T={m.T} d={m.d} R={m.R} tau={tau}  "
          f"차원={m.T*m.d}  기저=끌개(r={Rread})")

    stats = []
    for b in range(nsamp):
        af = a_fix[b:b+1] if a_fix is not None else None
        J = jac_at(m, traj[Rread, b], decay, Theta, AB, af, dt)
        ev, V = torch.linalg.eig(J)
        # 이산 사상이므로 성장 판정은 |ev| > 1
        gr = ev.abs()
        order = gr.argsort(descending=True)
        ev, V, gr = ev[order], V[:, order], gr[order]
        Vinv = torch.linalg.inv(V)
        nunst = int((gr > 1).sum())

        Vt = V.reshape(m.T, m.d, -1)
        Wcc = torch.complex(Wc, torch.zeros_like(Wc))
        ptc = torch.einsum('cd,tdj->tcj', Wcc, Vt)
        disc = (ptc.mean(0).abs().std(0) if m.pool else ptc.abs().std(1).mean(0)).real

        Xb = torch.complex((traj[:, b] - traj[Rread, b]).reshape(steps + 1, -1),
                           torch.zeros(1, device=DEV))
        A = torch.einsum('ij,rj->ri', Vinv, Xb).abs()          # [steps+1, N]

        tg = set(range(topk)); td = set(disc.argsort(descending=True)[:topk].tolist())
        ov = len(tg & td)
        # 성장률 상위 20% 안에 판별 상위 topk 가 몇 개나 있나 (완화된 겹침)
        n20 = len(gr) // 5
        ov20 = len(set(range(n20)) & td)
        # 판별 가중 평균 성장률 vs 전체 평균
        w = disc / disc.sum()
        gw = float((w * gr).sum()); gm = float(gr.mean())
        stats.append((nunst, ov, ov20, gw, gm, float(gr[0])))
        print(f"  샘플{b}: 불안정(|λ|>1) {nunst:5d}/{len(gr)}  max|λ|={float(gr[0]):.3f} | "
              f"겹침 top{topk} {ov}  top20% {ov20}/{topk} | "
              f"판별가중 |λ| {gw:.4f} vs 전체평균 {gm:.4f}")

    S = np.array(stats)
    print(f"  ── 평균: 불안정 {S[:,0].mean():.0f}  겹침 {S[:,1].mean():.1f}  "
          f"top20% 겹침 {S[:,2].mean():.1f}/{topk}  "
          f"판별가중|λ| {S[:,3].mean():.4f} vs 평균 {S[:,4].mean():.4f}  "
          f"→ {'판별 모드가 더 잘 자람' if S[:,3].mean() > S[:,4].mean() else '판별 모드가 더 빨리 죽음'}")
    return S


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["noov"]):
        ca = t.startswith(("ca", "cap", "tau", "b110"))
        try:
            run(t, ca=ca)
        except Exception as e:
            import traceback; traceback.print_exc()
        print("=" * 78)
