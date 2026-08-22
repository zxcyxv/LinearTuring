"""경계 W_O 를 포함한 블록 사상의 스펙트럼 반경 — 학습 후에도 수축이 유지되는가.
반복 단위 = R 스텝 + W_O.  ρ<1 이면 DEQ/IFT 적용 가능."""
import sys; sys.path.insert(0, '/workspace/LinearTuring')
import numpy as np, torch, modes as M
torch.set_grad_enabled(False)
DEV = M.DEV


def apply_wo(m, h):
    if not m.boundary_wo: return h
    if m.wo_mode == 'residual': return h + h @ m.w_bo.t()
    if m.wo_mode == 'orth':
        Q = torch.matrix_exp(m.w_bo_skew - m.w_bo_skew.t()); return h @ Q.t()
    if m.wo_mode == 'perhead':
        B_, T_, _ = h.shape
        return torch.einsum('bthc,hdc->bthd', h.view(B_, T_, m.H, m.dh), m.w_bo_blk).reshape(B_, T_, m.d)
    return h @ m.w_bo.t()


def block_with_wo(m, decay, Theta, AB, a_fix, dt):
    def GB(v):
        h = v.view(1, m.T, m.d)
        for _ in range(m.R):
            h = m.phi(h, dt / 2)
            f, *_ = m.field(h, decay, Theta, a_fix, AB)
            h = h + dt * f; h = m.phi(h, dt / 2)
        return apply_wo(m, h).reshape(-1)
    return GB


def run(tag, nsamp=2):
    m, cfg = M.load_ca(tag)
    tau = cfg.get('tau', 1)
    xb, _ = M.ca_batch(cfg, 8)
    decay, Theta = m.kernel(); AB = m.W_C(); dt = 1.0 / m.R
    h = m.embed_patches(xb)
    a_fix = m.attn(h, decay, Theta, AB)[0] if m.freeze_A else None
    for r in range(m.R * tau):
        h = m.phi(h, dt / 2)
        f, *_ = m.field(h, decay, Theta, a_fix, AB); h = h + dt * f; h = m.phi(h, dt / 2)
        if m.boundary_wo and (r + 1) % m.R == 0 and (r + 1) < m.R * tau: h = apply_wo(m, h)
    rho, smax, wo_rho = [], [], []
    for b in range(nsamp):
        af = a_fix[b:b+1] if a_fix is not None else None
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(
                block_with_wo(m, decay, Theta, AB, af, dt), h[b].reshape(-1).clone(),
                vectorize=True).detach()
        ev = torch.linalg.eigvals(J); s = torch.linalg.svdvals(J)
        rho.append(float(ev.abs().max())); smax.append(float(s[0]))
    if m.boundary_wo and m.wo_mode in ('plain', 'contract'):
        wo_rho = float(torch.linalg.eigvals(m.w_bo).abs().max())
    elif m.boundary_wo and m.wo_mode == 'orth':
        wo_rho = 1.0
    else:
        wo_rho = float('nan')
    return dict(tag=tag, noov=cfg.get('no_ov', False), mode=cfg.get('wo_mode', '—') if m.boundary_wo else '없음',
                rho=np.mean(rho), smax=np.mean(smax), wo_rho=wo_rho, d=m.d, T=m.T)


if __name__ == "__main__":
    print(f"{'태그':22s} {'구조':6s} {'W_O':9s} | {'ρ(W_O∘G^R)':>11s} {'σmax':>9s} | {'ρ(W_O)':>8s} | 판정")
    print("-" * 88)
    for t in sys.argv[1:]:
        try: r = run(t)
        except Exception as e:
            print(f"{t:22s}  실패: {type(e).__name__}: {e}"); continue
        v = "DEQ 가능" if r['rho'] < 1 else ("경계" if r['rho'] < 1.2 else "발산성")
        print(f"{r['tag']:22s} {'noov' if r['noov'] else 'full':6s} {r['mode']:9s} | "
              f"{r['rho']:11.4f} {r['smax']:9.3f} | {r['wo_rho']:8.4f} | {v}")
