"""장 추적 — 측정 교리 v1.

근거(코드 검사 사실): noov 의 채널 간 상호작용은 전부 공유 스칼라 장
{a^(m)(r), d(r), 반경} 을 경유한다. 따라서 장을 기록하고 얼리면
조건부 동역학이 채널별로 정확히 선형·아핀이 된다:
    h(r+1) = s2(r) ⊙ [ s1(r)⊙h + dt·( Λ(s1⊙h) + T_{a(r)}[s1⊙h] + d(r)·b ) ]
(s = 기록된 Φ 반경 스케일. full/OV 판도 '장이 주어지면 선형'은 동일하게 성립 —
 다만 채널 결합이 장 밖(W_OV)에도 있어 채널별 분리는 noov 만의 성질.)

측정:
  A. 재생 검증  — 기록 장으로 재생한 궤적 = 실제 궤적 (기계 정밀도). 틀이 정확함의 증거
  B. 정확한 분해 — logits = W·L_장[h0] + (W·c_장 + β).  c_장 = h0=0 재생 (장이 나르는 정보),
                  L 부 = 전체 − c (수송되는 초기상태가 나르는 정보). 각각 단독 분류 정확도
  C. 장 이식    — 샘플 i 의 장 × 샘플 j 의 초기상태. 예측이 장 공여자를 따르나 상태 공여자를 따르나
                  → '계산이 라우팅(그래프)에 있나 화물(상태)에 있나'의 인과 판정
  D. 정착 곡선  — ‖a(r)−a(r−1)‖/‖a‖, d 장과 입력/정답의 상관 궤적

사전 등록 (MNIST noov): §10 "비선형성은 x→A 에 있다" ⟹ 장 이식에서 장 공여자가 이겨야 한다.
CA 는 등록 안 함 — 어느 쪽이든 정보.
"""
import sys
sys.path.insert(0, '/workspace/LinearTuring')
import numpy as np, torch
import interp, modes as M

DEV = interp.DEV
torch.set_grad_enabled(False)


def phi_s(m, h, tau):
    """Φ 의 스칼라 스케일 (기록해 두면 반스텝이 선형이 된다)."""
    return 1.0 / torch.sqrt(1.0 + 2.0 * m.gamma * h.pow(2).sum(-1, keepdim=True) * tau)


def _wo(m, h, r, steps):
    """경계 W_O — 고정 선형 사상이므로 조건부 선형성을 깨지 않는다 (forward 와 동일 조건)."""
    if m.boundary_wo and (r + 1) % m.R == 0 and (r + 1) < steps:
        import spectral_wo
        return spectral_wo.apply_wo(m, h)
    return h


def record(m, h0, steps):
    assert not m.freeze_A
    decay, Theta = m.kernel(); AB = m.W_C(); dt = 1.0 / m.R
    h = h0.clone(); F = {'a': [], 's1': [], 's2': []}; traj = [h0.clone()]
    for r in range(steps):
        s1 = phi_s(m, h, dt / 2); h = h * s1
        f, a, dtm, *_ = m.field(h, decay, Theta, None, AB)
        h = h + dt * f
        s2 = phi_s(m, h, dt / 2); h = h * s2
        h = _wo(m, h, r, steps)
        F['a'].append(a); F['s1'].append(s1); F['s2'].append(s2); traj.append(h.clone())
    return F, traj, (decay, Theta, AB, dt)


def replay(m, h0, F, steps, ctx):
    decay, Theta, AB, dt = ctx
    h = h0.clone()
    for r in range(steps):
        h = h * F['s1'][r]
        f, *_ = m.field(h, decay, Theta, F['a'][r], AB)
        h = h + dt * f
        h = h * F['s2'][r]
        h = _wo(m, h, r, steps)
    return h


def readout(m, h):
    lg = m.w_cls(h)
    return lg.mean(1) if m.pool else lg


def run(tag, ca=False, nb=96):
    if ca:
        m, cfg = M.load_ca(tag); xb, yb = M.ca_batch(cfg, nb)
    else:
        m, cfg = interp.load(tag)
        x, y = interp.testset(noisy=cfg.get("noisy", False))
        xb, yb = x[:nb].to(DEV), y[:nb].to(DEV)
    R = m.R
    h0 = m.embed_patches(xb)
    F, traj, ctx = record(m, h0, R)
    print(f"[{tag}] {'CA' if ca else 'MNIST'}  T={m.T} d={m.d} R={R}  nb={nb}")

    # --- A. 재생 검증 --------------------------------------------------
    hR = replay(m, h0, F, R, ctx)
    err = float((hR - traj[R]).norm() / traj[R].norm())
    print(f"  A. 재생 상대오차 = {err:.2e}  (조건부 선형성 {'확인' if err < 1e-4 else '실패!'})")

    # --- B. 정확한 분해 ------------------------------------------------
    lg_full = readout(m, traj[R])
    c_state = replay(m, torch.zeros_like(h0), F, R, ctx)
    lg_c = readout(m, c_state)                        # 장 항 (bias 포함)
    lg_L = lg_full - lg_c                             # 상태 수송 항 (bias 상쇄)
    pf, pc, pL = lg_full.argmax(-1), lg_c.argmax(-1), lg_L.argmax(-1)
    def acc(p):
        e = (p == yb).float()
        return float(e.mean())
    print(f"  B. 분해 정확도:  전체 {acc(pf):.4f} | 장 항(c)만 {acc(pc):.4f} | 상태 수송(L)만 {acc(pL):.4f}"
          + (f"   [우연 {1.0/lg_full.shape[-1]:.2f}]" if m.pool else "   [우연 0.50]"))
    # 장 항의 크기 비율
    print(f"     ‖c‖/‖h_R‖ = {float(c_state.norm()/traj[R].norm()):.3f}")

    # --- C. 장 이식 ----------------------------------------------------
    h0_j = torch.roll(h0, 1, dims=0); y_j = torch.roll(yb, 1, dims=0)
    h_x = replay(m, h0_j, F, R, ctx)                  # 장 = i, 상태 = j
    p_x = readout(m, h_x).argmax(-1)
    a_field, a_state = acc(p_x), float((p_x == y_j).float().mean())
    print(f"  C. 장 이식:  장 공여자 정답률 {a_field:.4f}  vs  상태 공여자 정답률 {a_state:.4f}"
          f"   →  {'라우팅(장)이 계산' if a_field > a_state else '화물(상태)이 계산'}")

    # --- D. 정착 곡선 + d 장 상관 --------------------------------------
    dts = [F['a'][r].sum(-1).sum(1) for r in range(R)]           # [B,T] 헤드 합 d_t
    if ca:
        ref_in = xb.float(); ref_out = yb.float(); nm_in, nm_out = "입력셀", "정답셀"
    else:
        G, P = m.grid, m.patch
        ref_in = xb.view(nb, 1, G, P, G, P).permute(0, 2, 4, 3, 5, 1).reshape(nb, G * G, P * P).mean(-1)
        ref_out = None; nm_in, nm_out = "밝기", None
    def corr(u, v):
        u = u.flatten() - u.mean(); v = v.flatten() - v.mean()
        return float((u * v).sum() / (u.norm() * v.norm() + 1e-9))
    rows = []
    for r in range(R):
        da = float((F['a'][r] - F['a'][r - 1]).norm() / F['a'][r].norm()) if r > 0 else float('nan')
        ci = corr(dts[r], ref_in)
        co = corr(dts[r], ref_out) if ref_out is not None else float('nan')
        rows.append((r, da, ci, co))
    hdr = f"     r    Δa/‖a‖   corr(d,{nm_in})" + (f"  corr(d,{nm_out})" if ref_out is not None else "")
    print("  D. 정착·상관 곡선"); print(hdr)
    for r, da, ci, co in rows:
        line = f"     {r}   {da:7.4f}   {ci:+9.3f}"
        if ref_out is not None: line += f"     {co:+9.3f}"
        print(line)
    print()


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["noov", "full", "ca110_k4_full"]):
        ca = t.startswith(("ca", "cap", "tau", "b110"))
        try:
            run(t, ca=ca)
        except Exception:
            import traceback; traceback.print_exc()
