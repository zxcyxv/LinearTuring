"""진폭 방정식 틀 — 무엇을 측정해야 하는가에 대한 답.

1단계(분산관계)는 dispersion.py 에서 끝났다. 여기는 2~5단계:
  2. 모드 진폭 A_j(r)  : 어느 불안정 모드가 실제로 자라는가
  3. 읽기 투영 <u_c, v_j> : 어느 모드가 어느 클래스에 투표하는가 = 판단 기준
  4. 간선별 힘 기여     : 척력/인력의 귀속
  5. commitment depth   : 결정이 언제 비가역이 되는가

비정규계이므로 고유벡터가 직교하지 않는다. 진폭은 오른쪽 고유벡터 기저의
좌표, 즉 A = V^{-1} x 로 잡아야 한다 (수반 기저 = 왼쪽 고유벡터).
"""
import sys, math
sys.path.insert(0, '/workspace/LinearTuring')
import numpy as np, torch
import interp, dispersion as D
from model1 import Model1

DEV = interp.DEV
SCR = '/tmp/claude-0/-workspace-LinearTuring/ac978981-cfca-4f33-817b-3ad8423e9e35/scratchpad'
torch.set_grad_enabled(False)


def load_ca(tag):
    """ca_task.py 로 학습된 체크포인트 로드 (pool=False, 1D 위치, vocab=2)."""
    ck = torch.load(f'runs/{tag}.pt', map_location=DEV, weights_only=False)
    a = ck['args']
    pos = torch.stack([torch.arange(a['T']).float(), torch.zeros(a['T'])], 1)
    lam = a.get('lam', 'auto')
    lam_mode = ("diag" if (a.get('no_ov', False) or a.get('sheaf', False)) else "full") if lam == 'auto' else lam
    m = Model1(d=a['d'], H=a['H'], R=a['R'], n_classes=2, positions=pos, vocab=2,
               freeze_A=a.get('freeze_A', False), pool=False,
               boundary_wo=a.get('boundary_wo', False), wo_mode=a.get('wo_mode', 'plain'),
               sheaf=a.get('sheaf', False),
               use_ov=not a.get('no_ov', False), lam_mode=lam_mode).to(DEV)
    m.load_state_dict(ck['model']); m.eval()
    return m, a


def ca_batch(a, nb=64, seed=1):
    import ca_task
    g = torch.Generator(device=DEV).manual_seed(seed)
    return ca_task.batch(nb, a['T'], a['k'], a['rule'], DEV, g)


def ca_hdir(m, a, nb=128):
    x, _ = ca_batch(a, nb)
    decay, Theta = m.kernel(); AB = m.W_C(); dt = 1.0 / m.R
    h = m.embed_patches(x)
    a_fix = m.attn(h, decay, Theta, AB)[0] if m.freeze_A else None
    for _ in range(m.R * a.get('tau', 1)):
        h = m.phi(h, dt / 2)
        f, *_ = m.field(h, decay, Theta, a_fix, AB)
        h = h + dt * f; h = m.phi(h, dt / 2)
    return h.mean((0, 1)), float(h.norm(dim=-1).mean())


def trajectory(m, xb, steps):
    """H^(r) 기록. returns [steps+1, B, T, d]"""
    decay, Theta = m.kernel(); AB = m.W_C(); dt = 1.0 / m.R
    h = m.embed_patches(xb)
    a_fix = m.attn(h, decay, Theta, AB)[0] if m.freeze_A else None
    traj = [h.clone()]
    for _ in range(steps):
        h = m.phi(h, dt / 2)
        f, *_ = m.field(h, decay, Theta, a_fix, AB)
        h = h + dt * f; h = m.phi(h, dt / 2)
        traj.append(h.clone())
    return torch.stack(traj), decay, Theta, AB, a_fix, dt


def main(tag, nb=64, topk=12, ca=False):
    if ca:
        m, cfg = load_ca(tag)
        xb, yb = ca_batch(cfg, nb)
        hdir, rad = ca_hdir(m, cfg)
    else:
        m, cfg = interp.load(tag)
        x, y = interp.testset(noisy=cfg.get("noisy", False))
        xb, yb = x[:nb].to(DEV), y[:nb].to(DEV)
        hdir, rad, _ = D.get_hdir(m, cfg)
    hbar, res, _ = D.uniform_fixed_point(m, hdir, rad)
    tau = cfg.get('tau', 1) if ca else 1
    print(f"[{tag}] 균일 고정점 ‖h̄‖={float(hbar.norm()):.3f} (잔차 {res:.1e})")

    with torch.enable_grad():
        J = D.full_jacobian(m, hbar)
    ev, V = torch.linalg.eig(J)
    order = ev.real.argsort(descending=True)
    ev, V = ev[order], V[:, order]
    Vinv = torch.linalg.inv(V)
    nunst = int((ev.real > 0).sum())
    print(f"  불안정 모드 {nunst}/{len(ev)}   최대 Re λ = {float(ev.real[0]):+.4f}")

    # --- 3단계: 읽기 투영 --------------------------------------------
    # pool=True 이면 logit_c = (1/T) Σ_t W_cls[c]·h_t
    Wc = m.w_cls.weight                                    # [C,d]
    C = Wc.shape[0]
    Vt = V.reshape(m.T, m.d, -1)                           # [T,d,N]
    Wcc = torch.complex(Wc, torch.zeros_like(Wc))
    ptc = torch.einsum('cd,tdj->tcj', Wcc, Vt)             # [T,C,N] 토큰 t 에서 모드 j 의 클래스 c 기여
    if m.pool:
        proj = ptc.mean(0)                                 # [C,N]
        disc = proj.abs().std(0).real
    else:
        proj = ptc.mean(0)                                 # 보고용
        disc = ptc.abs().std(1).real.mean(0)               # 토큰별 클래스 분산의 평균

    # --- 2단계: 모드 진폭 추적 ---------------------------------------
    steps = m.R * tau * 3 if ca else m.R * 3
    Rread = m.R * tau if ca else m.R
    traj, decay, Theta, AB, a_fix, dt = trajectory(m, xb, steps)
    base = hbar.view(1, 1, -1)
    Xc = torch.complex((traj - base).reshape(steps + 1, nb, -1), torch.zeros(1, device=DEV))
    A = torch.einsum('ij,rbj->rbi', Vinv, Xc)              # [steps+1, B, N] 진폭
    Amag = A.abs().mean(1)                                 # 샘플 평균 [steps+1, N]

    # 성장률 상위 모드 + 판별력 상위 모드
    top_grow = list(range(topk))
    top_disc = disc.argsort(descending=True)[:topk].tolist()

    print(f"\n  [2·3단계] 상위 성장 모드 — 진폭 A_j(r) 과 클래스 판별력")
    print(f"  {'j':>3s} {'Reλ':>8s} {'Imλ':>8s} | {'A(0)':>9s} {'A(R)':>9s} {'A(3R)':>9s} | {'판별력':>8s}")
    for j in top_grow:
        print(f"  {j:3d} {float(ev.real[j]):+8.3f} {float(ev.imag[j].abs()):8.3f} | "
              f"{float(Amag[0,j]):9.3e} {float(Amag[Rread,j]):9.3e} {float(Amag[-1,j]):9.3e} | "
              f"{float(disc[j]):8.3e}")

    print(f"\n  [3단계] 판별력 상위 모드 (읽기와 정렬된 것)")
    print(f"  {'j':>3s} {'Reλ':>8s} | {'A(R)':>9s} | {'판별력':>9s} | 성장순위")
    for j in top_disc:
        rank = int((disc > disc[j]).sum())
        print(f"  {j:3d} {float(ev.real[j]):+8.3f} | {float(Amag[Rread,j]):9.3e} | "
              f"{float(disc[j]):9.3e} | 성장 {j}위 / 판별 {rank}위")

    # 성장 상위와 판별 상위가 겹치는가
    ov = len(set(top_grow) & set(top_disc))
    print(f"\n  성장 상위{topk} ∩ 판별 상위{topk} = {ov}개")

    # --- 5단계: commitment depth -------------------------------------
    if m.pool:
        logits = torch.stack([m.w_cls(traj[r]).mean(1) for r in range(steps + 1)])
    else:
        logits = torch.stack([m.w_cls(traj[r]) for r in range(steps + 1)])       # [steps+1,B,T,C]
    pred = logits.argmax(-1)
    final = pred[Rread]
    commit = torch.zeros(nb, device=DEV, dtype=torch.long)
    for b in range(nb):
        cd = 0
        for r in range(Rread + 1):
            same = (pred[r, b] == final[b])
            if not (same.all() if same.dim() else same): cd = r + 1
        commit[b] = cd
    acc = (final == yb).float().mean().item()
    print(f"\n  [5단계] commitment depth (Rread={Rread} 중)  평균 {commit.float().mean():.2f}  "
          f"중앙값 {int(commit.median())}  최종정확도 {acc:.3f}")
    hist = torch.bincount(commit, minlength=Rread + 1).cpu().numpy()
    print("   분포 r=0..R:", hist.tolist())
    correct = (final == yb) if final.dim() == 1 else (final == yb).all(-1)
    print(f"   맞은 샘플 평균 {commit[correct].float().mean():.2f}  "
          f"틀린 샘플 평균 {commit[~correct].float().mean() if (~correct).any() else float('nan'):.2f}")

    torch.save(dict(ev=ev.cpu(), Amag=Amag.cpu(), disc=disc.cpu(), proj=proj.cpu(),
                    commit=commit.cpu(), R=m.R, tag=tag), f'{SCR}/modes_{tag}.pt')
    return dict(tag=tag, nunst=nunst, overlap=ov, commit=float(commit.float().mean()), acc=acc)


if __name__ == "__main__":
    args = sys.argv[1:] or ["noov"]
    for t in args:
        ca = t.startswith("ca") or t.startswith("cap") or t.startswith("tau") or t.startswith("b110")
        try:
            main(t, ca=ca)
        except Exception as e:
            import traceback; traceback.print_exc()
        print("=" * 78)
