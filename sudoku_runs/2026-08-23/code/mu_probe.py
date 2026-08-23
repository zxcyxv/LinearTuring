"""μ 마찰항 그래디언트 프로브 (DAMPING.md §4 — 학습 아님, 측정만).

가설: τ 확장 실패의 원인은 e^{λτ} 그래디언트 잡음화이고, 방향 감쇠 μ 가
스펙트럼을 λ→λ−μ 로 평행이동시켜 이를 회복시킨다.

측정: 동일 가중치에서 (스텝 수 8 vs 16) × (μ ∈ {0, .3, .7}) 격자로
  ① 세그먼트 그래디언트 노름 ‖g‖
  ② 반배치 코사인 cos(g_A, g_B)  — 방향 SNR 대리지표
μ 구현은 DAMPING §2 의 적분인자: 매 스텝 끝에 h ← e^{−μ·dt}·h (연산자 분할 1항 추가).

예측 (사전 등록): 스텝 16 에서 μ=0 의 코사인이 스텝 8 대비 급락하고,
μ>0 에서 회복되면 마찰 가설 확정. 회복이 없으면 원인은 다른 데 있다.
"""
import argparse, json, math
import torch, torch.nn.functional as F
from model1 import Model1
from ca_task import batch


def rollout(m, x, steps, R, mu):
    """model1.forward 와 동일하되 μ 적분인자만 추가."""
    dt = 1.0 / R
    damp = math.exp(-mu * dt)
    h = m.embed_patches(x)
    decay, Theta = m.kernel()
    AB = m.W_C()
    a_fix = m.attn(h, decay, Theta, AB)[0] if m.freeze_A else None
    for r in range(steps):
        h = m.phi(h, dt / 2)
        f, *_ = m.field(h, decay, Theta, a_fix, AB)
        h = h + dt * f
        h = m.phi(h, dt / 2)
        if mu > 0.0:
            h = h * damp
    logits = m.w_cls(h)
    return logits if not m.pool else logits.mean(1)


def grad_vec(m, x, y, steps, R, mu):
    m.zero_grad(set_to_none=True)
    loss = F.cross_entropy(rollout(m, x, steps, R, mu).reshape(-1, 2), y.reshape(-1))
    loss.backward()
    g = torch.cat([p.grad.reshape(-1) for p in m.parameters() if p.grad is not None])
    m.zero_grad(set_to_none=True)
    return loss.item(), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ca110_k8_full.pt")
    ap.add_argument("--steps_list", default="8,16")
    ap.add_argument("--mu_list", default="0,0.3,0.7")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--reps", type=int, default=8, help="독립 배치 반복 — 코사인 평균/표준편차용")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", action="store_true", help="학습된 가중치 대신 무작위 초기화 (대조)")
    ap.add_argument("--out", default="runs/mu_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    T, R, rule, k = ca["T"], ca["R"], ca["rule"], ca["k"]
    pos = torch.stack([torch.arange(T).float(), torch.zeros(T)], 1)
    m = Model1(d=ca["d"], H=ca["H"], R=R, n_classes=2, positions=pos, vocab=2,
               freeze_A=ca["freeze_A"], pool=False,
               use_ov=not ca.get("no_ov", False), sheaf=ca.get("sheaf", False),
               boundary_wo=ca.get("boundary_wo", False), wo_mode=ca.get("wo_mode", "plain"),
               lam_mode="full").to(dev)
    if not a.init:
        # pos_u/pos_w 는 positions 에서 결정적으로 계산되는 버퍼 (구 체크포인트엔 없음)
        r = m.load_state_dict(ck["model"], strict=False)
        derived = {"pos_u", "pos_w"}
        assert not r.unexpected_keys, f"예상 밖 키: {r.unexpected_keys}"
        assert set(r.missing_keys) <= derived, f"가중치 누락: {set(r.missing_keys) - derived}"
        if r.missing_keys:
            print(f"(파생 버퍼 {sorted(r.missing_keys)} 는 positions 에서 재계산 — 정상)")
    m.train()

    steps_list = [int(s) for s in a.steps_list.split(",")]
    mu_list = [float(s) for s in a.mu_list.split(",")]
    rows = []
    print(f"ckpt={a.ckpt} {'(무작위 init 대조)' if a.init else ''}  rule={rule} k={k} T={T} R={R} d={ca['d']}")
    print(f"{'steps':>6} {'tau':>5} {'mu':>5} {'loss':>8} {'|g|':>11} {'cos(gA,gB)':>18}")
    for steps in steps_list:
        for mu in mu_list:
            gen = torch.Generator(device=dev).manual_seed(a.seed)   # 셀마다 동일 데이터
            cos_s, gn_s, ls_s = [], [], []
            for _ in range(a.reps):
                x, y = batch(a.bs, T, k, rule, dev, gen)
                half = a.bs // 2
                l1, g1 = grad_vec(m, x[:half], y[:half], steps, R, mu)
                l2, g2 = grad_vec(m, x[half:], y[half:], steps, R, mu)
                cos_s.append(F.cosine_similarity(g1, g2, dim=0).item())
                gn_s.append(((g1 + g2) / 2).norm().item())
                ls_s.append((l1 + l2) / 2)
            cm = sum(cos_s) / len(cos_s)
            cs = (sum((c - cm) ** 2 for c in cos_s) / len(cos_s)) ** 0.5
            gm = sum(gn_s) / len(gn_s)
            lm = sum(ls_s) / len(ls_s)
            rows.append(dict(steps=steps, tau=steps / R, mu=mu, loss=lm, gnorm=gm,
                             cos_mean=cm, cos_std=cs, cos_all=cos_s))
            print(f"{steps:>6} {steps/R:>5.1f} {mu:>5.2f} {lm:>8.4f} {gm:>11.4e} "
                  f"{cm:>10.4f} ± {cs:.4f}")

    # 요약: μ=0 에서 스텝 배증의 SNR 손실, 그리고 μ 가 그것을 얼마나 회복시키는가
    def get(s, mu): return next(r for r in rows if r["steps"] == s and r["mu"] == mu)
    s0, s1 = steps_list[0], steps_list[-1]
    base = get(s1, 0.0)["cos_mean"] / get(s0, 0.0)["cos_mean"]
    print(f"\nμ=0: steps {s0}→{s1} 코사인 비 {base:.3f}")
    for mu in mu_list[1:]:
        r = get(s1, mu)["cos_mean"] / get(s1, 0.0)["cos_mean"]
        print(f"  μ={mu}: steps {s1} 코사인 회복 배율 {r:.3f} "
              f"({get(s1,0.0)['cos_mean']:.4f} → {get(s1,mu)['cos_mean']:.4f})")
    json.dump({"args": vars(a), "ckpt_args": ca, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
