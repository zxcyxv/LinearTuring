"""다중 홉 연상 회상 (n-hop induction head) — 인컨텍스트 러닝 축.

시퀀스마다 심볼집합 S(|S|=N)와 그 위의 랜덤 N-순환 π 를 새로 뽑는다.
쌍을 무작위 순서로 나열하고 질의를 붙인다:
    s_{σ1} π(s_{σ1})  …  s_{σN} π(s_{σN})   q_1 … q_Q          T = 2N + Q
질의 위치에서 π^n(q_i) 를 출력해야 한다.

n=1 이 정확히 표준 induction head 다. n≥2 가 판별기:
  n=1 : q↔s_j 는 토큰 동일성(임베딩 수준), π(s_j) 이송은 위치 커널 → A 고정으로도 가능
  n≥2 : π(q) 로 다시 매칭해야 하는데 π(q) 는 임베딩이 아니라 *상태* 에 있다
        → A 를 상태에서 다시 그려야만 함. "x→A 1회 사상이 사주는 홉 수" 를 직접 센다.

π 가 시퀀스마다 재추출되므로 가중치에 저장 불가 — 순수 in-context.
목표가 문맥 안의 토큰이라 양방향이어도 누설이 아니다(예측이 아니라 검색).
"""
import argparse, json, time
import torch, torch.nn.functional as F
from model1 import Model1


def batch(bs, V, N, Q, n, dev, gen):
    """returns x:[bs,T] int64, y:[bs,Q] int64 (질의 위치의 목표).  토큰 1..V, 0 은 미사용."""
    # 시퀀스별로 서로 다른 심볼 N개 (순환 순서 = 뽑힌 순서)
    cyc = torch.rand(bs, V, device=dev, generator=gen).argsort(-1)[:, :N] + 1     # [bs,N] c_0..c_{N-1}
    # π(c_i) = c_{i+1 mod N}
    nxt = torch.roll(cyc, -1, dims=1)                                             # [bs,N]
    # 쌍의 제시 순서를 섞는다
    order = torch.rand(bs, N, device=dev, generator=gen).argsort(-1)              # [bs,N]
    keys = cyc.gather(1, order); vals = nxt.gather(1, order)                      # [bs,N]
    pairs = torch.stack([keys, vals], -1).reshape(bs, 2 * N)                      # k v k v ...
    # 질의: 순환 인덱스에서 뽑고 목표는 n홉 뒤
    qi = torch.randint(0, N, (bs, Q), device=dev, generator=gen)                  # [bs,Q]
    q = cyc.gather(1, qi)
    y = cyc.gather(1, (qi + n) % N)
    x = torch.cat([pairs, q], 1)                                                  # [bs, 2N+Q]
    return x, y


def evaluate(m, a, dev, gen, nb=4096):
    with torch.no_grad():
        x, y = batch(nb, a.V, a.N, a.Q, a.n, dev, gen)
        p = m(x, R=a.R, steps=a.R * a.tau)[:, -a.Q:].argmax(-1)
        per = (p == y).float().mean().item()
        exact = (p == y).all(-1).float().mean().item()
    return per, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--V", type=int, default=64, help="심볼 어휘 크기")
    ap.add_argument("--N", type=int, default=8, help="제시되는 쌍 개수 (= 순환 길이)")
    ap.add_argument("--Q", type=int, default=4, help="질의 개수")
    ap.add_argument("--n", type=int, default=1, help="홉 수 — 필요한 순차 라운드")
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--H", type=int, default=4); ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--freeze_A", action="store_true")
    ap.add_argument("--no_ov", action="store_true", help="noov 구조 (순수 좌곱셈)")
    ap.add_argument("--boundary_wo", action="store_true")
    ap.add_argument("--wo_mode", default="plain",
                    choices=["plain", "residual", "orth", "contract", "perhead"])
    ap.add_argument("--tau", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernel_lr_mult", type=float, default=30.0)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
    T = 2 * a.N + a.Q
    pos = torch.stack([torch.arange(T).float(), torch.zeros(T)], 1)
    m = Model1(d=a.d, H=a.H, R=a.R, n_classes=a.V + 1, positions=pos, vocab=a.V + 1,
               freeze_A=a.freeze_A, pool=False, boundary_wo=a.boundary_wo, wo_mode=a.wo_mode,
               use_ov=not a.no_ov, lam_mode="diag" if a.no_ov else "full").to(dev)
    nparam = sum(p.numel() for p in m.parameters())
    kn = {"psi", "theta", "alpha_raw"}
    kp = [p for nm, p in m.named_parameters() if nm in kn]
    rp = [p for nm, p in m.named_parameters() if nm not in kn]
    opt = torch.optim.AdamW([{"params": rp, "lr": a.lr},
                             {"params": kp, "lr": a.lr * a.kernel_lr_mult}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.lr * a.kernel_lr_mult],
                                              total_steps=a.steps, pct_start=0.15)
    net = torch.compile(m) if a.compile else m
    print(f"[{a.tag}] T={T} V={a.V} N={a.N} Q={a.Q} n={a.n} tau={a.tau} freeze_A={a.freeze_A} "
          f"noov={a.no_ov} d={a.d} params={nparam} 우연={1/a.N:.4f}", flush=True)

    log = []; t0 = time.time()
    for i in range(a.steps):
        x, y = batch(a.bs, a.V, a.N, a.Q, a.n, dev, gen)
        logits = net(x, R=a.R, steps=a.R * a.tau)[:, -a.Q:]              # 질의 위치만
        loss = F.cross_entropy(logits.reshape(-1, a.V + 1), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if (i + 1) % 250 == 0:
            per, ex = evaluate(m, a, dev, gen, 2048)
            log.append({"step": i + 1, "loss": loss.item(), "query": per, "exact": ex})
            print(f"  [{a.tag}] step {i+1:5d} loss {loss.item():.4f} 질의 {per:.4f} 전체일치 {ex:.4f}", flush=True)
    per, ex = evaluate(m, a, dev, gen, 8192)
    print(f"[{a.tag}] n={a.n} N={a.N} T={T} tau={a.tau} freeze_A={a.freeze_A} noov={a.no_ov} "
          f"FINAL 질의 {per:.4f} 전체일치 {ex:.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": m.state_dict(), "args": vars(a)}, f"runs/{a.tag}.pt")
    json.dump({"args": vars(a), "log": log, "query": per, "exact": ex, "params": nparam},
              open(f"runs/{a.tag}_hop.json", "w"))


if __name__ == "__main__":
    main()
