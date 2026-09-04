"""훈련 속도 시뮬레이션 — 측정한 ms/step 에서 실제 벽시계를 뽑는다.

학습 회계 (HANDOVER.md · pretrain.py):
  1 step  = 배치 128 하나에 대한 세그먼트 1회 fwd+bwd+opt
  1 epoch = 퍼즐 1,000개 × 1샘플 = 1,000 예제 = 7.8125 step
  1 iter  = 250 epoch = 1,953 step,  계획 200 iter = 390,600 step
  eval    = 1,953 step 마다. held-out 2,048 × loops 16 세그먼트, no_grad
"""
import argparse, json

STEP_PER_ITER = 1953
PLAN_STEPS = 390_600


def sim(ms_step, ms_eval_fwd, done, target, eval_every=STEP_PER_ITER, n_eval=2048,
        loops=16, bs=128, ckpt_s=3.0):
    left = max(0, target - done)
    evals = left // eval_every
    train_s = left * ms_step / 1e3
    eval_s = evals * (n_eval / bs) * loops * ms_eval_fwd / 1e3
    ckpt = evals * ckpt_s
    tot = train_s + eval_s + ckpt
    return dict(left_steps=left, evals=int(evals),
                train_h=round(train_s / 3600, 2), eval_h=round(eval_s / 3600, 2),
                ckpt_h=round(ckpt / 3600, 2), total_h=round(tot / 3600, 2),
                total_d=round(tot / 86400, 2), it_s=round(1000 / ms_step, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--done", type=int, default=310_527, help="이미 돈 step")
    ap.add_argument("--target", type=int, default=PLAN_STEPS)
    ap.add_argument("--json", default=None, help="벤치 결과 JSON 목록 파일 (한 줄에 하나)")
    ap.add_argument("--eval-fwd-ms", type=float, default=None,
                    help="eval 1 세그먼트 forward(no_grad) ms. 미지정이면 step 의 30%%")
    a = ap.parse_args()

    rows = []
    if a.json:
        for line in open(a.json):
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("--json 으로 bench_train.py 출력을 넘겨라")

    print(f"# 남은 학습: {a.done:,} → {a.target:,} step\n")
    hdr = f"{'설정':<26}{'ms/step':>9}{'it/s':>7}{'GB':>7}{'학습h':>8}{'eval h':>8}{'합 h':>8}{'일':>7}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ev = a.eval_fwd_ms if a.eval_fwd_ms else r["ms_median"] * 0.30
        s = sim(r["ms_median"], ev, a.done, a.target)
        tag = f"{r['tag']}"
        print(f"{tag:<26}{r['ms_median']:>9.1f}{s['it_s']:>7.2f}{r['peak_GB']:>7.2f}"
              f"{s['train_h']:>8.1f}{s['eval_h']:>8.1f}{s['total_h']:>8.1f}{s['total_d']:>7.2f}")
    print(f"\n(eval 은 {a.eval_fwd_ms or '측정 step ×0.30'} ms/세그먼트 · 2,048×16 세그먼트 · "
          f"{PLAN_STEPS//STEP_PER_ITER - a.done//STEP_PER_ITER}회 가정)")


if __name__ == "__main__":
    main()
