"""Model 1 을 MNIST 로 학습. 사양의 '최소 실험 사양' 표를 기본값으로 둔다."""
import argparse, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from model1 import Model1

def get_loaders(bs, root="/workspace/data", noisy=False, workers=4):
    tf = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if noisy:   # §3 경고(a): MNIST가 너무 쉬우면 동역학이 지루해진다
        tf.append(transforms.Lambda(lambda t: t + 0.6 * torch.randn_like(t)))
    tf = transforms.Compose(tf)
    tr = datasets.MNIST(root, True, download=True, transform=tf)
    te = datasets.MNIST(root, False, download=True, transform=tf)
    return (DataLoader(tr, bs, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True),
            DataLoader(te, 512, shuffle=False, num_workers=workers, pin_memory=True))

@torch.no_grad()
def evaluate(model, loader, dev, R=None):
    model.eval(); c = n = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        c += (model(x, R=R).argmax(1) == y).sum().item(); n += y.numel()
    return c / n

@torch.no_grad()
def diagnostics(model, x):
    """§'로깅' 항목: ψ 히스토그램, max_t||h_t^(r)||, ρ^(r), d_t 공간지도"""
    model.eval()
    _, tr = model(x, record=True)
    hn = [h.norm(dim=-1).max().item() for h in tr["h"]]          # max_t ||h_t^(r)||
    T = model.T
    P = torch.eye(T, device=x.device) - 1.0 / T
    rho = []
    for a in tr["a"]:                                            # [B,H,T,T]
        M = P @ a[:8].mean(0) @ P                                # 배치 평균 후 중심화
        rho.append(torch.linalg.matrix_norm(M, ord=2).mean().item())
    dstats = [(d.mean().item(), d.std().item()) for d in tr["d"]]
    return {"h_max": hn, "rho": rho, "d_mean_std": dstats,
            "psi": model.psi.detach().cpu().flatten().tolist(),
            "alpha": model.alpha.detach().cpu().flatten().tolist(),
            "gamma": model.gamma.item(), "kappa": model.kappa(),
            "absorb_R": model.absorbing_radius()[0]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--no_ov", action="store_true", help="W_OV 제거 (순수 좌곱셈 판정)")
    ap.add_argument("--lam_mode", default="full", choices=["full", "diag", "none"])
    ap.add_argument("--no_orth", action="store_true")
    ap.add_argument("--no_bias_v", action="store_true")
    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--freeze_A", action="store_true", help="A 를 h^(0) 에서 고정 → h 에 대해 선형")
    ap.add_argument("--fix_gamma", action="store_true", help="γ 를 학습하지 않고 고정")
    ap.add_argument("--psi_zero", action="store_true", help="ψ≡0 고정: 대칭 함정 강제 (§2.4 검증)")
    ap.add_argument("--kernel_lr_mult", type=float, default=1.0, help="ψ,θ,α 전용 학습률 배수")
    ap.add_argument("--rand_R", type=str, default="", help="예: 4,16 — 매 배치 R 을 균등 샘플 (τ=1 고정)")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tr_loader, te_loader = get_loaders(args.bs, noisy=args.noisy)
    model = Model1(d=args.d, H=args.H, R=args.R, gamma_init=args.gamma, alpha_init=args.alpha,
                   eps=args.eps, use_ov=not args.no_ov, lam_mode=args.lam_mode,
                   orth_wc=not args.no_orth, use_bias_v=not args.no_bias_v,
                   psi_zero=args.psi_zero, learn_gamma=not args.fix_gamma,
                   freeze_A=args.freeze_A).to(dev)
    kern_names = {"psi", "theta", "alpha_raw"}
    kern = [p for n, p in model.named_parameters() if n in kern_names]
    rest = [p for n, p in model.named_parameters() if n not in kern_names]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr},
                             {"params": kern, "lr": args.lr * args.kernel_lr_mult}],
                            lr=args.lr, weight_decay=1e-4)
    rR = [int(v) for v in args.rand_R.split(",")] if args.rand_R else None
    steps = args.epochs * len(tr_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr, args.lr*args.kernel_lr_mult],
                                               total_steps=steps, pct_start=0.15)
    probe_x = next(iter(te_loader))[0][:64].to(dev)

    net = torch.compile(model) if args.compile else model
    log = {"args": vars(args), "steps": [], "epochs": [], "n_params": sum(p.numel() for p in model.parameters())}
    print(f"[{args.tag}] params={log['n_params']}  dev={dev}  steps={steps}")
    gstep = 0; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            Rb = int(torch.randint(rR[0], rR[1] + 1, (1,)).item()) if rR else None
            logits = net(x, R=Rb)      # R=None → 학습 기본값. rand_R 이면 매 배치 dt=1/Rb, steps=Rb (τ=1)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step(); sched.step(); gstep += 1
            if gstep % args.log_every == 0:
                d = diagnostics(model, probe_x); model.train()
                acc = (logits.argmax(1) == y).float().mean().item()
                log["steps"].append({"step": gstep, "loss": loss.item(), "acc": acc, "gn": gn, **d})
                print(f"  step {gstep:5d} loss {loss.item():.4f} acc {acc:.3f} "
                      f"|h|max {d['h_max'][-1]:.2f} rho {d['rho'][-1]:.2f} "
                      f"gam {d['gamma']:.3f} kap {d['kappa']:.1f}", flush=True)
        te = evaluate(model, te_loader, dev)
        log["epochs"].append({"epoch": ep, "test_acc": te, "time": time.time() - t0})
        print(f"[{args.tag}] epoch {ep} test_acc {te:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "runs"); os.makedirs(RUNS, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(RUNS, f"{args.tag}.pt"))
    json.dump(log, open(os.path.join(RUNS, f"{args.tag}_log.json"), "w"))
    print(f"[{args.tag}] done. final test_acc={log['epochs'][-1]['test_acc']:.4f}")

if __name__ == "__main__":
    main()
