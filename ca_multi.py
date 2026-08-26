"""다중 규칙 셀룰러 오토마타 — 규칙 id 임베딩(모든 칸 가산 = 스도쿠 puzzle_emb 방식)으로 한 모델이 여러 규칙을 실행하는가.
   Wolfram 클래스별 학습 가능성 표 + "가산 벡터 = 분기 매개변수" 검증 (ARC 전 단계). ca_task.py 재사용."""
import argparse, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model1 import Model1
from ca_task import step

CLASS = {0:1, 4:2, 184:2, 232:2, 30:3, 45:3, 90:3, 150:3, 54:4, 110:4}

class MultiRuleModel(Model1):
    def __init__(self, n_rules, **kw):
        super().__init__(**kw)
        self.rule_emb = nn.Embedding(n_rules, self.d)          # 과제 벡터 (전 칸 브로드캐스트 가산)
        nn.init.normal_(self.rule_emb.weight, std=0.5)
    def embed_patches(self, x):
        tok, rid = x
        return self.embed(tok) + self.rule_emb(rid)[:, None, :]

def batch(bs, T, k, rules, dev, gen):
    x = torch.randint(0, 2, (bs, T), device=dev, generator=gen)
    rid = torch.randint(0, len(rules), (bs,), device=dev, generator=gen)
    y = x.clone()
    for i, r in enumerate(rules):
        m = rid == i
        if m.any():
            yy = x[m]
            for _ in range(k): yy = step(yy, r)
            y[m] = yy
    return x, rid, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--rules", default="4,184,232,30,45,90,54,110")
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--d", type=int, default=64); ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--R", type=int, default=8); ap.add_argument("--tau", type=int, default=4)
    ap.add_argument("--sheaf", action="store_true"); ap.add_argument("--lam", default="full")
    ap.add_argument("--wo_mode", default="contract"); ap.add_argument("--no_wo", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--kernel_lr_mult", type=float, default=30.0)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args(); rules = [int(r) for r in a.rules.split(",")]
    dev = "cuda"; torch.manual_seed(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
    pos = torch.stack([torch.arange(a.T).float(), torch.zeros(a.T)], 1)
    m = MultiRuleModel(len(rules), d=a.d, H=a.H, R=a.R, n_classes=2, positions=pos, vocab=2, pool=False,
                       boundary_wo=not a.no_wo, wo_mode=a.wo_mode, use_ov=not a.sheaf, sheaf=a.sheaf, lam_mode=a.lam).to(dev)
    kn = {"psi", "theta", "alpha_raw"}
    kp = [p for n, p in m.named_parameters() if n in kn]; rp = [p for n, p in m.named_parameters() if n not in kn]
    opt = torch.optim.AdamW([{"params": rp, "lr": a.lr}, {"params": kp, "lr": a.lr * a.kernel_lr_mult}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.lr * a.kernel_lr_mult], total_steps=a.steps, pct_start=0.15)
    net = torch.compile(m) if a.compile else m
    log = []; t0 = time.time()
    def evaluate(n=2048):
        with torch.no_grad():
            xe, re, ye = batch(n, a.T, a.k, rules, dev, gen); p_ = m((xe, re), R=a.R, steps=a.R * a.tau).argmax(-1)
            ok = p_ == ye; out = {}
            for i, r in enumerate(rules):
                mm = re == i; out[r] = {"cell": ok[mm].float().mean().item(), "exact": ok[mm].all(-1).float().mean().item(), "n": int(mm.sum())}
            return out, ok.float().mean().item(), ok.all(-1).float().mean().item()
    for i in range(a.steps):
        x, rid, y = batch(a.bs, a.T, a.k, rules, dev, gen)
        loss = F.cross_entropy(net((x, rid), R=a.R, steps=a.R * a.tau).reshape(-1, 2), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if (i + 1) % 500 == 0:
            per, cell, exact = evaluate()
            log.append({"step": i + 1, "loss": loss.item(), "cell": cell, "exact": exact, "per_rule": per})
            print(f"  [{a.tag}] step {i+1:5d} loss {loss.item():.4f} 셀 {cell:.4f} 전체일치 {exact:.4f} | " +
                  " ".join(f"r{r}(c{CLASS.get(r,'?')}):{per[r]['exact']:.2f}" for r in rules), flush=True)
    per, cell, exact = evaluate(8192)
    print(f"[{a.tag}] rules={rules} k={a.k} FINAL 셀 {cell:.4f} 전체일치 {exact:.4f} ({time.time()-t0:.0f}s)")
    for r in rules: print(f"   rule {r:3d} class {CLASS.get(r,'?')}: 셀 {per[r]['cell']:.4f} 전체일치 {per[r]['exact']:.4f}")
    torch.save({"model": m.state_dict(), "args": vars(a)}, f"runs/{a.tag}.pt")
    json.dump({"args": vars(a), "log": log, "final": {"cell": cell, "exact": exact, "per_rule": {str(r): v for r, v in per.items()}}}, open(f"runs/{a.tag}_cam.json", "w"))

if __name__ == "__main__": main()
