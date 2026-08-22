"""n-홉 연상 회상의 트랜스포머 기준선 — 파라미터를 Model 1 (d=256, 427,330개) 에 맞춘다.

공정성 조건:
  · 데이터 생성기는 hop_task.batch 를 그대로 import (동일 분포·동일 시드)
  · 양방향(캐주얼 마스크 없음) — 우리 모델이 양방향이므로 같은 조건
  · 질의 위치에서만 손실 (동일 읽기)
  · 같은 옵티마이저 계열(AdamW+OneCycle)·같은 total_steps
    → OneCycleLR 은 total_steps 에 궤적이 종속되므로 예산이 다르면 비교가 오염된다
  · lr 은 쓸어서 기준선을 과소조정하지 않는다

L=1 은 대조군이다: induction head 는 2층 이상을 요구한다는 게 표준 결과이므로
L=1 이 실패하고 L≥2 가 성공하면 기준선이 문헌대로 작동한다는 확인이 된다.
"""
import argparse, json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from hop_task import batch


class TFBaseline(nn.Module):
    def __init__(self, V, T, d, L, nhead=4):
        super().__init__()
        self.emb = nn.Embedding(V + 1, d)
        # GPT-2 규약: 토큰/위치 임베딩을 같은 스케일(std 0.02)로 초기화한다.
        # PyTorch 기본 nn.Embedding init 은 N(0,1) 이라 pos(0.02) 와 50배 차이가 나고,
        # pre-LN 을 지나면 위치 성분이 소멸한다 -> 어텐션이 순열 동변이 되어
        # "이전 토큰" 관계를 만들 수 없다 (induction 회로의 1층이 원천 불가능).
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4 * d,
                                           batch_first=True, norm_first=True,
                                           dropout=0.0, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, L)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V + 1)

    def forward(self, x):
        h = self.emb(x) + self.pos
        return self.head(self.ln(self.enc(h)))


def evaluate(m, a, dev, gen, nb=4096):
    with torch.no_grad():
        x, y = batch(nb, a.V, a.N, a.Q, a.n, dev, gen)
        p = m(x)[:, -a.Q:].argmax(-1)
        return (p == y).float().mean().item(), (p == y).all(-1).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--V", type=int, default=64); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--Q", type=int, default=4);  ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sched", default="onecycle", choices=["onecycle", "const"])
    ap.add_argument("--opt", default="adamw", choices=["adamw", "adam"])
    ap.add_argument("--eval_every", type=int, default=250)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
    T = 2 * a.N + a.Q
    m = TFBaseline(a.V, T, a.d, a.L, a.nhead).to(dev)
    nparam = sum(p.numel() for p in m.parameters())
    opt = (torch.optim.Adam(m.parameters(), lr=a.lr) if a.opt == "adam"
           else torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4))
    sch = (torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.15)
           if a.sched == "onecycle" else None)
    print(f"[{a.tag}] TF L={a.L} d={a.d} heads={a.nhead} lr={a.lr} opt={a.opt} sched={a.sched} "
          f"bs={a.bs} steps={a.steps} T={T} n={a.n} params={nparam} "
          f"(Model1 d=256 = 427,330)  우연={1/a.N:.4f}", flush=True)
    log = []; t0 = time.time(); best = 0.0
    for i in range(a.steps):
        x, y = batch(a.bs, a.V, a.N, a.Q, a.n, dev, gen)
        loss = F.cross_entropy(m(x)[:, -a.Q:].reshape(-1, a.V + 1), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if sch is not None: sch.step()
        if (i + 1) % a.eval_every == 0:
            per, ex = evaluate(m, a, dev, gen, 2048); best = max(best, per)
            log.append({"step": i + 1, "loss": loss.item(), "query": per, "exact": ex})
            print(f"  [{a.tag}] step {i+1:5d} loss {loss.item():.4f} 질의 {per:.4f} 전체일치 {ex:.4f}", flush=True)
    per, ex = evaluate(m, a, dev, gen, 8192)
    print(f"[{a.tag}] TF L={a.L} d={a.d} lr={a.lr} n={a.n} params={nparam} "
          f"FINAL 질의 {per:.4f} 전체일치 {ex:.4f} (최고 {best:.4f}) ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"args": vars(a), "log": log, "query": per, "exact": ex, "best": best,
               "params": nparam}, open(f"runs/{a.tag}_hop.json", "w"))


if __name__ == "__main__":
    main()
