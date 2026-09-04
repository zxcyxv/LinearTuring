"""진폭이 칸마다 갈리는가 — Φ 의 소프트 영역이 실제로 쓰이는지.

Φ 는 RMSNorm 과 달리 작은 ‖h_t‖ 를 부풀리지 않는다. 그래서 진폭이 "발언권" 채널이 될 수 있다:
어텐션은 진폭을 못 보지만(ẑ 정규화), 수송 f 는 진폭에 선형, 경계 Δ 는 2차, 로짓도 선형이다.
칸마다 진폭이 안 갈리면 그 채널은 안 쓰이는 것이고, Φ 는 사실상 RMSNorm 이다.
"""
import os, sys, argparse, importlib.util, yaml, torch, numpy as np
import torch.nn.functional as F
from dataclasses import replace
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "refs", "URM"))

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/tmp/claude-0/-workspace-LinearTuring/0ad42191-f4ae-4687-8bbb-b765c14ec6cc/scratchpad/minimal_310k.py")
ap.add_argument("--ckpt", default="checkpoints/R1B8_min_faith_step310527.pt")
ap.add_argument("--config", default="checkpoints/R1B8_min_faith_config.yaml")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--segs", type=int, default=16)
a = ap.parse_args()

sp = importlib.util.spec_from_file_location("mm", a.src); mm = importlib.util.module_from_spec(sp)
sys.modules["mm"] = mm; sp.loader.exec_module(mm)

cfg = yaml.safe_load(open(os.path.join(ROOT, a.config)))["arch"]
cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
cfg.update(batch_size=a.n, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=cfg.get("hidden_size", 832), amp=False)
m = mm.LT(cfg).cuda().eval()
sd = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu", weights_only=False)["model_state_dict"]
m.load_state_dict({k.replace("_orig_mod.model.", ""): v for k, v in sd.items()}, strict=True)
gam = F.softplus(m.inner.gamma_raw).item()

z = np.load(os.path.join(ROOT, "kaggle", "upload", "sudoku_lt_1k.npz"))
inp = torch.from_numpy(z["test_inputs"][:a.n].reshape(a.n, 81).astype(np.int64)) + 1
lab = torch.from_numpy(z["test_labels"][:a.n].reshape(a.n, 81).astype(np.int64)) + 1
batch = {"inputs": inp.cuda(), "puzzle_identifiers": torch.zeros(a.n, dtype=torch.long).cuda()}
given = (inp == 1).logical_not().cuda()          # 토큰 1 = 빈칸 → given = 숫자가 주어진 칸

c = m.initial_carry(batch)
c = replace(c, current_hidden=c.current_hidden.cuda(), steps=c.steps.cuda(), halted=c.halted.cuda(),
            current_data={k: v.cuda() for k, v in c.current_data.items()})
hist = []
with torch.no_grad():
    for _ in range(a.segs):
        c, out = m(c, batch)
        hist.append(c.current_hidden.norm(dim=-1))          # [B,81]
pred = out["logits"].argmax(-1)
correct = (pred == lab.cuda())

print(f"γ = {gam:.3e}   Φ 흡수구 반경 = {gam**-0.5:.2f}")
print(f"\n{'세그':>4}{'‖h_t‖ 평균':>12}{'표준편차':>10}{'변동계수':>10}{'최소':>9}{'최대':>9}{'γ‖h‖² 평균':>13}")
for s in (0, 3, 7, 15):
    v = hist[s]
    print(f"{s+1:>4}{v.mean():>12.3f}{v.std():>10.3f}{(v.std()/v.mean()):>10.4f}{v.min():>9.3f}{v.max():>9.3f}{(gam*v.pow(2)).mean():>13.1f}")

v = hist[-1]
g_, e_ = v[given], v[~given]
ec, ew = v[~given & correct], v[~given & ~correct]
print(f"\n마지막 세그먼트 (판독 시점) 집단별 ‖h_t‖")
print(f"  주어진 칸   n={given.sum():>5}   {g_.mean():.3f} ± {g_.std():.3f}")
print(f"  빈칸        n={(~given).sum():>5}   {e_.mean():.3f} ± {e_.std():.3f}")
print(f"    └ 맞춘 칸 n={(~given&correct).sum():>5}   {ec.mean():.3f} ± {ec.std():.3f}")
print(f"    └ 틀린 칸 n={(~given&~correct).sum():>5}   {ew.mean():.3f} ± {ew.std():.3f}")
d = (g_.mean() - e_.mean()) / v.std()
d2 = (ec.mean() - ew.mean()) / e_.std()
print(f"\n  주어진 vs 빈칸  차이 = {d:+.3f}σ")
print(f"  맞춤 vs 틀림    차이 = {d2:+.3f}σ")
mg = torch.stack([(v[i] - v[i].mean()).abs().mean() / v[i].mean() for i in range(a.n)]).mean()
print(f"  퍼즐 내 칸 간 상대 산포(평균절대편차/평균) = {mg:.4f}   ← RMSNorm 이면 0")
