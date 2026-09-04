"""판독 지점의 상태 진폭 — block_order 가 Φ 를 지나쳐 읽는가.

가설: `pre` 의 블록은 Φ 로 끝나고(`step` 의 마지막이 `phi(h+f)`), `post` 는 경계(`h+Δ`, 정규화 없음)로 끝난다.
Φ 가 이 모델의 **유일한 진폭 통제 장치**(THEORY §5)이므로, post 는 판독과 carry 를 둘 다 Φ 끌개 밖에서 한다.
"""
import os, sys, argparse, yaml, torch, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "refs", "URM")); sys.path.insert(0, os.path.join(ROOT, "core"))
import minimal
from dataclasses import replace

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["PSI_post", "FAITH_psi_pre", "FAITH_beta_pre"])
ap.add_argument("--step", type=int, default=3906)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--segs", type=int, default=16)
a = ap.parse_args()

z = np.load(os.path.join(ROOT, "kaggle", "upload", "sudoku_lt_1k.npz"))
inp = torch.from_numpy(z["test_inputs"][:a.n].reshape(a.n, 81).astype(np.int64)) + 1   # 0=PAD,1=빈칸,2..10
batch = {"inputs": inp.cuda(), "puzzle_identifiers": torch.zeros(a.n, dtype=torch.long).cuda()}

print(f"{'run':<16}{'order':<6}{'win':<6}{'‖h‖ @판독 (seg1/4/8/16)':<34}{'‖Δ_경계‖/‖h‖':>14}")
for run in a.runs:
    d = os.path.join(ROOT, "refs", "URM", "checkpoints", run)
    ck = os.path.join(d, f"step_{a.step}.pt")
    if not os.path.exists(ck):
        print(f"{run:<16}(step_{a.step}.pt 없음)"); continue
    cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))["arch"]
    cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
    cfg.update(batch_size=a.n, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
               puzzle_emb_ndim=cfg.get("hidden_size", 832), amp=False)
    m = minimal.LT(cfg).cuda().eval()
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
    sd = {k.replace("_orig_mod.model.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)

    norms, ratio = [], []
    hook_h = {}
    orig_b = m.inner.boundary
    def b_wrap(h, gate=None, _o=orig_b):
        out = _o(h, gate)
        ratio.append(((out - h).norm(dim=-1) / (h.norm(dim=-1) + 1e-6)).mean().item())
        return out
    m.inner.boundary = b_wrap

    c = m.initial_carry(batch)
    c = replace(c, current_hidden=c.current_hidden.cuda(), steps=c.steps.cuda(), halted=c.halted.cuda(),
                current_data={k: v.cuda() for k, v in c.current_data.items()})
    with torch.no_grad():
        for _ in range(a.segs):
            c, _o = m(c, batch)
            norms.append(c.current_hidden.norm(dim=-1).mean().item())
    sel = [norms[0], norms[3], norms[7], norms[15]]
    print(f"{run:<16}{cfg.get('block_order','pre'):<6}{cfg.get('stdp_window','beta'):<6}"
          f"{'  '.join(f'{v:6.2f}' for v in sel):<34}{np.mean(ratio):>14.3f}")

g = torch.nn.functional.softplus(torch.tensor(minimal.inv_softplus(0.1)))
print(f"\nΦ 포화 반경(γ init 0.1 기준) = 1/√γ = {1/g.sqrt().item():.2f}  — 학습된 γ 는 런마다 다르다")
