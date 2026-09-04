"""상태 스케일 해부 — β 가 왜 커졌나. 각 항의 크기를 h 대비로 잰다."""
import os, sys, argparse, yaml, torch, numpy as np
import torch.nn.functional as F
from dataclasses import replace
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "refs", "URM")); sys.path.insert(0, os.path.join(ROOT, "core"))
import minimal

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/R1B8_min_faith_step310527.pt")
ap.add_argument("--config", default="checkpoints/R1B8_min_faith_config.yaml")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--segs", type=int, default=16)
a = ap.parse_args()

cfg = yaml.safe_load(open(os.path.join(ROOT, a.config)))["arch"]
cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
cfg.update(batch_size=a.n, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
           puzzle_emb_ndim=cfg.get("hidden_size", 832), amp=False)
m = minimal.LT(cfg).cuda().eval()
sd = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu", weights_only=False)["model_state_dict"]
# [2026-09-04] num_layers 리팩터 이전 체크포인트는 레이어 파라미터가 평평하다 → layers.0.* 로 매핑
_LK = ("wc_raw","psi","theta","alpha_raw","w_sh","eta_raw","lam_raw","gain_raw","beta",
       "gamma_raw","mu","gate_s_raw","b_gate_up.weight","b_down.weight")
def _remap(k):
    k = k.replace("_orig_mod.model.", "")
    if k.startswith("inner.") and not k.startswith("inner.layers."):
        t = k[len("inner."):]
        if t in _LK:
            return "inner.layers.0." + t
    return k
m.load_state_dict({_remap(k): v for k, v in sd.items()}, strict=True)
inner = m.inner

z = np.load(os.path.join(ROOT, "kaggle", "upload", "sudoku_lt_1k.npz"))
inp = torch.from_numpy(z["test_inputs"][:a.n].reshape(a.n, 81).astype(np.int64)) + 1
batch = {"inputs": inp.cuda(), "puzzle_identifiers": torch.zeros(a.n, dtype=torch.long).cuda()}

gam = F.softplus(inner.layers[0].gamma_raw).item()
beta = inner.inj_gate.item() if hasattr(inner, "inj_gate") else inner.embed_scale
with torch.no_grad():
    inj = inner.injection(batch)
print(f"γ = {gam:.3e}   흡수구 반경 = {gam**-0.5:.1f}   β = {beta:.3f}")
print(f"‖E(x)‖ 토큰 평균 = {inj.norm(dim=-1).mean().item():.3f}   ‖β·E(x)‖ = {beta*inj.norm(dim=-1).mean().item():.3f}")

rec = {"h_in": [], "after_bnd": [], "after_inj": [], "f": [], "after_phi": [], "phi_scale": []}
orig_b, orig_s = inner.boundary, inner.step
L0 = inner.layers[0]
def b_wrap(_L, h, gate=None):
    rec["h_in"].append(h.norm(dim=-1).mean().item()); out = orig_b(_L, h, gate)
    rec["after_bnd"].append(out.norm(dim=-1).mean().item()); return out
def s_wrap(_L, h, AB, kc, w=None, fresh=None, kcb=None, vprev=None, apply_phi=True):
    rec["after_inj"].append(h.norm(dim=-1).mean().item())
    out = orig_s(_L, h, AB, kc, w, fresh, kcb, vprev, apply_phi)
    hh = out[0]
    rec["after_phi"].append(hh.norm(dim=-1).mean().item())
    pre_phi = hh * torch.sqrt(1 + F.softplus(inner.layers[0].gamma_raw) * hh.pow(2).sum(-1, keepdim=True)) if apply_phi else hh
    rec["f"].append((pre_phi - h).norm(dim=-1).mean().item())
    rec["phi_scale"].append((hh.norm(dim=-1) / (pre_phi.norm(dim=-1) + 1e-9)).mean().item())
    return out
inner.boundary, inner.step = b_wrap, s_wrap

c = m.initial_carry(batch)
c = replace(c, current_hidden=c.current_hidden.cuda(), steps=c.steps.cuda(), halted=c.halted.cuda(),
            current_data={k: v.cuda() for k, v in c.current_data.items()})
with torch.no_grad():
    for _ in range(a.segs):
        c, out = m(c, batch)

B = cfg["blocks_per_seg"]
def seg(k, key): return rec[key][k*B:(k+1)*B]
print(f"\n{'세그':>4}{'블록':>4}{'‖h‖ 입력':>10}{'경계 뒤':>10}{'주입 뒤':>10}{'‖f‖':>9}{'Φ 뒤':>9}{'Φ 배율':>9}")
for s in (0, 7, 15):
    for b in (0, B-1):
        i = s*B+b
        print(f"{s+1:>4}{b+1:>4}{rec['h_in'][i]:>10.2f}{rec['after_bnd'][i]:>10.2f}{rec['after_inj'][i]:>10.2f}"
              f"{rec['f'][i]:>9.2f}{rec['after_phi'][i]:>9.2f}{rec['phi_scale'][i]:>9.4f}")
h = rec["h_in"]; ab = rec["after_bnd"]; ai = rec["after_inj"]
print(f"\n전체 평균 비율:  ‖Δ_경계‖/‖h‖ ≈ {np.mean([(ab[i]-h[i])/h[i] for i in range(len(h))]):.3f}"
      f"   ‖β·E‖/‖h_경계뒤‖ ≈ {beta*inj.norm(dim=-1).mean().item()/np.mean(ab):.3f}"
      f"   ‖f‖/‖h‖ ≈ {np.mean([rec['f'][i]/ai[i] for i in range(len(h))]):.3f}")
print(f"Φ 배율 평균 = {np.mean(rec['phi_scale']):.4f}  (1.0 이면 Φ 가 사실상 항등)")
print(f"로짓 std = {out['logits'].std().item():.3f}")
