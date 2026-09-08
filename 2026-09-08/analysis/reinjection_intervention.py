"""추론 시점 개입 — 입력 재주입 '비중' 램프.

블록마다(pre 순서: boundary → 주입 → step):
    C: h ← (1-ρ)·h + (1+ρ)·embed_scale·inj      (계수 교환, 총량 보존)
    B: h ← (1-ρ)·h +        embed_scale·inj      (은닉만 감쇠)
    A: h ←        h + (1+ρ)·embed_scale·inj      (주입만 증폭, 대조군)
ρ(s) = 0 for s<S0, 이후 S0..SEG-1 에서 0→ρ_max 선형.

usage: python interv2.py '<json list of {mode,S0,rho_max,tag}>'
"""
import os, sys, importlib.util, json, numpy as np, torch
from dataclasses import replace

ROOT = "/workspace/LinearTuring"
OUT  = "/tmp/claude-0/-workspace-LinearTuring/8092b54b-030b-4f2c-b7a7-8107892a6681/scratchpad/runs"
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
NB, SEG = 512, 256

s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "2026-09-06/analysis/train_0901.py"))
tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
sys.path.insert(0, os.path.join(ROOT, "checkpoints"))
from ckpt_npz import load as npz_load

cfg = dict(tk.CFG); cfg.update(batch_size=NB, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1)
m = tk.LT(cfg).cuda().eval()
sd, meta = npz_load(os.path.join(ROOT, "checkpoints/0901_R1B8_min_faith_step310527.npz"), which="ema")
PER = {"wc_raw","psi","theta","alpha_raw","w_sh","eta_raw","lam_raw","gain_raw","beta","b_gate_up","b_down"}
def remap(k):
    k = k.replace("_orig_mod.", ""); k = k[len("model."):] if k.startswith("model.") else k
    tail = k[len("inner."):]
    if tail == "inj_gate": return "inner.embed_scale"
    return f"inner.layers.0.{tail}" if tail.split(".")[0] in PER else k
m.load_state_dict({remap(k): v for k, v in sd.items()}, strict=True)
I = m.inner
ES0 = float(I.embed_scale.data.item())

# --- 블록 단위 은닉 감쇠: pre 순서에서 boundary 직후 = 주입 직전 ---
_orig_boundary = tk.LT_Inner.boundary
STATE = {"decay": 0.0, "noise": 0.0}
def patched_boundary(self, L, h, gate=None):
    out = _orig_boundary(self, L, h, gate)
    d = STATE["decay"]
    if d != 0.0: out = (1.0 - d) * out
    n = STATE["noise"]
    if n != 0.0:
        sc = out.norm(dim=-1, keepdim=True) / (out.shape[-1] ** 0.5)
        out = out + n * sc * torch.randn_like(out)
    return out
tk.LT_Inner.boundary = patched_boundary

z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
X = z["test_inputs"].reshape(-1,81).astype(np.int32)[:NB]
Y = z["test_labels"].reshape(-1,81).astype(np.int32)[:NB]
x = torch.from_numpy(X+1).cuda(); y = torch.from_numpy(Y+1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(NB, dtype=torch.int32, device="cuda"))
clue = torch.from_numpy(X > 0).cuda()
nclue = clue.float().sum(-1)

gam = None
for nm, p in m.named_parameters():
    if "gamma" in nm.lower(): gam = (nm, float(p.reshape(-1)[0]))
SNAPS = (16, 64, 128, 192, 256)

def run(mode, S0, RHOMAX, TAG, SEED=0):
    def rho(si):
        if mode == "base" or si < S0: return 0.0
        return RHOMAX * (si - S0 + 1) / (SEG - S0)
    with torch.device("cuda"): carry = m.initial_carry(batch)
    STATE["decay"] = 0.0; STATE["noise"] = 0.0; I.embed_scale.data.fill_(ES0)
    torch.manual_seed(SEED)
    prev = None; churn_sum = 0.0; churn_n = 0
    snap = {}; curve = []; hn = []
    for si in range(SEG):
        r = rho(si)
        STATE["decay"] = r if mode in ("B", "C") else 0.0
        STATE["noise"] = r if mode == "N" else 0.0
        I.embed_scale.data.fill_(ES0 * (1.0 + r) if mode in ("A", "C", "S") else ES0)
        if mode == "S" and r > 0:
            STATE["decay"] = -r / (1.0 + r)   # (1-decay)=(1+r) → h·(1+r), inj·(1+r): 비율 불변
        carry, out = m(carry, batch)
        pred = out["logits"].argmax(-1)
        ok = (pred == y).all(-1)
        curve.append(int(ok.sum().item()))
        hn.append(float(carry.current_hidden.norm(dim=-1).mean()))
        if prev is not None:
            ch = (pred != prev).float().mean().item()
            if si >= 128: churn_sum += ch; churn_n += 1
        prev = pred
        if si + 1 in SNAPS:
            cok = ((pred == x) & clue).float().sum(-1) / nclue     # 단서칸 유지율
            snap[si+1] = dict(pred=pred.cpu().numpy().astype(np.int8), exact=ok.cpu().numpy(),
                              cell=(pred == y).float().mean().item(), clue=cok.cpu().numpy())
    STATE["decay"] = 0.0; I.embed_scale.data.fill_(ES0)
    res = dict(mode=mode, S0=S0, rho_max=RHOMAX, tag=TAG, seed=SEED, es0=ES0, gamma=gam,
               exact={k: int(v["exact"].sum()) for k, v in snap.items()},
               cell={k: round(v["cell"], 5) for k, v in snap.items()},
               clue_keep={k: round(float(v["clue"].mean()), 5) for k, v in snap.items()},
               churn_128_256=churn_sum / max(churn_n, 1),
               hnorm={k: round(hn[k-1], 2) for k in SNAPS})
    np.savez(os.path.join(OUT, TAG + ".npz"),
             **{f"pred{k}": v["pred"] for k, v in snap.items()},
             **{f"exact{k}": v["exact"] for k, v in snap.items()},
             **{f"clue{k}": v["clue"] for k, v in snap.items()},
             curve=np.array(curve), hnorm=np.array(hn), meta=json.dumps(res))
    print(json.dumps(res), flush=True)

for c in json.loads(sys.argv[1]):
    run(c["mode"], c.get("S0", 16), c.get("rho_max", 0.0), c["tag"], c.get("seed", 0))
