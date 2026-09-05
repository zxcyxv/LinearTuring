"""9/1 R1B8_min_faith@310527 외삽 이득이 w 기억에 의존하는가. 512퍼즐, seg128. 변형: base | freeze(seg≥16 갱신 중지) | seg_reset | block_reset."""
import os, sys, importlib.util, time, yaml, numpy as np, torch
from dataclasses import replace
ROOT = "/workspace/LinearTuring"; H = os.path.join(ROOT, "refs", "ckpt_hist"); sys.path.insert(0, H); torch.set_grad_enabled(False)
BSZ = 64; NB = 512; SEG = 128
s = importlib.util.spec_from_file_location("minimal", os.path.join(H, "minimal_0901.py")); mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod)
cfg = yaml.safe_load(open(os.path.join(H, "R1B8_min_faith_config.yaml")))["arch"]; cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
cfg.update(batch_size=BSZ, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, loops=SEG + 1)
ck = torch.load(os.path.join(H, "R1B8_min_faith_step310527.pt"), map_location="cpu", weights_only=False)
full = dict(ck["raw_model_state_dict"]); full.update(ck["model_state_dict"])
m = mod.LT(cfg).cuda().eval(); m.load_state_dict({k.replace("_orig_mod.model.", ""): v for k, v in full.items()}, strict=False); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:NB].reshape(NB, 81).astype(int); Y = z["test_labels"][:NB].reshape(NB, 81).astype(int)
orig_step = I.step; mode = {"m": "base", "seg": 0}
def step_patched(h, AB, kc, w=None, fresh=None, kcb=None, vprev=None):
    if mode["m"] == "freeze" and mode["seg"] >= 16 and w is not None:
        out = orig_step(h, AB, kc, w, fresh, kcb, vprev); return (out[0], w) + tuple(out[2:])
    if mode["m"] == "block_reset" and w is not None:
        return orig_step(h, AB, kc, None, None, kcb, vprev)
    return orig_step(h, AB, kc, w, fresh, kcb, vprev)
I.step = step_patched
def run(m_):
    mode["m"] = m_; ex = np.zeros(SEG); t0 = time.time()
    for b in range(0, NB, BSZ):
        x = torch.from_numpy(X[b:b+BSZ].astype(np.int32)+1).cuda(); y = torch.from_numpy(Y[b:b+BSZ].astype(np.int32)+1).cuda().long()
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(len(x), dtype=torch.int32, device="cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.device("cuda"):
            carry = m.initial_carry(batch)
            for s_ in range(SEG):
                mode["seg"] = s_
                if m_ == "seg_reset" and s_ > 0: carry = replace(carry, fresh=torch.ones(len(x), dtype=torch.bool, device="cuda"))
                carry, out = m(carry, batch); ex[s_] += (out["logits"].argmax(-1) == y).all(-1).sum().item()
        print(f"  [{m_}] {b+len(x)}/{NB}  {time.time()-t0:.0f}s  누적 seg16 {int(ex[15])} seg128 {int(ex[-1])}", flush=True)
    return ex
res = {k: run(k) for k in ["base", "freeze", "seg_reset", "block_reset"]}
print(f"\n9/1 R1B8_min_faith@310527\n{'변형':>14} | seg16 | seg32 | seg64 | seg128 | 이득(16→128)")
for k, e in res.items(): print(f"{k:>14} | {int(e[15]):5d} | {int(e[31]):5d} | {int(e[63]):5d} | {int(e[127]):6d} | {int(e[127]-e[15]):+d}")
