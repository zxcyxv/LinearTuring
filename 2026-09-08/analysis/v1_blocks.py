"""세 퍼즐(#115 #295 #460)을 128세그 x 8블록 돌리며 블록 단위로 G·w·예측 기록."""
import os, sys, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False)
PIDS = [115, 295, 460]; NB, SEG, BPS = len(PIDS), 128, 8
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "2026-09-06/analysis/train_0901.py"))
tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
sys.path.insert(0, os.path.join(ROOT, "checkpoints"))
from ckpt_npz import load as npz_load
cfg = dict(tk.CFG); cfg.update(batch_size=NB, seq_len=81, num_puzzle_identifiers=1, loops=SEG + 1)
m = tk.LT(cfg).cuda().eval()
sd, _ = npz_load(os.path.join(ROOT, "checkpoints/0901_R1B8_min_faith_step310527.npz"), which="ema")
PER = {"wc_raw","psi","theta","alpha_raw","w_sh","eta_raw","lam_raw","gain_raw","beta","b_gate_up","b_down"}
def remap(k):
    k = k.replace("_orig_mod.", ""); k = k[len("model."):] if k.startswith("model.") else k
    t = k[len("inner."):]
    if t == "inj_gate": return "inner.embed_scale"
    return f"inner.layers.0.{t}" if t.split(".")[0] in PER else k
m.load_state_dict({remap(k): v for k, v in sd.items()}, strict=True)
I = m.inner; L = I.layers[0]

z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
X = z["test_inputs"].reshape(-1,81).astype(np.int32)[PIDS]; Y = z["test_labels"].reshape(-1,81).astype(np.int32)[PIDS]
x = torch.from_numpy(X+1).cuda(); y = torch.from_numpy(Y+1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(NB, dtype=torch.int32, device="cuda"))

H, T = 8, 81
sumG   = np.zeros((SEG, NB, H, T, T), np.float16)     # 세그먼트 안 8블록 합
sumAbs = np.zeros((SEG, NB, H, T, T), np.float16)
wEnd   = np.zeros((SEG, NB, H, T, T), np.float16)     # 세그먼트 끝 w
predB  = np.zeros((SEG*BPS, NB, T), np.int8)          # 블록마다 예측
ctr = {"b": 0}
orig = I.step
def hooked(Lx, h, AB, kc, w=None, fresh=None, kcb=None, vprev=None, apply_phi=True):
    b = ctr["b"]; si, bi = divmod(b, BPS)
    xy = I.addr(h, AB)
    a_b = I.attn_xy(xy, kcb)                                  # 쓰기 창 a_β
    v = torch.einsum('btd,hcd->bthc', h, Lx.w_sh)
    vv = v / (v.norm(dim=-1, keepdim=True) + I.config.eps)
    G = (a_b * torch.einsum('bthc,bnhc->bhtn', vv, vv)).float()   # G = a_β · agree
    out = orig(Lx, h, AB, kc, w, fresh, kcb, vprev, apply_phi)
    hout, wnew = out[0], out[1]
    g = G.cpu().numpy()
    sumG[si]   += g.astype(np.float16); sumAbs[si] += np.abs(g).astype(np.float16)
    if bi == BPS-1: wEnd[si] = wnew.float().cpu().numpy().astype(np.float16)
    predB[b] = I.w_cls(hout).argmax(-1).cpu().numpy().astype(np.int8)
    ctr["b"] += 1
    return out
I.step = hooked

t0 = time.time()
with torch.device("cuda"): carry = m.initial_carry(batch)
for si in range(SEG):
    carry, out = m(carry, batch)
print(f"{time.time()-t0:.0f}s   블록 {ctr['b']}")
np.savez_compressed("/tmp/claude-0/-workspace-LinearTuring/8092b54b-030b-4f2c-b7a7-8107892a6681/scratchpad/v1_blocks.npz",
                    sumG=sumG, sumAbs=sumAbs, wEnd=wEnd, predB=predB, X=X, Y=Y, pids=np.array(PIDS))
for i, p in enumerate(PIDS):
    corr = (predB[:, i] == (Y[i]+1)[None]).sum(1)
    print(f"#{p}: 맞은칸(블록 1,8,64,256,512,1024) = " + " ".join(str(corr[k]) for k in [0,7,63,255,511,1023]))
