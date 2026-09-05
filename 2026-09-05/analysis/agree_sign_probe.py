"""stdp1 에서 agree = ⟨v̂_t, v̂_n⟩ 이 실제로 음수가 되는가. 창 a 의 부호를 agree 가 얼마나 뒤집는가.
9/1 커밋의 core/minimal.py 로 로드 (학습 당시 코드). 테스트 256 퍼즐, 16 세그 × 8 블록 = 128 블록 전부 기록."""
import os, sys, importlib.util, numpy as np, torch, yaml
ROOT = "/workspace/LinearTuring"; H = os.path.join(ROOT, "refs", "ckpt_hist")
sys.path.insert(0, H)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(H, "minimal_0901.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False)
name = sys.argv[1] if len(sys.argv) > 1 else "R1B8_min_stdp1_step228501"
cfgy = [f for f in os.listdir(H) if f.endswith("_config.yaml") and name.split("_step")[0] in f][0]
cfg = yaml.safe_load(open(os.path.join(H, cfgy)))["arch"]; cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
cfg.update(batch_size=256, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, loops=17)
ck = torch.load(os.path.join(H, name + ".pt"), map_location="cpu", weights_only=False)
full = dict(ck["raw_model_state_dict"]); full.update(ck["model_state_dict"])
sd = {k.replace("_orig_mod.model.", ""): v for k, v in full.items()}
m = mod.LT(cfg).cuda().eval(); miss = m.load_state_dict(sd, strict=False)
assert not miss.unexpected_keys, miss.unexpected_keys
if miss.missing_keys: print("누락(기본값):", miss.missing_keys)
I = m.inner
print(f"{name}: step={ck['step']} target={cfg.get('stdp_target')} lam_fixed={cfg.get('stdp_lam_fixed')}  δ={torch.sigmoid(I.eta_raw).flatten().mean():.3f}")
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
x = torch.from_numpy(z["test_inputs"][:256].reshape(-1, 81).astype(np.int32) + 1).cuda()
y = torch.from_numpy(z["test_labels"][:256].reshape(-1, 81).astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(256, dtype=torch.int32, device="cuda"))
# 쌍 마스크
r = torch.arange(81, device="cuda") // 9; c = torch.arange(81, device="cuda") % 9; b = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (b[:, None] == b[None])) & ~torch.eye(81, dtype=torch.bool, device="cuda")
offd = ~torch.eye(81, dtype=torch.bool, device="cuda")
same_digit = (y[:, :, None] == y[:, None, :]) & offd            # 정답 숫자가 같은 쌍 (동료는 절대 아님)
rec = []
orig_step = I.step
def hooked(h, AB, kc, *a_, **k_):
    xy = I.addr(h, AB); a = I.attn_xy(xy, kc)                          # [B,H,T,T]
    v = torch.einsum('btd,hcd->bthc', h, I.w_sh); vv = v / (v.norm(dim=-1, keepdim=True) + I.config.eps)
    ag = torch.einsum('bthc,bnhc->bhtn', vv, vv)
    G = a * ag
    P = peer[None, None]; O = offd[None, None]; S = same_digit[:, None]
    rec.append(dict(
        neg_all=(ag[O.expand_as(ag)] < 0).float().mean().item(),
        neg_peer=(ag[P.expand_as(ag)] < 0).float().mean().item(),
        mean_peer=ag[P.expand_as(ag)].mean().item(), mean_same=ag[S.expand_as(ag)].mean().item(),
        neg_same=(ag[S.expand_as(ag)] < 0).float().mean().item(),
        flip_mass=(G.abs() * (ag < 0)).sum().item() / G.abs().sum().item(),   # |Γ| 질량 중 창 부호가 뒤집힌 몫
        a_peer=a[P.expand_as(a)].mean().item(), G_peer=G[P.expand_as(G)].mean().item(),
        a_neg_peer=(a[P.expand_as(a)] < 0).float().mean().item()))
    return orig_step(h, AB, kc, *a_, **k_)
I.step = hooked
with torch.device("cuda"): carry = m.initial_carry(batch)
for s in range(16): carry, _ = m(carry, batch)
n = len(rec); print(f"기록 블록 수 {n}")
keys = ["neg_all", "neg_peer", "neg_same", "mean_peer", "mean_same", "flip_mass", "a_peer", "G_peer", "a_neg_peer"]
print(f"{'블록':>8} | " + " ".join(f"{k:>10}" for k in keys))
for lab, idx in [("seg1 blk1", [0]), ("seg1 blk8", [7]), ("seg2 blk1", [8]), ("seg8", range(56, 64)), ("seg16", range(120, 128)), ("전체", range(n))]:
    print(f"{lab:>8} | " + " ".join(f"{np.mean([rec[i][k] for i in idx]):10.3f}" for k in keys))
