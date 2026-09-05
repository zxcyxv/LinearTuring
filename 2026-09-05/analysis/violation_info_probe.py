"""동료 쌍(peer)을 '현재 예측 숫자가 같은가(위반)' 로 갈라 창 a 와 agree 가 그 정보를 얼마나 담는지.
a 가 위반/비위반을 이미 구별하면 agree 는 중복이고, 못 하면 agree 가 유일한 위반 정보원.
예측 = 블록 입력 h 의 w_cls 로짓 argmax (빈칸만 대상, 주어진 칸 제외)."""
import os, sys, importlib.util, numpy as np, torch, yaml
ROOT = "/workspace/LinearTuring"; H = os.path.join(ROOT, "refs", "ckpt_hist"); sys.path.insert(0, H)
torch.set_grad_enabled(False)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
x = torch.from_numpy(z["test_inputs"][:256].reshape(-1, 81).astype(np.int32) + 1).cuda()
y = torch.from_numpy(z["test_labels"][:256].reshape(-1, 81).astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(256, dtype=torch.int32, device="cuda"))
r = torch.arange(81, device="cuda") // 9; c = torch.arange(81, device="cuda") % 9; b = (r // 3) * 3 + c // 3
peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (b[:, None] == b[None])) & ~torch.eye(81, dtype=torch.bool, device="cuda")
blank = (x == 1)                                                     # 입력 토큰 1 = 빈칸(0+1)
def load(name):
    if name == "kaggle":
        s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
        ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
        cfg = dict(ck["cfg"]); cfg.update(batch_size=256, seq_len=81, num_puzzle_identifiers=1, loops=17)
        sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
        m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); return m, True
    s = importlib.util.spec_from_file_location("minimal", os.path.join(H, "minimal_0901.py")); mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod)
    cfgy = [f for f in os.listdir(H) if f.endswith("_config.yaml") and name.split("_step")[0] in f][0]
    cfg = yaml.safe_load(open(os.path.join(H, cfgy)))["arch"]; cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
    cfg.update(batch_size=256, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, loops=17)
    ck = torch.load(os.path.join(H, name + ".pt"), map_location="cpu", weights_only=False)
    full = dict(ck["raw_model_state_dict"]); full.update(ck["model_state_dict"])
    m = mod.LT(cfg).cuda().eval(); m.load_state_dict({k.replace("_orig_mod.model.", ""): v for k, v in full.items()}, strict=False); return m, False
for name in ["R1B8_min_stdp1_step228501", "R1B8_min_faith_step310527", "kaggle"]:
    m, layered = load(name); I = m.inner
    acc = {}
    def record(tag, h, a, wsh):
        pred = I.w_cls(h).argmax(-1)                                      # [B,T]
        v = torch.einsum('btd,hcd->bthc', h, wsh); vv = v / (v.norm(dim=-1, keepdim=True) + I.config.eps)
        ag = torch.einsum('bthc,bnhc->bhtn', vv, vv)
        bb = blank[:, :, None] & blank[:, None, :]                        # 둘 다 빈칸
        viol = peer[None] & bb & (pred[:, :, None] == pred[:, None, :])   # 동료 + 같은 예측 = 위반
        ok = peer[None] & bb & (pred[:, :, None] != pred[:, None, :])     # 동료 + 다른 예측 = 만족
        V = viol[:, None].expand_as(a); O = ok[:, None].expand_as(a)
        d = acc.setdefault(tag, dict(a_v=[], a_o=[], g_v=[], g_o=[], G_v=[], G_o=[], nv=[]))
        d["a_v"].append(a[V].mean().item()); d["a_o"].append(a[O].mean().item())
        d["g_v"].append(ag[V].mean().item()); d["g_o"].append(ag[O].mean().item())
        d["G_v"].append((a * ag)[V].mean().item()); d["G_o"].append((a * ag)[O].mean().item())
        d["nv"].append(viol.float().sum().item() / max(bb.float().sum().item(), 1))
    orig = I.step
    if layered:
        def hooked(L, h, AB, kc, *a_, **k_):
            record(f"layer{0 if L is I.layers[0] else 1}", h, I.attn_xy(I.addr(h, AB), kc), L.w_sh); return orig(L, h, AB, kc, *a_, **k_)
    else:
        def hooked(h, AB, kc, *a_, **k_):
            record("single", h, I.attn_xy(I.addr(h, AB), kc), I.w_sh); return orig(h, AB, kc, *a_, **k_)
    I.step = hooked
    with torch.device("cuda"): carry = m.initial_carry(batch)
    for s in range(16): carry, _ = m(carry, batch)
    print(f"\n=== {name} ===")
    print(f"{'':>8} | {'a 위반':>8} {'a 만족':>8} {'차':>8} | {'agree 위반':>10} {'agree 만족':>10} {'차':>8} | {'Γ 위반':>8} {'Γ 만족':>8} | 위반쌍비율")
    for tag, d in acc.items():
        for lab, idx in [("seg1", range(0, 8)), ("seg8", range(56, 64)), ("seg16", range(120, 128)), ("전체", range(len(d["a_v"])))]:
            f = lambda k: np.nanmean([d[k][i] for i in idx])
            print(f"{tag+' '+lab:>8} | {f('a_v'):8.4f} {f('a_o'):8.4f} {f('a_v')-f('a_o'):+8.4f} | {f('g_v'):10.4f} {f('g_o'):10.4f} {f('g_v')-f('g_o'):+8.4f} | {f('G_v'):8.4f} {f('G_o'):8.4f} | {f('nv'):.3f}")
    del m; torch.cuda.empty_cache()
