"""v2.1 206k: 곱셈 읽기의 지수 x = sign(a)·w 를 실제로 잰다.
w 가 a 와 독립인 자유도를 갖는가 = x 가 음수로 갈 수 있는가.
step() 이 갱신된 w 를 반환하므로 블록 안에서 실제로 쓰인 x 를 정확히 복원할 수 있다 (근사 아님)."""
import importlib.util, torch, numpy as np
ROOT = "/workspace/LinearTuring"
spec = importlib.util.spec_from_file_location("tk", f"{ROOT}/kaggle/pull/output/lt_train.py")
tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)

CK = f"{ROOT}/kaggle/pull/output/checkpoints/step_206100.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=64, seq_len=81, num_puzzle_identifiers=1, loops=10000, compile=False)
m = tk.LT(cfg).cuda().eval()
def _strip(d): return {k[len("model."):] if k.startswith("model.") else k: v for k, v in tk.strip_prefix(d).items()}
sd = _strip(ck["model_state_dict"])
m.load_state_dict(sd, strict=True)
ema = _strip(ck["ema_shadow"])                                    # 외삽과 같은 EMA 가중치를 덮어씀
missing = m.load_state_dict(ema, strict=False)
print(f"EMA 덮어쓴 텐서 {len(ema)}개, EMA 에 없어 원본 유지: {list(missing.missing_keys)}")
I = m.inner
print(f"ckpt step={ck['step']}  read={cfg['stdp_read']} target={cfg['stdp_target']} window={cfg['stdp_window']} "
      f"diag={cfg['stdp_diag']} layers={cfg['num_layers']} order={cfg['block_order']}")

z = np.load(f"{ROOT}/kaggle/upload/sudoku_lt_1k.npz")
B = 64
X = torch.from_numpy(z["test_inputs"][:B].reshape(B, 81).astype(np.int32) + 1).cuda()
Y = torch.from_numpy(z["test_labels"][:B].reshape(B, 81).astype(np.int64) + 1).cuda()
batch = dict(inputs=X, labels=Y, puzzle_identifiers=torch.zeros(B, dtype=torch.int32, device="cuda"))
eye = torch.eye(81, device="cuda").bool()

def probe(carry, tag):
    """다음 세그먼트의 블록 0(레이어0)·블록 1(레이어1)을 그대로 재현해 x 를 뽑는다."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inj = I.injection(batch); w = carry.current_hidden.new_tensor(0) if carry.coupling is None else carry.coupling
        h = carry.current_hidden
        out = []
        for li in (0, 1):
            L = I.layers[li]; AB = I.W_C(L); kc = I.kernel(L)
            h = h + I.embed_scale * inj                                   # block_order="post"
            a = I.attn(h, AB, kc).float()
            h, w = I.step(L, h, AB, kc, w, None, None, apply_phi=False)
            x = (torch.sign(a) * w.float()).clamp(-4, 4)                  # step 내부와 동일
            h = I.phi(I.boundary(L, h))
            out.append((a, w.float().clone(), x))
    print(f"\n===== {tag} =====")
    a_all = []
    for li, (a, w, x) in enumerate(out):
        off = ~eye.expand_as(a[0, 0]).expand(a.shape)
        xo, ao = x[off], a[off]; g = torch.exp(xo)
        pos, neg = ao > 0, ao < 0
        print(f"[L{li}]  x=sign(a)·w :  평균 {xo.mean():+.4f}   음수 비율 {(xo<0).float().mean()*100:5.2f}%   "
              f"x<-0.1 비율 {(xo<-0.1).float().mean()*100:5.2f}%   |x| 평균 {xo.abs().mean():.4f}")
        print(f"       이득 exp(x) :  전체 {g.mean():.3f}   a>0 쌍 {g[pos].mean():.3f} (<1 비율 {(g[pos]<1).float().mean()*100:.1f}%)"
              f"   a<0 쌍 {g[neg].mean():.3f} (<1 비율 {(g[neg]<1).float().mean()*100:.1f}%)")
        # w 가 â 를 얼마나 그대로 베끼는가
        ahat = a / (a.float().pow(2).mean(dim=(-2,-1), keepdim=True).sqrt() + 1e-4)
        c = torch.nn.functional.cosine_similarity(w[off].flatten(), ahat[off].flatten(), dim=0)
        print(f"       cos(w, â) = {c:.4f}     부호 일치율 sign(w)==sign(a): {(torch.sign(w[off])==torch.sign(ao)).float().mean()*100:.2f}%")
        a_all.append(ao)
    c01 = torch.nn.functional.cosine_similarity(a_all[0].flatten(), a_all[1].flatten(), dim=0)
    print(f"[두 레이어] cos(a_L0, a_L1) = {c01:.4f}")

with torch.no_grad(), torch.device("cuda"):
    carry = m.initial_carry(batch)
    for s in range(1, 129):
        carry, out = m(carry, batch)
        if s in (16, 64, 128):
            probe(carry, f"{s} 세그먼트 뒤")
    logits = out["logits"]; pred = logits.argmax(-1)
    print(f"\n최종 seg128 완답 {(pred == Y).all(-1).sum().item()}/{B}")
