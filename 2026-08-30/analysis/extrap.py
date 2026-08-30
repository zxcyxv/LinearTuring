"""반복 외삽: 학습은 16 세그먼트, 추론에서 재귀를 계속 돌리면(24…128) 정확도·손실이 더 내려가는가.
원판(R1B8_bilin_r2, 결합 기억 없음) vs stdp1(minimal + STDP 충실형, w 가 세그먼트를 넘어 누적).
변형: 원판+추론 EMA(η .05 λ .25, 어제 +52 설정) · stdp1 w 세그먼트마다 초기화(누적 축 절단) · stdp1 η×0.5/×2.
세그먼트마다 기록: 완답 · 유효 격자 · 빈칸 셀 정확도 · 하네스 손실(stablemax CE, 81칸, 학습 lm_loss 와 같은 정의) · 빈칸만 손실 · softmax 9숫자 CE(참고) · 뒤집힘 비율 · '한 번이라도 완답' 합집합 · 16 에서 완답이던 것의 유지.
사용: python analysis/extrap.py [--max 128] [--n 2048] [--variants base base_ema stdp stdp_reset stdp_eta0.5 stdp_eta2] [--out results/json/extrap.json]"""
import argparse, json, os, sys, time, importlib.util, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
from common import ROOT, URM, load_test, make_batch, peer_mask, load_lt, logits as base_logits
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.join(ROOT, "checkpoints", "R1B8_bilin_r2_step123039.pt"))
ap.add_argument("--stdp", default=os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"))
ap.add_argument("--n", type=int, default=2048); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--max", type=int, default=128)
ap.add_argument("--variants", nargs="+", default=["base", "base_ema", "stdp", "stdp_reset", "stdp_eta0.5", "stdp_eta2"])
ap.add_argument("--marks", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64, 96, 128, 192, 256])
ap.add_argument("--out", default=os.path.join(ROOT, "results", "json", "extrap.json"))
args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = (lab - 2).long(); bl = inp == 1; pm = torch.tensor(peer_mask(), device="cuda")
K = 8; MAX = args.max


class Rec:
    """세그먼트별 누적 채점."""
    def __init__(self):
        self.exact = np.zeros(MAX); self.valid = np.zeros(MAX); self.cell = np.zeros(MAX); self.ce = np.zeros(MAX); self.flip = np.zeros(MAX)
        self.loss = np.zeros(MAX); self.loss_bl = np.zeros(MAX)
        self.solved = torch.zeros(N, MAX, dtype=torch.bool, device="cuda"); self.nbl = 0

    def add(self, s, b, n, lg, prevP):
        lg = lg.float(); x = inp[b:b + n]; g = G[b:b + n]; m = bl[b:b + n]
        sx = torch.where(lg < 0, 1 / (1 - lg.double() + 1e-30), lg.double() + 1)              # 하네스 stablemax (losses.py)
        lp = torch.log(sx / sx.sum(-1, keepdim=True)); nll = -lp.gather(-1, lab[b:b + n].long().unsqueeze(-1)).squeeze(-1)   # [n,81] 11-way
        self.loss[s] += float(nll.mean(-1).sum()); self.loss_bl[s] += float(nll[m].sum())
        lg = lg[:, :, 2:11]; P = lg.argmax(-1)
        ok = ((P == g) | ~m).all(-1); fin = torch.where(m, P, x - 2)
        valid = ~((fin[:, :, None] == fin[:, None, :]) & pm[None]).any(-1).any(-1)
        ce = F.cross_entropy(lg.reshape(-1, 9), g.reshape(-1), reduction="none").view(n, 81)[m].sum()
        self.exact[s] += int(ok.sum()); self.valid[s] += int(valid.sum()); self.cell[s] += int(((P == g) & m).sum()); self.ce[s] += float(ce)
        if prevP is not None: self.flip[s] += int(((P != prevP) & m).sum())
        self.solved[b:b + n, s] = ok
        return P

    def done(self):
        nbl = int(bl.sum()); ever = self.solved.cumsum(1).bool().sum(0).cpu().numpy()
        keep16 = (self.solved & self.solved[:, 15:16]).sum(0).cpu().numpy() if MAX >= 16 else None
        d = dict(exact=self.exact.astype(int).tolist(), valid=self.valid.astype(int).tolist(), cell=(self.cell / nbl).round(4).tolist(),
                 ce9=(self.ce / nbl).round(4).tolist(), flip=(self.flip / nbl).round(4).tolist(), ever=ever.astype(int).tolist(),
                 loss=(self.loss / N).round(4).tolist(), loss_bl=(self.loss_bl / nbl).round(4).tolist())
        if keep16 is not None: d["keep16"] = keep16.astype(int).tolist()
        return d


def run_base(m, eta=1.0, lam=0.0):
    inner = m.inner; core = inner.core; fc = core.kernel_fast(); AB = core.W_C(); r = Rec()
    for b in range(0, N, args.bs):
        x = inp[b:b + args.bs]; n = len(x); inj = inner._injection(make_batch(x, x)); h = inner.init_hidden.expand(n, 81, -1).clone(); w = None; P = None
        for s in range(MAX):
            for _ in range(K):
                h = inner._boundary(h) + inner.inj_gate * inj; hp = core.phi(h, .5); a, *_ = core.attn_fast(hp, *fc, AB=AB)
                w = a.clone() if w is None else w + eta * (a - w)
                f, *_ = core.field(hp, None, None, a_fixed=(1 - lam) * a + lam * w, AB=AB, fast_ctx=fc); h = core.phi(hp + f, .5)
            P = r.add(s, b, n, base_logits(m, h), P)
    return r.done()


def load_stdp(path, bs):
    ck = torch.load(path, map_location="cpu", weights_only=False); sd = ck.get("model_state_dict", ck)
    sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") or k.startswith("model.") else k): v for k, v in sd.items()}
    cfg = dict(batch_size=bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8,
               psi_zero=False, addr_dim=0, stdp=True, stdp_target="product", stdp_eta_init=0.1, stdp_lam_init=0.25, stdp_lam_fixed=1.0, amp=True)
    m = mod.LT(cfg).cuda(); m.load_state_dict(sd, strict=True); return m


def run_stdp(m, reset=False, eta_scale=None):
    inner = m.inner; r = Rec(); backup = inner.eta_raw.data.clone()
    if eta_scale is not None:
        e = (torch.sigmoid(inner.eta_raw) * eta_scale).clamp(1e-6, 1 - 1e-6); inner.eta_raw.data = torch.log(e / (1 - e))
    for b in range(0, N, args.bs):
        x = inp[b:b + args.bs]; n = len(x); h = inner.init_hidden.expand(n, 81, -1).clone(); w = None; P = None
        for s in range(MAX):
            if reset: w = None
            with torch.autocast("cuda", dtype=torch.bfloat16):                       # 하네스와 동일 (세그먼트 단위 autocast, 사이는 float)
                AB = inner.W_C(); kc = inner.kernel(); kcb = inner.kernel(inner.beta); inj = inner.injection(make_batch(x, x))
                for _ in range(K):
                    h = inner.boundary(h); h = h + inner.inj_gate * inj
                    h, w = inner.step(h, AB, kc, w, None, kcb)
                lg = inner.w_cls(h)
            h = h.float(); w = w.float(); P = r.add(s, b, n, lg, P)
    inner.eta_raw.data = backup
    return r.done()


res = {}; marks = [k for k in args.marks if k <= MAX]
def show(name, d):
    print(f"\n[{name}]  seg:   " + "".join(f"{k:>7d}" for k in marks))
    for key, fmt in (("exact", "d"), ("valid", "d"), ("ever", "d"), ("keep16", "d"), ("cell", ".4f"), ("loss", ".4f"), ("loss_bl", ".4f"), ("ce9", ".3f"), ("flip", ".4f")):
        if key in d: print(f"  {key:<6}       " + "".join(f"{d[key][k - 1]:>7{fmt}}" for k in marks))
    sys.stdout.flush()

mb = ms = None
for v in args.variants:
    t = time.time()
    if v.startswith("base"):
        if mb is None: mb = load_lt(args.base, bilinear=True, batch_size=args.bs)
        d = run_base(mb) if v == "base" else run_base(mb, 0.05, 0.25)
    else:
        if ms is None: ms = load_stdp(args.stdp, args.bs)
        d = run_stdp(ms, reset=(v == "stdp_reset"), eta_scale=(float(v[8:]) if v.startswith("stdp_eta") else None))
    d["sec"] = round(time.time() - t, 1); res[v] = d; show(v, d)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(dict(n=N, max=MAX, base=args.base, stdp=args.stdp, res=res), open(args.out, "w"))
print("saved", args.out)
