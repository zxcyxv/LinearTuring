"""4단계(STDP) 체크포인트의 추론 시 유용성 — 학습 곡선이 아니라 기억의 존재 이유를 잰다.
  (a) 절제: λ 를 0/학습값/2배 로 바꿔 정확도 변화 (기억 의존도)
  (b) loops 외삽: 16/32/64 — 3단계는 포화, 기억이 있으면 계속 오르는가
  (c) 섭동 회복: 무효 판에서 정착-gradient 흔들기(selfgrad) 로 유효 격자 도달 수 — 4단계 vs 3단계
사용: python stdp_eval.py --stdp <ckpt.pt> --base <ckpt.pt> [--n 512] [--target product]"""
import argparse, sys, importlib.util, numpy as np, torch, torch.nn.functional as F
from common import URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "core", "minimal.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ap = argparse.ArgumentParser()
ap.add_argument("--stdp", required=True); ap.add_argument("--base", required=True); ap.add_argument("--n", type=int, default=512)
ap.add_argument("--target", default="product"); ap.add_argument("--hidden", type=int, default=1248); ap.add_argument("--addr", type=int, default=416)
ap.add_argument("--kick", type=int, default=64); ap.add_argument("--only_c", type=int, default=0)
args = ap.parse_args(); torch.set_grad_enabled(False)
inp, lab, _ = load_test(args.n); N = len(inp); G = lab.cuda() - 2; bl = (inp == 1); pm = torch.tensor(peer_mask(), device="cuda")

def load(path, stdp, bs=128):
    ck = torch.load(path, map_location="cpu", weights_only=False); sd = ck.get("model_state_dict", ck)
    sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") or k.startswith("model.") else k): v for k, v in sd.items()}
    cfg = dict(batch_size=bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=args.hidden, hidden_size=args.hidden,
               psi_zero=True, addr_dim=args.addr, stdp=stdp, stdp_target=args.target)
    m = mod.LT(cfg).cuda(); m.load_state_dict(sd, strict=True); return m

def rollout(m, x, loops=16, lam_scale=None, h0=None):
    inner = m.inner; AB = inner.W_C(); kc = inner.kernel(); kcb = inner.kernel(inner.beta) if inner.stdp else None
    inj = inner.injection(make_batch(x, x)); h = inner.init_hidden.expand(len(x), 81, -1).clone() if h0 is None else h0.clone(); w = None
    lam_raw_backup = None
    if inner.stdp and lam_scale is not None:
        lam_raw_backup = inner.lam_raw.data.clone()
        lam = torch.sigmoid(inner.lam_raw) * lam_scale; lam = lam.clamp(1e-6, 1 - 1e-6); inner.lam_raw.data = torch.log(lam / (1 - lam))
    for _ in range(loops):
        for _ in range(inner.config.blocks_per_seg):
            h = inner.boundary(h) + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, kcb)
    if lam_raw_backup is not None: inner.lam_raw.data = lam_raw_backup
    return h, inner.w_cls(h)[:, :, 2:11]

def score(P):
    fin = torch.where(bl, P, inp - 2); valid = ~((fin[:, :, None] == fin[:, None, :]) & pm[None]).any(-1).any(-1)
    return int(((P == G) | ~bl).all(-1).sum()), int(valid.sum()), round(float(((P == G) & bl).sum() / bl.sum()), 4)

def full(m, loops=16, lam_scale=None):
    P = torch.zeros(N, 81, dtype=torch.long, device="cuda")
    for b in range(0, N, 128): P[b:b + 128] = rollout(m, inp[b:b + 128], loops, lam_scale)[1].argmax(-1)
    return score(P)

ms = load(args.stdp, True); mb = load(args.base, False)
if not args.only_c: print(f"[기준] 3단계 {full(mb)}   4단계(학습 λ) {full(ms)}   (완답, 유효, 빈칸셀)")
if not args.only_c: print("(a) 절제  λ×0:", full(ms, lam_scale=0.0), "  λ×0.5:", full(ms, lam_scale=0.5), "  λ×2:", full(ms, lam_scale=2.0))
for L in ((32, 64) if not args.only_c else ()): print(f"(b) loops={L}  3단계 {full(mb, L)}   4단계 {full(ms, L)}")
# (c) 섭동 회복: 무효 판 --kick 개, 정착 목적 J 의 gradient 로 6스텝 흔들고 16 loop 재실행, 유효 격자 채택
def kick(m, idx):
    got = 0; torch.cuda.empty_cache()
    for i in idx:
        x = inp[i:i + 1]; blank = bl[i:i + 1]
        with torch.no_grad(): h0, _ = rollout(m, x)
        h = h0
        for s in range(6):
            with torch.enable_grad():
                hh = h.detach().requires_grad_(True); ps = []; hcur = hh
                for _ in range(2):
                    hcur, lg = rollout(m, x, 1, None, hcur); ps.append(F.softmax(lg.float(), -1))
                J = sum(((ps[k] - ps[k - 1]) ** 2).sum(-1)[blank].sum() for k in range(1, 2)) + (-(ps[-1] * torch.log(ps[-1] + 1e-9)).sum(-1))[blank].sum()
                g = torch.autograd.grad(J, hh)[0]
            d = g * blank.unsqueeze(-1); h = (h - d / (d.norm() + 1e-9) * h.norm() * 0.1).detach()
            P = rollout(m, x, 16, None, h)[1].argmax(-1); fin = torch.where(blank, P, x - 2)
            if not ((fin[:, :, None] == fin[:, None, :]) & pm[None]).any(): got += 1; break
    return got
Pb = torch.zeros(N, 81, dtype=torch.long, device="cuda")
for b in range(0, N, 128): Pb[b:b + 128] = rollout(mb, inp[b:b + 128])[1].argmax(-1)
finb = torch.where(bl, Pb, inp - 2); invalid_b = ((finb[:, :, None] == finb[:, None, :]) & pm[None]).any(-1).any(-1)
idx = torch.where(invalid_b)[0][:args.kick].tolist()
ms1 = load(args.stdp, True, 1); mb1 = load(args.base, False, 1)
for mm in (ms1, mb1):
    for p in mm.parameters(): p.requires_grad_(False)
print(f"(c) 섭동 회복 (3단계 기준 무효 판 {len(idx)}개, 6스텝 흔들기): 3단계 {kick(mb1, idx)}  4단계 {kick(ms1, idx)}")
