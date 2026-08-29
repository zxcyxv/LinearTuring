"""STDP 국소 갱신 Δw = η(Γ − w) 와 지도 손실의 결합-경사 −∇_w L 의 정렬 측정 (STDP.md §6.10).
  세그먼트 s 의 시작 carry (h, w) 에서: Δw 는 첫 블록의 Γ 로, ∇_w L 은 w 를 leaf 로 한 세그먼트 롤아웃의 CE 자동미분.
  칸별 정렬 = Σ_{h,n} Δw_tn·(−∇_w L)_tn, 최종 정답/오답 칸으로 분리. 대조 = 같은 크기의 무작위 방향.
  결과(4단계 @46,872, 512 퍼즐, s=8): 양의 정렬 비율 정답 .539 / 오답 .435 (무작위 .497/.505); 칸별 cos +.003 / −.004.
사용: python stdp_alignment.py <stdpp checkpoint> [N=512] [seg=8]"""
import sys, glob, torch, importlib.util, torch.nn.functional as F
from common import URM, load_test, make_batch; sys.path.insert(0, URM)
spec = importlib.util.spec_from_file_location("minimal", __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ck_path = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 512; SEG = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ck = torch.load(ck_path, map_location="cpu", weights_only=False); sd = ck.get("model_state_dict", ck)
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") or k.startswith("model.") else k): v for k, v in sd.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=1248, hidden_size=1248, psi_zero=True, addr_dim=416, stdp=True, stdp_target="product")).cuda()
m.load_state_dict(sd, strict=True); inner = m.inner
for p in m.parameters(): p.requires_grad_(False)
torch.manual_seed(0); inp, lab, _ = load_test(N); G = (lab.cuda() - 2).long(); bl = (inp == 1); eta = torch.sigmoid(inner.eta_raw)
def segment(h, w, inj, AB, kc, kcb):
    for _ in range(8): h = inner.boundary(h) + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, kcb)
    return h, w
out = {k: {"r": [], "w": []} for k in ("stdp", "random", "stdp_cos", "random_cos")}
for b in range(0, N, 128):
    x = inp[b:b + 128]; Gb = G[b:b + 128]; blb = bl[b:b + 128]
    with torch.no_grad():
        AB = inner.W_C(); kc = inner.kernel(); kcb = inner.kernel(inner.beta); inj = inner.injection(make_batch(x, x)); h = inner.init_hidden.expand(128, 81, -1).clone(); w = None
        for s in range(SEG): h, w = segment(h, w, inj, AB, kc, kcb)
        h0, w0 = h.clone(), w.clone()
        for s in range(16 - SEG): h, w = segment(h, w, inj, AB, kc, kcb)
        P = inner.w_cls(h)[:, :, 2:11].argmax(-1); right = (P == Gb) & blb; wrong = (P != Gb) & blb
        hb = inner.boundary(h0) + inner.inj_gate * inj; a = inner.attn(hb, AB, kc); hv = hb[..., 416:]
        vv = torch.einsum("btd,hcd->bthc", hv, inner.w_sh); vv = vv / (vv.norm(dim=-1, keepdim=True) + 1e-4)
        dw = eta * (a * torch.einsum("bthc,bnhc->bhtn", vv, vv) - w0); dr = torch.randn_like(dw) * dw.std()
    with torch.enable_grad():
        wl = w0.clone().requires_grad_(True); hs, _ = segment(h0, wl, inj, AB, kc, kcb); lg = inner.w_cls(hs)[:, :, 2:11]
        loss = F.cross_entropy(lg[blb], Gb[blb]); gw = torch.autograd.grad(loss, wl)[0]
    for name, d in (("stdp", dw), ("random", dr)):
        al = (d * (-gw)).sum(dim=(1, 3)); out[name]["r"].append(al[right].cpu()); out[name]["w"].append(al[wrong].cpu())
        dn = d.permute(0, 2, 1, 3).flatten(2); gn = (-gw).permute(0, 2, 1, 3).flatten(2); cs = F.cosine_similarity(dn, gn, dim=-1)
        out[name + "_cos"]["r"].append(cs[right].cpu()); out[name + "_cos"]["w"].append(cs[wrong].cpu())
for name in ("stdp", "random"):
    r = torch.cat(out[name]["r"]); w_ = torch.cat(out[name]["w"]); rc = torch.cat(out[name + "_cos"]["r"]); wc = torch.cat(out[name + "_cos"]["w"])
    print(f"[{name:6s}] 정답 칸 양의 정렬 {float((r > 0).float().mean()):.3f} (n={len(r)})  오답 칸 {float((w_ > 0).float().mean()):.3f} (n={len(w_)})  |  칸별 cos 평균: 정답 {rc.mean():+.4f}  오답 {wc.mean():+.4f}")
