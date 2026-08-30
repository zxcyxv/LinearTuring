"""한 퍼즐·한 칸이 받는 메시지를 보낸 칸별로 분해 (stdp1, w 세그먼트 초기화 모드, 규칙 없음).
  메시지 f_tn = Σ_h w_h,tn · WᵀW_h v_h,n 은 은닉 벡터. 읽기(로짓)가 선형이라 숫자 d 의 로짓 변화 = W_cls[d]·f_tn 으로 정확히 분해된다.
  출력: 대상 칸의 현재 로짓, 피어/비피어 합계, 기여 큰 칸 목록 (보낸 칸의 현재 숫자, 피어 여부, 숫자별 로짓 변화)
사용: python analysis/msg_decompose.py [퍼즐 idx=57] [칸들=2,4] [세그먼트=16]"""
import sys, os, importlib.util, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis")); from common import ROOT, URM, load_test, make_batch, peer_mask
sys.path.insert(0, URM); spec = importlib.util.spec_from_file_location("minimal", os.path.join(ROOT, "core", "minimal.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.set_grad_enabled(False); PZ = int(sys.argv[1]) if len(sys.argv) > 1 else 57; CELLS = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else "2,4").split(",")]; SEG = int(sys.argv[3]) if len(sys.argv) > 3 else 16
inp, lab, _ = load_test(512); G = (lab - 2).long(); bl = inp == 1; pm = peer_mask()
ck = torch.load(os.path.join(ROOT, "checkpoints", "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
sd = {(k.replace("_orig_mod.", "").replace("model.", "", 1) if k.startswith("_orig_mod.model.") else k): v for k, v in ck.items()}
m = mod.LT(dict(batch_size=128, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832, hidden_size=832, num_heads=8, stdp=True, stdp_target="product", stdp_lam_fixed=1.0, amp=True)).cuda(); m.load_state_dict(sd, strict=True); inner = m.inner
eps = inner.config.eps; b0 = (PZ // 128) * 128; x = inp[b0:b0 + 128]; i = PZ - b0; h = inner.init_hidden.expand(128, 81, -1).clone()
for s in range(SEG):                                                     # 평소 추론 (reset 모드)
    w = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        AB = inner.W_C(); kc = inner.kernel(); inj = inner.injection(make_batch(x, x))
        for _ in range(8):
            h = inner.boundary(h); h = h + inner.inj_gate * inj; h, w = inner.step(h, AB, kc, w, None, None)
    h = h.float()
# SEG 뒤 상태에서 다음 블록의 메시지 (경계 → 주입 → 어텐션/값), float32 로 정확히
P = inner.w_cls(h)[:, :, 2:11].argmax(-1)                                   # 현재 예측
hp = inner.boundary(h) + inner.inj_gate * inner.injection(make_batch(x, x))
AB = inner.W_C(); kc = inner.kernel(); a = inner.attn(hp, AB, kc)            # [B,H,T,T]
vv = torch.einsum('btd,hcd->bthc', hp, inner.w_sh); vn = vv / (vv.norm(dim=-1, keepdim=True) + eps)
w = a * torch.einsum('bthc,bnhc->bhtn', vn, vn)                              # reset 모드 첫 블록: w = a·⟨v̂,v̂⟩
Wc = inner.w_cls.weight[2:11]                                                # [9, d] 숫자 1..9 읽기
rc = lambda c: f"{c//9+1}행{c%9+1}열"
for t in CELLS:
    # 보낸 칸 n 별 메시지 벡터 f_tn = Σ_h w_h,tn (Wᵀ_h W_h v_n)
    o = torch.einsum('hn,nhc->nhc', w[i, :, t, :], vv[i])                     # [T,H,C]  w_h,tn · W_h v_n
    f = torch.einsum('nhc,hcd->nd', o, inner.w_sh)                            # [T,d]
    dl = f @ Wc.T                                                             # [T,9] 보낸 칸별 숫자 로짓 변화
    cur = (inner.w_cls(h)[i, t, 2:11]).cpu().numpy(); tot = dl.sum(0).cpu().numpy()
    peer = torch.tensor(pm[t], device="cuda"); dlp = dl[peer].sum(0).cpu().numpy(); dln = dl[~peer & (torch.arange(81, device="cuda") != t)].sum(0).cpu().numpy(); dls = dl[t].cpu().numpy()
    fmt = lambda v: " ".join(f"{v[d]:+6.2f}" for d in range(9))
    print(f"\n===== 퍼즐 {PZ} 칸 {t} ({rc(t)}): 정답 {int(G[PZ,t])+1}, 현재 예측 {int(P[i,t])+1}   (숫자 1..9 순)")
    print(f"  현재 로짓            {fmt(cur)}")
    print(f"  메시지 합계          {fmt(tot)}")
    print(f"   피어 {int(peer.sum())}칸 합계    {fmt(dlp)}")
    print(f"   비피어 {81-1-int(peer.sum())}칸 합계  {fmt(dln)}")
    print(f"   자기 자신           {fmt(dls)}")
    d7, d9 = int(G[PZ, t]), int(P[i, t])                                      # 정답 숫자, 현재(틀린) 숫자 인덱스
    key = (dl[:, d9] - dl[:, d7]).abs(); top = torch.argsort(key, descending=True)[:14].tolist()
    print(f"  기여 큰 칸 (|Δ로짓[{d9+1}] − Δ로짓[{d7+1}]| 순):   칸(위치)  피어?  그 칸의 숫자  Δ로짓[{d7+1}] Δ로짓[{d9+1}]   헤드별 w_tn(0..7)")
    for n in top:
        ws = " ".join(f"{float(w[i,hh,t,n]):+.2f}" for hh in range(8))
        print(f"    {n:2d}({rc(n)})  {'피어' if pm[t,n] else '  - '}   {int(P[i,n])+1}{'*' if (P[i,n]!=G[PZ,n] and bl[PZ,n]) else ' '}{'(주어짐)' if not bl[PZ,n] else '      '}   {float(dl[n,d7]):+6.2f}  {float(dl[n,d9]):+6.2f}    {ws}")
