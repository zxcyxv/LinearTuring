# -*- coding: utf-8 -*-
"""LinearTuring v1.1 — 9/1 판(R1B8_min_faith)에서 ψ 만 제거. 순수 동역학 (하네스·데이터 인터페이스 없음).

원본 v1 설정 (2026-09-06/analysis/train_0901.py 의 CFG):
    dist_decay=True, alpha_init=0.1        거리 감쇠 e^{−α_h‖Δ‖₁} 있음
    legacy_gauge=True                      embed_scale·γ 가 학습 파라미터 (0.25 / 0.1)
    block_order="pre"                      경계 → 주입 → 스텝(수송+Φ)
    stdp_target="faithful"                 쓰기 목표 G = a_β · agree
    stdp_window="beta"                     쓰기 커널을 별도 위상 β 로 다시 뜬다 (읽기 ψ 와 별개)
    stdp_read="add"                        읽기 a_eff = (1−λ)a + λw
    stdp_diag="keep"                       자기시냅스 유지
    num_layers=1, blocks_per_seg=8

v1.1 의 변경 (한 가지):
    읽기 커널의 ψ 제거.   a_tn = D(Δ)·Σ_j r r cos(Δφ_j + θ_j·Δ + ψ_j) → D(Δ)·Σ_j r r cos(Δφ_j + θ_j·Δ)
    → 읽기 a 가 대칭이 된다 (감쇠 D 도 대칭이므로).  쓰기 커널의 β 는 그대로 둔다.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(y):  return math.log(math.expm1(y))
def inv_sigmoid(y):   return math.log(y / (1 - y))


class LTLayer(nn.Module):
    """블록 한 벌의 파라미터. (v1 의 LTLayer 에서 psi 만 삭제)"""

    def __init__(self, d, H, dh, p, mlp_expansion=4.0,
                 alpha_init=0.1, eta_init=0.05, lam_init=0.25, gain_init=1.0):
        super().__init__()
        self.wc_raw    = nn.Parameter(torch.randn(H, dh, d) / math.sqrt(d))
        self.theta     = nn.Parameter((torch.rand(H, p, 2) * 2 - 1) * (math.pi / 2))
        self.alpha_raw = nn.Parameter(torch.full((H, 1), inv_softplus(alpha_init)))   # 거리 감쇠율
        w_sh = torch.zeros(H, dh, d)
        for m in range(H):
            w_sh[m, :, m * dh:(m + 1) * dh] = torch.eye(dh)
        self.w_sh = nn.Parameter(w_sh + 0.01 * torch.randn(H, dh, d) / math.sqrt(d))
        # 가소성
        self.eta_raw  = nn.Parameter(torch.full((H, 1, 1), inv_sigmoid(eta_init)))    # δ
        self.lam_raw  = nn.Parameter(torch.full((H, 1, 1), inv_sigmoid(lam_init)))    # λ
        self.gain_raw = nn.Parameter(torch.full((H, 1, 1), inv_softplus(gain_init)))  # G
        self.beta     = nn.Parameter(torch.zeros(H, p))                               # 쓰기 창의 위상
        self.beta.data.normal_(0.0, 0.5)          # faithful + window=beta 면 N(0,0.5) 초기화
        # 경계
        inter = int(mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.b_down    = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()

    @property
    def alpha(self): return F.softplus(self.alpha_raw)


class LT11(nn.Module):
    def __init__(self, d=832, H=8, grid=9, blocks_per_seg=8, vocab_size=11, eps=1e-4,
                 inj_gate_init=0.25, gamma_init=0.1):
        super().__init__()
        T = grid * grid
        assert d % H == 0 and (d // H) % 2 == 0
        self.d, self.H, self.T, self.blocks_per_seg, self.eps = d, H, T, blocks_per_seg, eps
        self.dh = d // H
        self.p  = self.dh // 2

        u = torch.arange(T).float() // grid
        w = torch.arange(T).float() %  grid
        self.register_buffer("pos_u", u, persistent=False)
        self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("l1", (u[:, None] - u[None]).abs() + (w[:, None] - w[None]).abs(),
                             persistent=False)                      # ‖Δ‖₁  [T,T]

        # legacy_gauge=True — 게이지가 학습된다 (v4 는 상수 √d / 1/d)
        self.embed      = nn.Embedding(vocab_size, d)               # 파이토치 기본 초기화 N(0,1)
        self.embed_scale = nn.Parameter(torch.tensor(float(inj_gate_init)))
        self.gamma_raw   = nn.Parameter(torch.tensor(inv_softplus(gamma_init)))

        self.w_cls = nn.Linear(d, vocab_size)
        self.layer = LTLayer(d, H, self.dh, self.p)
        self.register_buffer("init_hidden",
                             nn.init.trunc_normal_(torch.empty(d), std=1.0, a=-2.0, b=2.0),
                             persistent=True)

    # ---------------------------------------------------------------- 부품
    def W_C(self, L):
        Q, _ = torch.linalg.qr(L.wc_raw.transpose(-1, -2))
        AB = Q.transpose(-1, -2)
        return AB[:, :self.p, :], AB[:, self.p:, :]

    def decay(self, L):
        """거리 감쇠 D(Δ) = e^{−α_h‖Δ‖₁}  [H,T,T].  v1 은 dist_decay=True."""
        return torch.exp(-L.alpha[:, 0, None, None] * self.l1)

    def kernel(self, L, phase=None):
        """위상각 → cos/sin [T,H,p].
        phase=None : 읽기 커널 (v1.1 — ψ 제거, q·k 가 같은 회전)
        phase=β    : 쓰기 커널 (q 에 +β/2, k 에 −β/2)"""
        ppos = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w   # [H,p,T]
        if phase is None:
            P = ppos.permute(2, 0, 1)
            c, s = torch.cos(P), torch.sin(P)
            return c, s, c, s                                        # A = B
        A = (ppos + phase[..., None] / 2).permute(2, 0, 1)
        B = (ppos - phase[..., None] / 2).permute(2, 0, 1)
        return torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)

    def addr(self, h, AB):
        A, Bm = AB
        x = torch.einsum('btd,hjd->bthj', h, A)
        y = torch.einsum('btd,hjd->bthj', h, Bm)
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()
        return x / (nrm + self.eps), y / (nrm + self.eps)

    def attn_xy(self, xy, kc, D):
        """a = D(Δ) · Re[conj(q̂_t) k̂_n]   [B,H,T,T]"""
        x, y = xy; cosA, sinA, cosB, sinB = kc
        qx = x * cosA - y * sinA; qy = x * sinA + y * cosA
        kx = x * cosB - y * sinB; ky = x * sinB + y * cosB
        a = (torch.einsum('bthj,bnhj->bhtn', qx, kx)
           + torch.einsum('bthj,bnhj->bhtn', qy, ky))
        return a * D.unsqueeze(0)

    def phi(self, h):
        g = F.softplus(self.gamma_raw)                               # γ 학습됨
        return h / torch.sqrt(1.0 + g * h.pow(2).sum(-1, keepdim=True))

    def boundary(self, L, h):
        g, u = L.b_gate_up(h).chunk(2, dim=-1)
        return h + L.b_down(0.5 * g * u)

    def step(self, L, h, AB, kc, kcb, D, w):
        """faithful STDP: 쓰기 G = a_β·agree, 읽기 a_eff = (1−λ)a + λw. Φ 는 step 끝(block_order=pre)."""
        xy = self.addr(h, AB)
        a  = self.attn_xy(xy, kc, D)                                 # 읽기 커널 (ψ 없음 → 대칭)
        v  = torch.einsum('btd,hcd->bthc', h, L.w_sh)

        # ---- 쓰기: 창(β) × 동시활동(agree)
        vv    = v / (v.norm(dim=-1, keepdim=True) + self.eps)
        agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)              # 값 공간 코사인
        a_b   = self.attn_xy(xy, kcb, D)                             # 쓰기 창 a_β (주소 사영 재사용)
        G     = a_b * agree
        tgt   = F.softplus(L.gain_raw) * G
        eta   = torch.sigmoid(L.eta_raw)
        w = tgt if w is None else (1 - eta) * w + eta * tgt

        # ---- 읽기: 덧셈
        lam   = torch.sigmoid(L.lam_raw)
        a_eff = (1 - lam) * a + lam * w

        o = torch.einsum('bhtn,bnhc->bthc', a_eff, v)
        f = torch.einsum('bthc,hcd->btd', o, L.w_sh)
        return self.phi(h + f), w                                    # pre 순서: Φ 가 step 안

    # ---------------------------------------------------------------- 세그먼트 1회
    def forward(self, tokens, h=None, w=None):
        """tokens: [B,T] long.  반환 logits [B,T,vocab], h, w"""
        B = tokens.shape[0]
        if h is None:
            h = self.init_hidden.expand(B, self.T, self.d).contiguous()
        inj = self.embed(tokens.long())

        L   = self.layer
        AB  = self.W_C(L)
        D   = self.decay(L)
        kc  = self.kernel(L)                 # 읽기 — ψ 없음
        kcb = self.kernel(L, L.beta)         # 쓰기 — β

        for _ in range(self.blocks_per_seg):
            h    = self.boundary(L, h)                               # ① 경계 (pre 순서: 맨 앞)
            h    = h + self.embed_scale * inj                        # ② 주입 (학습 게이트)
            h, w = self.step(L, h, AB, kc, kcb, D, w)                # ③ 어텐션·쓰기·읽기·수송·Φ

        return self.w_cls(h), h, w


if __name__ == "__main__":
    torch.manual_seed(0)
    m = LT11()
    tokens = torch.randint(1, 11, (2, m.T))
    logits, h, w = m(tokens)

    print(f"파라미터           {sum(p.numel() for p in m.parameters()):,}")
    print(f"logits             {tuple(logits.shape)}")
    print(f"h                  {tuple(h.shape)}   ‖h_t‖ 평균 {h.norm(dim=-1).mean():.2f}"
          f"   (Φ 반경 γ^-½ = {F.softplus(m.gamma_raw).pow(-0.5).item():.2f})")
    print(f"w                  {tuple(w.shape)}   |w| 평균 {w.abs().mean():.4f}")
    print(f"embed_scale(학습)  {m.embed_scale.item():.3f}     γ(학습) {F.softplus(m.gamma_raw).item():.4f}")

    with torch.no_grad():
        L, AB, D = m.layer, m.W_C(m.layer), m.decay(m.layer)
        xy = m.addr(h, AB)
        a   = m.attn_xy(xy, m.kernel(L), D)             # 읽기 (ψ 없음)
        a_b = m.attn_xy(xy, m.kernel(L, L.beta), D)     # 쓰기 (β)
        print(f"max|a − aᵀ|        {(a - a.transpose(-1,-2)).abs().max():.2e}   ← 읽기: ψ 없어 대칭")
        print(f"max|a_β − a_βᵀ|    {(a_b - a_b.transpose(-1,-2)).abs().max():.2e}   ← 쓰기: β 살아 있어 비대칭")
        al = L.alpha.flatten()
        print(f"α 헤드별           {[round(v,3) for v in al.tolist()]}")
        print(f"D(Δ) |Δ|=1/8/16    {torch.exp(-al[0]*1):.3f} / {torch.exp(-al[0]*8):.3f} / {torch.exp(-al[0]*16):.3f}")
        print(f"a 범위             [{a.min():.3f}, {a.max():.3f}]   (감쇠가 대각 외를 눌러 |a|<1)")
