# -*- coding: utf-8 -*-
"""LinearTuring v1.5 — v1(9/1) 어텐션 + v4 관례. 순수 동역학 (하네스·데이터 인터페이스 없음).

v1 에서 그대로 가져온 것 (어텐션·가소성 — 건드리지 않음):
    거리 감쇠   D(Δ) = e^{−α_h‖Δ‖₁},  α 헤드별 학습
    읽기 커널   a   = D(Δ)·Σ_j r_tj r_nj cos(Δφ_j + θ_j·Δ + ψ_j)
    쓰기 창     a_β = D(Δ)·Σ_j r_tj r_nj cos(Δφ_j + θ_j·Δ + β_j)      ψ 와 별개 위상
    쓰기 목표   G = a_β · agree,   agree = ⟨v̂_t, v̂_n⟩ (값공간 코사인)
    w 갱신      w ← (1−δ)w + δ·gain·G,   gain 단일 [H,1,1] (v4 의 대각/비대각 분리 안 씀)
    읽기        a_eff = (1−λ)a + λw

v4 로 바꾼 것 (어텐션 밖 — 관례를 맞춤):
    블록 순서   pre → **post**:  주입 → (어텐션+수송) → 경계 → Φ
                v1 은 경계 → 주입 → (어텐션+수송+Φ) 라 판독이 어텐션 출력 직후였다
    주입 계수   학습 파라미터(0.25→7.60) → **상수 √d**
    γ (Φ 반경)  학습(0.1→9.2e-5, 반경 104) → **상수 1/d (반경 √d)**
    임베딩 초기화  파이토치 기본 N(0,1) → **trunc_normal(std=1/√d)**

주의: 게이지가 바뀌어 v1 체크포인트가 넘어오지 않는다. 경계가 쌍선형(2차)이라
      게이지 등변이 아니고(수송·Φ·주입은 1차), 단순 재스케일로 흡수되지 않는다. 재학습 필요.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(y):  return math.log(math.expm1(y))
def inv_sigmoid(y):   return math.log(y / (1 - y))


class LTLayer(nn.Module):
    """블록 한 벌의 파라미터. 어텐션·가소성 파라미터는 v1 그대로."""

    def __init__(self, d, H, dh, p, mlp_expansion=4.0,
                 alpha_init=0.1, eta_init=0.05, lam_init=0.25, gain_init=1.0):
        super().__init__()
        self.wc_raw    = nn.Parameter(torch.randn(H, dh, d) / math.sqrt(d))         # 복소 주소 투영
        self.psi       = nn.Parameter(torch.rand(H, p) * 2 * math.pi - math.pi)     # 읽기 위상
        self.theta     = nn.Parameter((torch.rand(H, p, 2) * 2 - 1) * (math.pi / 2))# 2D 파수
        self.alpha_raw = nn.Parameter(torch.full((H, 1), inv_softplus(alpha_init))) # 거리 감쇠율 (v1)
        w_sh = torch.zeros(H, dh, d)
        for m in range(H):
            w_sh[m, :, m * dh:(m + 1) * dh] = torch.eye(dh)
        self.w_sh = nn.Parameter(w_sh + 0.01 * torch.randn(H, dh, d) / math.sqrt(d))
        self.eta_raw  = nn.Parameter(torch.full((H, 1, 1), inv_sigmoid(eta_init)))   # δ
        self.lam_raw  = nn.Parameter(torch.full((H, 1, 1), inv_sigmoid(lam_init)))   # λ
        self.gain_raw = nn.Parameter(torch.full((H, 1, 1), inv_softplus(gain_init))) # G — v1 단일
        self.beta     = nn.Parameter(torch.zeros(H, p))                              # 쓰기 창 위상
        self.beta.data.normal_(0.0, 0.5)
        inter = int(mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.b_down    = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()                                               # 시작 시 항등

    @property
    def alpha(self): return F.softplus(self.alpha_raw)


class LT15(nn.Module):
    def __init__(self, d=832, H=8, grid=9, blocks_per_seg=8, vocab_size=11, eps=1e-4):
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
                             persistent=False)                      # ‖Δ‖₁ [T,T]

        # ---- v4 관례: 게이지 상수화. 주입 √d, Φ 반경 √d (γ = 1/d)
        self.gamma       = 1.0 / d
        self.embed_scale = math.sqrt(d)
        self.embed = nn.Embedding(vocab_size, d)
        with torch.no_grad():
            nn.init.trunc_normal_(self.embed.weight, std=1.0 / self.embed_scale, a=-2.0, b=2.0)

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
        """v1 거리 감쇠 D(Δ) = e^{−α_h‖Δ‖₁}  [H,T,T]."""
        return torch.exp(-L.alpha[:, 0, None, None] * self.l1)

    def kernel(self, L, phase):
        """위상각 A_t = phase/2 + θ·pos_t (q),  B_t = −phase/2 + θ·pos_t (k) → cos/sin [T,H,p].
        phase=ψ 면 읽기 커널, phase=β 면 쓰기 창."""
        ppos = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w   # [H,p,T]
        A = (ppos + phase[..., None] / 2).permute(2, 0, 1)
        B = (ppos - phase[..., None] / 2).permute(2, 0, 1)
        return torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)

    def addr(self, h, AB):
        """정규화된 복소 주소 ẑ = (x + iy). 두 커널이 공유한다 (블록당 1회)."""
        A, Bm = AB
        x = torch.einsum('btd,hjd->bthj', h, A)
        y = torch.einsum('btd,hjd->bthj', h, Bm)
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()
        return x / (nrm + self.eps), y / (nrm + self.eps)

    def attn_xy(self, xy, kc, D):
        """a = D(Δ) · Re[conj(q̂_t) k̂_n]  [B,H,T,T]."""
        x, y = xy; cosA, sinA, cosB, sinB = kc
        qx = x * cosA - y * sinA; qy = x * sinA + y * cosA
        kx = x * cosB - y * sinB; ky = x * sinB + y * cosB
        a = (torch.einsum('bthj,bnhj->bhtn', qx, kx)
           + torch.einsum('bthj,bnhj->bhtn', qy, ky))
        return a * D.unsqueeze(0)

    def phi(self, h):
        """4차 소산의 닫힌 해. γ = 1/d 상수 → 반경 √d (v4 관례)."""
        return h / torch.sqrt(1.0 + self.gamma * h.pow(2).sum(-1, keepdim=True))

    def boundary(self, L, h):
        """토큰 내 쌍선형 (활성화 0개). W_d 영 초기화 → 시작 시 항등."""
        g, u = L.b_gate_up(h).chunk(2, dim=-1)
        return h + L.b_down(0.5 * g * u)

    def step(self, L, h, AB, kc, kcb, D, w):
        """v1 어텐션·가소성 그대로. Φ 는 여기 없다 — post 순서라 블록 끝에서 건다."""
        xy = self.addr(h, AB)
        a  = self.attn_xy(xy, kc, D)                                  # 읽기 커널 (ψ)
        v  = torch.einsum('btd,hcd->bthc', h, L.w_sh)

        # ---- 쓰기: 창(β) × 동시활동(agree)
        vv    = v / (v.norm(dim=-1, keepdim=True) + self.eps)
        agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
        a_b   = self.attn_xy(xy, kcb, D)                              # 쓰기 창 a_β (주소 사영 재사용)
        tgt   = F.softplus(L.gain_raw) * (a_b * agree)                # gain 단일
        eta   = torch.sigmoid(L.eta_raw)
        w = tgt if w is None else (1 - eta) * w + eta * tgt

        # ---- 읽기: 덧셈
        lam   = torch.sigmoid(L.lam_raw)
        a_eff = (1 - lam) * a + lam * w

        o = torch.einsum('bhtn,bnhc->bthc', a_eff, v)
        f = torch.einsum('bthc,hcd->btd', o, L.w_sh)
        return h + f, w

    # ---------------------------------------------------------------- 세그먼트 1회
    def forward(self, tokens, h=None, w=None):
        """tokens: [B,T] long (0 PAD, 1 빈칸, 2..10 = 숫자 1..9)"""
        B = tokens.shape[0]
        if h is None:
            h = self.init_hidden.expand(B, self.T, self.d).contiguous()
        inj = self.embed(tokens.long())

        L   = self.layer
        AB  = self.W_C(L)
        D   = self.decay(L)
        kc  = self.kernel(L, L.psi)      # 읽기
        kcb = self.kernel(L, L.beta)     # 쓰기

        for _ in range(self.blocks_per_seg):            # ---- post 순서 (v4)
            h    = h + self.embed_scale * inj           # ① 주입 (상수 √d)
            h, w = self.step(L, h, AB, kc, kcb, D, w)   # ② 어텐션 · 쓰기 · 읽기 · 수송
            h    = self.boundary(L, h)                  # ③ 경계 (쌍선형)
            h    = self.phi(h)                          # ④ Φ — 판독·carry 가 흡수구 안

        return self.w_cls(h), h, w


if __name__ == "__main__":
    torch.manual_seed(0)
    m = LT15()
    tokens = torch.randint(1, 11, (2, m.T))
    logits, h, w = m(tokens)

    print(f"파라미터           {sum(p.numel() for p in m.parameters()):,}")
    print(f"logits             {tuple(logits.shape)}")
    print(f"h                  {tuple(h.shape)}   ‖h_t‖ 평균 {h.norm(dim=-1).mean():.2f}"
          f"   (Φ 반경 √d = {math.sqrt(m.d):.2f})")
    print(f"w                  {tuple(w.shape)}   |w| 평균 {w.abs().mean():.4f}")
    print(f"게이지             주입 {m.embed_scale:.2f} (상수)   γ {m.gamma:.3e} (상수)")

    with torch.no_grad():
        L, AB, D = m.layer, m.W_C(m.layer), m.decay(m.layer)
        xy = m.addr(h, AB)
        a   = m.attn_xy(xy, m.kernel(L, L.psi),  D)
        a_b = m.attn_xy(xy, m.kernel(L, L.beta), D)
        print(f"max|a − aᵀ|        {(a - a.transpose(-1,-2)).abs().max():.3f}   ← 읽기: ψ 있어 비대칭")
        print(f"max|a_β − a_βᵀ|    {(a_b - a_b.transpose(-1,-2)).abs().max():.3f}   ← 쓰기: β 있어 비대칭")
        al = L.alpha.flatten()
        print(f"D(Δ) |Δ|=1/8/16    {torch.exp(-al[0]*1):.3f} / {torch.exp(-al[0]*8):.3f} / {torch.exp(-al[0]*16):.3f}")
        print(f"a 범위             [{a.min():.3f}, {a.max():.3f}]")

    logits2, h2, w2 = m(tokens, h.detach(), w.detach())
    print(f"세그먼트 2         변화량 {(logits2-logits).abs().mean():.4f}")
