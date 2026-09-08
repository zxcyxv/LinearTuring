# -*- coding: utf-8 -*-
"""LinearTuring v4.1 — ψ 제거판. 순수 동역학만 (하네스·데이터 인터페이스 없음).

v4 와의 차이는 한 가지:
    v4    q̂ = ẑ_t·e^{+iψ/2},  k̂ = ẑ_n·e^{−iψ/2}   →  a_tn = Σ_j r r cos(Δφ_j + θ_j·Δ + ψ_j)
    v4.1  q̂ = ẑ_t,            k̂ = ẑ_n             →  a_tn = Σ_j r r cos(Δφ_j + θ_j·Δ)

  결과: q 와 k 가 같은 회전을 받으므로 커널 한 벌만 계산하면 되고,
        a 가 구조적으로 대칭이 된다 (a_tn = a_nt).  [THEORY.md 명제 7]

한 세그먼트 = [ 주입 → step(쓰기·읽기·수송) → 경계 → Φ ] × 8 블록, 가중치 공유.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(y):  return math.log(math.expm1(y))
def inv_sigmoid(y):   return math.log(y / (1 - y))


class LTLayer(nn.Module):
    """블록 한 벌의 파라미터. (v4 의 LTLayer 에서 psi 만 삭제)"""

    def __init__(self, d, H, dh, p, mlp_expansion=4.0, eta_init=0.05, lam_init=0.25, gain_init=1.0):
        super().__init__()
        # 복소 주소 투영 (forward 에서 QR 로 행직교화해서 씀)
        self.wc_raw = nn.Parameter(torch.randn(H, dh, d) / math.sqrt(d))
        # 2D 파수 — 위치 위상 θ·pos 를 만든다
        self.theta = nn.Parameter((torch.rand(H, p, 2) * 2 - 1) * (math.pi / 2))
        # sheaf 값 수송, 블록 항등 초기화
        w_sh = torch.zeros(H, dh, d)
        for m in range(H):
            w_sh[m, :, m * dh:(m + 1) * dh] = torch.eye(dh)
        self.w_sh = nn.Parameter(w_sh + 0.01 * torch.randn(H, dh, d) / math.sqrt(d))
        # 가소성 스칼라 (헤드별)
        self.eta_raw      = nn.Parameter(torch.full((H, 1, 1), inv_sigmoid(eta_init)))    # δ = sigmoid
        self.lam_raw      = nn.Parameter(torch.full((H, 1, 1), inv_sigmoid(lam_init)))    # λ = sigmoid
        self.gain_raw     = nn.Parameter(torch.full((H, 1, 1), inv_softplus(gain_init)))  # g 비대각
        self.gain_diag_raw= nn.Parameter(torch.full((H, 1, 1), inv_softplus(gain_init)))  # g 대각
        # 경계: 토큰 내 쌍선형 (활성화 0개), W_d 영 초기화 → 시작 시 항등
        inter = int(mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.b_down    = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()


class LT41(nn.Module):
    def __init__(self, d=832, H=8, grid=9, blocks_per_seg=8, vocab_size=11, eps=1e-4):
        super().__init__()
        T = grid * grid
        assert d % H == 0 and (d // H) % 2 == 0
        self.d, self.H, self.T, self.blocks_per_seg, self.eps = d, H, T, blocks_per_seg, eps
        self.dh = d // H
        self.p  = self.dh // 2                      # 복소 성분 개수

        # 2D 위치 (행, 열)
        u = torch.arange(T).float() // grid
        w = torch.arange(T).float() %  grid
        self.register_buffer("pos_u", u, persistent=False)
        self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("eye", torch.eye(T), persistent=False)

        # 게이지 상수: 주입 √d, Φ 반경 √d (γ = 1/d)
        self.gamma        = 1.0 / d
        self.embed_scale  = math.sqrt(d)

        self.embed  = nn.Embedding(vocab_size, d)
        with torch.no_grad():
            nn.init.trunc_normal_(self.embed.weight, std=1.0 / self.embed_scale, a=-2.0, b=2.0)
        self.w_cls  = nn.Linear(d, vocab_size)
        self.layer  = LTLayer(d, H, self.dh, self.p)
        self.register_buffer("init_hidden",
                             nn.init.trunc_normal_(torch.empty(d), std=1.0, a=-2.0, b=2.0),
                             persistent=True)

    # ---------------------------------------------------------------- 부품
    def W_C(self, L):
        """행직교 [A;B] (헤드별 dh×d) → 복소 주소의 실부/허부 투영 두 벌."""
        Q, _ = torch.linalg.qr(L.wc_raw.transpose(-1, -2))       # [H,d,dh]
        AB = Q.transpose(-1, -2)                                  # [H,dh,d]
        return AB[:, :self.p, :], AB[:, self.p:, :]               # 각 [H,p,d]

    def kernel(self, L):
        """위치 위상 θ_j·pos_t → cos/sin [T,H,p].
        v4 는 q 에 +ψ/2, k 에 −ψ/2 를 더해 두 벌을 만들었다. v4.1 은 ψ 가 없어 한 벌."""
        ppos = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w   # [H,p,T]
        P = ppos.permute(2, 0, 1)                                                        # [T,H,p]
        return torch.cos(P), torch.sin(P)

    def addr(self, h, AB):
        """정규화된 복소 주소 ẑ = (x + iy),  성분별 크기 r_j, Σ_j r_j² = 1."""
        A, Bm = AB
        x = torch.einsum('btd,hjd->bthj', h, A)
        y = torch.einsum('btd,hjd->bthj', h, Bm)
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()
        return x / (nrm + self.eps), y / (nrm + self.eps)

    def attn(self, h, AB, kc):
        """a_tn = Re[ p_t · conj(p_n) ],  p_t = ẑ_t · e^{i θ·pos_t}
             = Σ_j r_tj r_nj cos( (φ_tj − φ_nj) + θ_j·Δ_tn )        [B,H,T,T]
        ψ 가 없으므로 q 와 k 가 같은 회전을 받는다 → a 는 대칭 (a_tn = a_nt)."""
        x, y = self.addr(h, AB)
        cosP, sinP = kc
        px = x * cosP - y * sinP
        py = x * sinP + y * cosP
        return (torch.einsum('bthj,bnhj->bhtn', px, px)
              + torch.einsum('bthj,bnhj->bhtn', py, py))

    def phi(self, h):
        """4차 소산 ḣ = −γ‖h‖²h 의 닫힌 해. 반경 √d 로 정착."""
        return h / torch.sqrt(1.0 + self.gamma * h.pow(2).sum(-1, keepdim=True))

    def boundary(self, L, h):
        """토큰 내 쌍선형 사상 (활성화 0개). W_d 영 초기화 → 시작 시 항등."""
        g, u = L.b_gate_up(h).chunk(2, dim=-1)
        return h + L.b_down(0.5 * g * u)

    def step(self, L, h, AB, kc, w):
        """한 블록의 어텐션 · 쓰기 · 읽기 · 수송."""
        a = self.attn(h, AB, kc)                                     # 순간 시냅스 K_fast
        v = torch.einsum('btd,hcd->bthc', h, L.w_sh)                 # sheaf 값

        # 쓰기: 느린 시냅스 K_slow = LPF_δ(a).  g 는 대각(자기 관성)·비대각(쌍 결합) 따로
        gain = (F.softplus(L.gain_raw)      * (1 - self.eye)
              + F.softplus(L.gain_diag_raw) * self.eye)              # [H,T,T]
        tgt  = gain.unsqueeze(0) * a
        eta  = torch.sigmoid(L.eta_raw)
        w = tgt if w is None else (1 - eta) * w + eta * tgt           # K̇ = ε(G − K), dt=1

        # 읽기: (K_fast + K_slow) 의 합
        lam   = torch.sigmoid(L.lam_raw)
        a_eff = (1 - lam) * a + lam * w

        # 수송: h ← h + Wᵀ Σ_n a_eff,tn W h_n
        o = torch.einsum('bhtn,bnhc->bthc', a_eff, v)
        f = torch.einsum('bthc,hcd->btd', o, L.w_sh)
        return h + f, w

    # ---------------------------------------------------------------- 세그먼트 1회
    def forward(self, tokens, h=None, w=None):
        """tokens: [B,T] long (0 PAD, 1 빈칸, 2..10 = 숫자 1..9)
        반환: logits [B,T,vocab], h [B,T,d], w [B,H,T,T]"""
        B = tokens.shape[0]
        if h is None:
            h = self.init_hidden.expand(B, self.T, self.d).contiguous()
        inj = self.embed(tokens.long())                              # [B,T,d]

        L  = self.layer
        AB = self.W_C(L)
        kc = self.kernel(L)

        for _ in range(self.blocks_per_seg):
            h    = h + self.embed_scale * inj                        # ① 주입 (매 블록)
            h, w = self.step(L, h, AB, kc, w)                        # ② 어텐션·쓰기·읽기·수송
            h    = self.boundary(L, h)                               # ③ 경계 (쌍선형)
            h    = self.phi(h)                                       # ④ 소산

        return self.w_cls(h), h, w


if __name__ == "__main__":
    torch.manual_seed(0)
    m = LT41()
    B = 2
    tokens = torch.randint(1, 11, (B, m.T))                          # 아무 격자
    logits, h, w = m(tokens)

    n = sum(p.numel() for p in m.parameters())
    print(f"파라미터           {n:,}")
    print(f"logits             {tuple(logits.shape)}")
    print(f"h                  {tuple(h.shape)}   ‖h_t‖ 평균 {h.norm(dim=-1).mean():.2f}  (반경 √d = {math.sqrt(m.d):.2f})")
    print(f"w (결합 기억)      {tuple(w.shape)}   |w| 평균 {w.abs().mean():.4f}")

    # ψ 제거의 구조적 귀결: a 가 대칭인지 확인
    with torch.no_grad():
        a = m.attn(h, m.W_C(m.layer), m.kernel(m.layer))
        print(f"max|a − aᵀ|        {(a - a.transpose(-1, -2)).abs().max():.2e}   ← ψ 없으면 0 (명제 7)")
        print(f"a 범위             [{a.min():.3f}, {a.max():.3f}]")

    # 두 세그먼트째 — h 와 w 를 이어받아 계속 굴릴 수 있다
    logits2, h2, w2 = m(tokens, h.detach(), w.detach())
    print(f"세그먼트 2 logits  {tuple(logits2.shape)}   변화량 {(logits2-logits).abs().mean():.4f}")
