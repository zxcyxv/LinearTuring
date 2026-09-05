"""경사흐름 구조(EqLT v2) 학습 스모크.
값 블록: a 고정 아래 E_v = −½Σ a⟨Wh,Wh⟩ − V(h_v) − ⟨√d inj_v, h_v⟩ 의 분할 경사 스텝(수송 → 퍼텐셜) 뒤 RMSNorm. V = Σ g·tanh(u) (게이트, 대칭).
주소 블록: 이전 분리 모델(psi0_split)과 같은 쌍선형 경계를 h 전체 입력으로 받아 주소 출력만 취함 → 값이 매 블록 주소로 흘러듦. 대칭 조건 불필요(값 스텝 동안 a 고정).
train_kaggle.py 의 데이터·학습 루프를 그대로 쓰고 모델 클래스만 교체. 1만 스텝 안에 train acc 70% 를 넘는지 본다."""
import os, sys, math, importlib.util, torch, torch.nn.functional as F
from torch import nn
from dataclasses import replace
ROOT = "/workspace/LinearTuring"
s_ = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s_); s_.loader.exec_module(tk)

class EqInner(tk.LT_Inner):
    def __init__(self, config):
        super().__init__(config)
        assert self.split and config.psi_zero and not config.stdp and config.num_layers == 1
        L = self.layers[0]; del L.b_gate_up; del L.b_down
        d, d_a, d_v = self.d, self.d_a, self.d_v; inter = int(config.mlp_expansion * d * 2 / 3 + 255) // 256 * 256   # 2304 (이전과 동일)
        self.A_gu = nn.Linear(d, 2 * inter, bias=False); self.A_d = nn.Linear(inter, d_a, bias=False)     # 주소용 쌍선형 경계 (입력 h 전체)
        self.W_gv = nn.Linear(d_v, inter, bias=False); self.W_uv = nn.Linear(d_v, inter, bias=False)      # 값용 대칭 퍼텐셜 V = Σ g·tanh(u)
        with torch.no_grad():
            self.A_d.weight.zero_()                                                                       # 이전과 같이 항등 출발
            for lin in (self.W_gv, self.W_uv): lin.weight.mul_(config.pot_init)
    @staticmethod
    def rms(h): return h * (math.sqrt(h.shape[-1]) / (h.norm(dim=-1, keepdim=True) + 1e-6))
    @staticmethod
    def gated_grad(h, Wg, Wu):
        g = Wg(h); u = Wu(h); t = torch.tanh(u)
        return F.linear(t, Wg.weight.t()) + F.linear(g * (1 - t * t), Wu.weight.t())
    def _forward(self, carry, batch):
        L = self.layers[0]; inj = self.injection(batch) * self.embed_scale
        AB = self.W_C(L); kc = self.kernel(L); h = carry.current_hidden; d_a = self.d_a
        for _ in range(self.config.blocks_per_seg):
            h_a, h_v = h[..., :d_a], h[..., d_a:]
            gate, up = self.A_gu(h).chunk(2, dim=-1)
            h_a = self.rms(h_a + inj[..., :d_a] + self.A_d(0.5 * gate * up))                 # 주소 갱신: h 전체를 읽음
            a = self.attn_xy(self.addr(torch.cat([h_a, h_v], -1), AB), kc)                  # ψ=0 → 대칭, 값 스텝 동안 고정
            v = torch.einsum('btd,hcd->bthc', h_v, L.w_sh); o = torch.einsum('bhtn,bnhc->bthc', a, v)
            h_v = h_v + torch.einsum('bthc,hcd->btd', o, L.w_sh) + inj[..., d_a:]           # 값 스텝 1: 수송 (+주입)
            h_v = h_v + self.gated_grad(h_v, self.W_gv, self.W_uv)                           # 값 스텝 2: 퍼텐셜 경사
            h_v = self.rms(h_v)
            h = torch.cat([h_a, h_v], -1)
        return replace(carry, current_hidden=h.detach(), coupling=None, fresh=None, vprev=None, gate=None), self.w_cls(h)

class EqLT(tk.LT):
    def __init__(self, config_dict):
        nn.Module.__init__(self)
        self.config = tk.LTConfig.from_dict(config_dict)
        self.config.pot_init = float(config_dict.get("pot_init", os.environ.get("POT_INIT", 0.5)))
        self.inner = EqInner(self.config)

tk.LT = EqLT
tk.CFG.update(data_npz=os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"),
              out_dir=os.environ.get("OUT", "/tmp/claude-0/-workspace/08bc710b-d194-40cc-8cfb-cb09bdc9e744/scratchpad/eqlt3"),
              hidden_size=832, num_heads=8, blocks_per_seg=8, num_layers=1, addr_dim=416, psi_zero=True,
              stdp=False, block_order="post", mlp_expansion=4.0, compile=False,
              max_steps=int(os.environ.get("MAX_STEPS", 10000)), log_every=200, eval_interval=250,
              save_every_steps=10**9, milestone_every=0, keep_last=1, max_hours=4.0,
              scan_kaggle_input=False, num_processes=1, expect_processes=None)
if __name__ == "__main__":
    tk.main()
