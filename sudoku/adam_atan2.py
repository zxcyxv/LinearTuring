"""AdamATan2 순수 torch 심 — pip 패키지의 CUDA 백엔드(adam_atan2_backend) 빌드 불가 대체.
레포 루트에 있어 `from adam_atan2 import AdamATan2` 가 깨진 site 패키지 대신 이 파일을 잡는다.

수식 (Everett et al. 2024 정준형):
    p ← p · (1 − lr·wd)                        # AdamW 식 분리 감쇠
    p ← p − lr · atan2(m̂, √v̂)                 # ε 없는 스케일 불변 갱신
주의: imbue CUDA 커널의 스케일 상수(a,b)와 다를 수 있음 — a=b=1 채택, lr 로 흡수 가능.
"""
import torch
from torch.optim.optimizer import Optimizer


class AdamATan2(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]; lr = g["lr"]; wd = g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1; t = st["step"]
                m, v = st["m"], st["v"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mh = m / (1 - b1 ** t)
                vh = v / (1 - b2 ** t)
                p.mul_(1 - lr * wd)
                p.add_(torch.atan2(mh, vh.sqrt()), alpha=-lr)
