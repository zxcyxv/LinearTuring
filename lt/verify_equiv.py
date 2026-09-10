"""lt/train.py 가 리팩토링 전 스크립트 세 벌(train_0901 / train_v11 / train_v17)과 수치적으로 같은지.
옛 스크립트는 git 이력(f5793d22344e601c6e3a25d12c519e82ad2b7632)에서 꺼내 temp 에 두고, 같은 체크포인트(npz, EMA)·같은 64 퍼즐·8 세그먼트를
fp32(amp 끔)·bf16(amp 켬) 두 모드로 돌려 로짓·w 를 비교한다.  2026-09-10 결과: 여섯 경우 전부 max|Δ| = 0."""
import os, sys, subprocess, tempfile, importlib.util, torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT,"lt"))
from ckpt_npz import load, load_data, _strip, _remap_legacy
torch.set_grad_enabled(False); NB, SEG = 64, 8
LEGACY = "f5793d22344e601c6e3a25d12c519e82ad2b7632"
def load_mod(path):
    s=importlib.util.spec_from_file_location("m_"+os.path.basename(path).replace(".","_"), path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def legacy_mod(relpath):
    src=subprocess.run(["git","-C",ROOT,"show",f"{LEGACY}:{relpath}"],capture_output=True,text=True,check=True).stdout
    p=os.path.join(tempfile.mkdtemp(), os.path.basename(relpath)); open(p,"w",encoding="utf-8").write(src); return load_mod(p)
new=load_mod(os.path.join(ROOT,"lt/train.py"))
X,Y,batch=load_data(ROOT,NB)
def run(mod, cfg, sd, amp):
    cfg=dict(cfg); cfg.update(batch_size=NB, seq_len=81, num_puzzle_identifiers=1, loops=SEG+1, amp=amp)
    m=mod.LT(cfg).cuda().eval(); miss,unexp=m.load_state_dict(sd, strict=False)
    assert not [k for k in miss if "puzzle_emb" not in k] and not unexp, (miss, unexp)
    with torch.device("cuda"): carry=m.initial_carry(batch)
    L=[]; W=[]
    for si in range(SEG):
        carry,out=m(carry,batch); L.append(out["logits"].float().cpu()); W.append(carry.coupling.float().cpu())
    return torch.stack(L), torch.stack(W)
cases=[("v1   (train_0901, 0901@310527)", "2026-06/analysis/train_0901.py".replace("06/","09-06/"), "checkpoints/0901_R1B8_min_faith_step310527.npz", True),
       ("v1.1 (train_v11,  v11@160000)",   "2026-09-09/analysis/train_v11.py", "checkpoints/v11_step_160000.npz", False),
       ("v1.7 (train_v17,  v17@200000)",   "2026-09-09/analysis/train_v17.py", "checkpoints/v17_step_200000.npz", False)]
ok=True
for name, oldrel, npz, legacy_keys in cases:
    old=legacy_mod(oldrel)
    sd,meta=load(os.path.join(ROOT,npz), which="ema")
    sd={(_remap_legacy(k) if legacy_keys else _strip(k)):v for k,v in sd.items()}
    cfg=dict(meta.get("cfg") or {}) or dict(old.CFG)
    for amp in (False, True):
        Lo,Wo=run(old,cfg,sd,amp); Ln,Wn=run(new,cfg,sd,amp)
        dl=(Lo-Ln).abs().max().item(); dw=(Wo-Wn).abs().max().item(); agree=(Lo.argmax(-1)==Ln.argmax(-1)).float().mean().item()
        flag="OK" if (dl==0 and dw==0) else ("~OK" if dl<1e-3 and agree>0.999 else "MISMATCH"); ok &= flag!="MISMATCH"
        print(f"{name}  amp={str(amp):5s}  max|Δlogit|={dl:.2e}  max|Δw|={dw:.2e}  argmax 일치 {agree*100:.3f}%   {flag}")
print("\n전체:", "동치" if ok else "불일치 있음")
