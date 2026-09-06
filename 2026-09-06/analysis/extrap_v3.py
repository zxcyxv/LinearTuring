"""v3 체크포인트 외삽 — 학습 스크립트의 extrapolate() 그대로 (EMA, 마일스톤과 같은 채점)."""
import importlib.util, sys, os, torch
SP="/tmp/claude-0/-workspace-LinearTuring/09399f2f-6f99-4ecc-9a96-4caf5a7ff697/scratchpad"
ck_path=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 512; segs=int(sys.argv[3]) if len(sys.argv)>3 else 128
spec=importlib.util.spec_from_file_location("v3","/workspace/LinearTuring/kaggle/train_kaggle_v3.py"); v3=importlib.util.module_from_spec(spec); sys.modules["v3"]=v3; spec.loader.exec_module(v3)
cfg=dict(v3.CFG); cfg.update(data_npz="/workspace/LinearTuring/kaggle/upload/sudoku_lt_1k.npz", milestone_extrap_n=n, batch_size=128, seq_len=81, num_puzzle_identifiers=1, compile=False)
ck=torch.load(ck_path, map_location="cpu", weights_only=False)
device=torch.device("cuda")
with torch.device(device):
    base=v3.ACTLossHead(v3.LT(cfg), "stablemax_cross_entropy", q_weight=0.5)
base.load_state_dict(v3.strip_prefix(ck["model_state_dict"]), strict=True)
ema=v3.EMAHelper(mu=cfg["ema_rate"]); ema.register(base); ema.load_state_dict({k: v.to(device) for k,v in ck["ema_shadow"].items()})
_,_,te_in,te_lb=v3.load_data(cfg)
out=f"{SP}/run_v3/extrap_adhoc_step_{ck['step']}.txt"
v3.extrapolate(base, te_in, te_lb, cfg, 0, 1, device, ck["step"], ema, segs, out)
print(open(out).read())
