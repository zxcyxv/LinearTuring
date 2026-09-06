"""kaggle/train_kaggle.py(플래그판) 체크포인트 외삽 — 그 파일의 extrapolate() 그대로."""
import importlib.util, sys, torch
SP="/tmp/claude-0/-workspace-LinearTuring/09399f2f-6f99-4ecc-9a96-4caf5a7ff697/scratchpad"
ck_path=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 512; segs=int(sys.argv[3]) if len(sys.argv)>3 else 128
spec=importlib.util.spec_from_file_location("tk","/workspace/LinearTuring/kaggle/train_kaggle.py"); tk=importlib.util.module_from_spec(spec); sys.modules["tk"]=tk; spec.loader.exec_module(tk)
ck=torch.load(ck_path, map_location="cpu", weights_only=False)
cfg=dict(ck["cfg"]); cfg.update(data_npz="/workspace/LinearTuring/kaggle/upload/sudoku_lt_1k.npz", milestone_extrap_n=n, batch_size=128, seq_len=81, num_puzzle_identifiers=1, compile=False)
device=torch.device("cuda")
with torch.device(device):
    base=tk.ACTLossHead(tk.LT(cfg), "stablemax_cross_entropy", q_weight=0.5)
base.load_state_dict(tk.strip_prefix(ck["model_state_dict"]), strict=True)
ema=tk.EMAHelper(mu=cfg["ema_rate"]); ema.register(base); ema.load_state_dict({k: v.to(device) for k,v in ck["ema_shadow"].items()})
_,_,te_in,te_lb=tk.load_data(cfg)
out=f"{SP}/kaggle_extrap_step_{ck['step']}.txt"
tk.extrapolate(base, te_in, te_lb, cfg, 0, 1, device, ck["step"], ema, segs, out)
print(open(out).read())
