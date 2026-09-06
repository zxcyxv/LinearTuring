import importlib.util, os, sys
SP="/tmp/claude-0/-workspace-LinearTuring/09399f2f-6f99-4ecc-9a96-4caf5a7ff697/scratchpad"
spec=importlib.util.spec_from_file_location("v3", "/workspace/LinearTuring/kaggle/train_kaggle_v3.py")
m=importlib.util.module_from_spec(spec); sys.modules["v3"]=m; spec.loader.exec_module(m)
m.CFG.update(num_processes=1,
             data_npz="/workspace/LinearTuring/kaggle/upload/sudoku_lt_1k.npz",
             out_dir=f"{SP}/run_v3",
             scan_kaggle_input=False,
             max_hours=24.0,                # 밤새. 필요하면 죽인다
             milestone_every=60000, milestone_extrap_n=512,
             log_every=250)
for k,v in {"RANK":"0","WORLD_SIZE":"1","MASTER_ADDR":"127.0.0.1","MASTER_PORT":"29574","LOCAL_RANK":"0"}.items():
    os.environ.setdefault(k, v)
m.run()
