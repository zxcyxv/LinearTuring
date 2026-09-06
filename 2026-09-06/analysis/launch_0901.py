import importlib.util, os, sys
SP="/tmp/claude-0/-workspace-LinearTuring/09399f2f-6f99-4ecc-9a96-4caf5a7ff697/scratchpad"
spec=importlib.util.spec_from_file_location("t0901", f"{SP}/train_0901.py")
m=importlib.util.module_from_spec(spec); sys.modules["t0901"]=m; spec.loader.exec_module(m)
m.CFG.update(num_processes=1)
for k in ("RANK","WORLD_SIZE","MASTER_ADDR","MASTER_PORT","LOCAL_RANK"):
    os.environ.setdefault(k, {"RANK":"0","WORLD_SIZE":"1","MASTER_ADDR":"127.0.0.1",
                              "MASTER_PORT":"29572","LOCAL_RANK":"0"}[k])
m.run()
