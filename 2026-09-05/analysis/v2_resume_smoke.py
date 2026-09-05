"""v2 재개 스모크: step_1500.pt 에서 이어서 100스텝 (compile=True 로 컴파일 경로도 확인)."""
import os, importlib.util
ROOT = "/workspace/LinearTuring"; OUT = "/tmp/claude-0/-workspace/08bc710b-d194-40cc-8cfb-cb09bdc9e744/scratchpad/v2smoke"
spec = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle/train_kaggle.py")); tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)
tk.CFG.update(data_npz=os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"), out_dir=OUT, resume_from=OUT,
              stdp_read="mul", compile=True, global_batch_size=64, max_steps=1600, log_every=20, eval_interval=250,
              save_every_steps=10**9, milestone_every=0, keep_last=1, max_hours=1.0, scan_kaggle_input=False, num_processes=1, expect_processes=None)
if __name__ == "__main__":
    tk.main()
