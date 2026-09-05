"""v2(stdp_read=mul) 로컬 학습 스모크: 손실 하강·NaN 없음·w 통계 확인. 1 GPU, 배치 64."""
import os, importlib.util
ROOT = "/workspace/LinearTuring"
spec = importlib.util.spec_from_file_location("tk", os.environ.get("TK", os.path.join(ROOT, "kaggle/train_kaggle.py"))); tk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tk)
tk.CFG.update(data_npz=os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"),
              out_dir=os.environ.get("OUT", "/tmp/claude-0/-workspace/08bc710b-d194-40cc-8cfb-cb09bdc9e744/scratchpad/v2smoke"),
              stdp_read=os.environ.get("READ", "mul"), compile=False, global_batch_size=int(os.environ.get("GBS", 64)),
              max_steps=int(os.environ.get("MAX_STEPS", 1500)), log_every=100, eval_interval=125,
              save_every_steps=10**9, milestone_every=0, keep_last=1, max_hours=2.0,
              scan_kaggle_input=False, num_processes=1, expect_processes=None)
import json
if os.environ.get("CFG_JSON"): tk.CFG.update(json.loads(os.environ["CFG_JSON"]))
if __name__ == "__main__":
    tk.main()
