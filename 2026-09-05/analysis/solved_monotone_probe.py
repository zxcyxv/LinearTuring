"""완답이 세그먼트를 넘어 유지되는가 — 집계(완답 수 감소) + 퍼즐 단위(맞았다가 틀림) 사건.
대상: 123k 원판(R1B8_bilin_r2, sudoku/lt.py) · stdp1(minimal_0901). 테스트 앞 N 퍼즐, 256 세그.
사용: python solved_monotone_probe.py --n 1024 --segs 256"""
import os, sys, argparse, importlib.util, time, json, numpy as np, torch, yaml
ROOT = "/workspace/LinearTuring"; H = os.path.join(ROOT, "refs", "ckpt_hist"); sys.path.insert(0, H)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1024); ap.add_argument("--segs", type=int, default=256); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--which", default="bilin123k,stdp1"); args = ap.parse_args()
torch.set_grad_enabled(False)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
inp = torch.from_numpy(z["test_inputs"][:args.n].reshape(-1, 81).astype(np.int32) + 1); lab = torch.from_numpy(z["test_labels"][:args.n].reshape(-1, 81).astype(np.int32) + 1).long()
def load(which):
    if which == "kaggle":
        s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
        ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
        cfg = dict(ck["cfg"]); cfg.update(batch_size=args.bs, seq_len=81, num_puzzle_identifiers=1, loops=args.segs + 1)
        sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
        m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True)
        return m, tk.LTCarry, "kaggle@182450 (λ 혼합, 레이어 2벌)"
    if which == "bilin123k":
        s = importlib.util.spec_from_file_location("ltmod", os.path.join(ROOT, "sudoku", "lt.py")); lt = importlib.util.module_from_spec(s); s.loader.exec_module(lt)
        m = lt.load_lt(os.path.join(H, "R1B8_bilin_r2_step123039.pt"), batch_size=args.bs, loops=args.segs + 1, amp=True)
        return m, lt.LTCarry, "R1B8_bilin_r2@123039 (STDP 없음)"
    s = importlib.util.spec_from_file_location("minimal", os.path.join(H, "minimal_0901.py")); mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod)
    cfg = yaml.safe_load(open(os.path.join(H, "R1B8_min_stdp1_config.yaml")))["arch"]; cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
    cfg.update(batch_size=args.bs, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, loops=args.segs + 1)
    ck = torch.load(os.path.join(H, "R1B8_min_stdp1_step228501.pt"), map_location="cpu", weights_only=False)
    full = dict(ck["raw_model_state_dict"]); full.update(ck["model_state_dict"])
    m = mod.LT(cfg).cuda().eval(); m.load_state_dict({k.replace("_orig_mod.model.", ""): v for k, v in full.items()}, strict=False)
    return m, mod.LTCarry, "R1B8_min_stdp1@228501 (λ=1)"
out = {}
for which in args.which.split(","):
    m, Carry, title = load(which); t0 = time.time()
    solved = torch.zeros(args.segs, args.n, dtype=torch.bool)          # [seg, puzzle]
    for b in range(0, args.n, args.bs):
        x = inp[b:b+args.bs].cuda(); y = lab[b:b+args.bs].cuda(); nb = len(x)
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(nb, dtype=torch.int32, device="cuda"))
        with torch.device("cuda"): carry = m.initial_carry(batch)
        for s_ in range(args.segs):
            carry, o = m(carry, batch)
            solved[s_, b:b+nb] = (o["logits"].argmax(-1) == y).all(-1).cpu()
    cnt = solved.sum(1).numpy()
    drops = np.diff(cnt); n_drop = int((drops < 0).sum())
    lost = (solved[:-1] & ~solved[1:]).sum(0)                           # 퍼즐별 맞았다→틀림 횟수
    ever = solved.any(0); final = solved[-1]
    res = dict(title=title, n=args.n, segs=args.segs,
               exact16=int(cnt[15]), exact_end=int(cnt[-1]), exact_max=int(cnt.max()), argmax=int(cnt.argmax()) + 1,
               agg_drop_transitions=n_drop, agg_max_drop=int(drops.min()),
               puzzles_with_loss=int((lost > 0).sum()), total_loss_events=int(lost.sum()),
               ever_solved=int(ever.sum()), solved_end=int(final.sum()), ever_but_not_end=int((ever & ~final).sum()),
               lost_after_seg16=int((solved[15:-1] & ~solved[16:]).any(0).sum()))
    out[which] = res
    print(f"\n=== {title}  n={args.n} segs={args.segs}  {time.time()-t0:.0f}s")
    print(f"  완답: seg16 {res['exact16']}  끝 {res['exact_end']}  최고 {res['exact_max']} @seg{res['argmax']}")
    print(f"  집계 감소 전이: {n_drop}/{args.segs-1}회  최대 하락 {res['agg_max_drop']}")
    print(f"  퍼즐 단위: 맞았다→틀림 겪은 퍼즐 {res['puzzles_with_loss']}개, 사건 {res['total_loss_events']}회  (seg16 이후 겪은 퍼즐 {res['lost_after_seg16']}개)")
    print(f"  한 번이라도 맞은 퍼즐 {res['ever_solved']}  끝에 맞은 퍼즐 {res['solved_end']}  맞았다가 끝엔 틀린 퍼즐 {res['ever_but_not_end']}")
    del m; torch.cuda.empty_cache()
jp = os.path.join(ROOT, "2026-09-05/results/json/solved_monotone.json")
prev = json.load(open(jp)) if os.path.exists(jp) else {}; prev.update(out); json.dump(prev, open(jp, "w"), indent=1)
