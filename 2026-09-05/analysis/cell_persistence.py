"""칸 단위 고정성: 맞은 칸이 나중에 틀리는가. 256퍼즐 × 64세그, 세그 끝 예측으로 계산. 해결/미해결 퍼즐 분리."""
import os, importlib.util, time, numpy as np, torch
ROOT = "/workspace/LinearTuring"; torch.set_grad_enabled(False); N, SEGS = 256, 64
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=N, seq_len=81, num_puzzle_identifiers=1, loops=SEGS + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True)
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz")); X = z["test_inputs"][:N].reshape(N, 81).astype(int); Y = z["test_labels"][:N].reshape(N, 81).astype(int)
x = torch.from_numpy(X.astype(np.int32) + 1).cuda(); y = torch.from_numpy(Y.astype(np.int32) + 1).cuda().long()
batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device="cuda"))
with torch.device("cuda"): carry = m.initial_carry(batch)
P = []; M = []; t0 = time.time()
for si in range(SEGS):
    carry, o = m(carry, batch); lg = o["logits"].float(); P.append((lg.argmax(-1) - 1).cpu().numpy()); M.append(lg.topk(2, -1).values.diff(dim=-1).abs()[..., 0].cpu().numpy())
    if (si + 1) % 16 == 0: print(f"  seg {si+1}/{SEGS} {time.time()-t0:.0f}s", flush=True)
P = np.stack(P); M = np.stack(M); C = (P == Y[None]); blank = (X == 0)[None].repeat(SEGS, 0)
solved_end = C[-1].all(-1); print(f"\n퍼즐: 해결 {solved_end.sum()}  미해결 {(~solved_end).sum()}")
for name, sel in [("해결 퍼즐", solved_end), ("미해결 퍼즐", ~solved_end)]:
    Cs = C[:, sel]; Bs = blank[:, sel]; Ms = M[:, sel]
    c2w = (Cs[:-1] & ~Cs[1:] & Bs[1:]).sum(); w2c = (~Cs[:-1] & Cs[1:] & Bs[1:]).sum(); ncorr = (Cs[:-1] & Bs[:-1]).sum()
    print(f"\n[{name}] 세그 전이 {SEGS-1}회 × 퍼즐 {sel.sum()}")
    print(f"  맞음→틀림 전이 {c2w}회  (맞은 빈칸-세그 {ncorr}개 중 {100*c2w/max(ncorr,1):.2f}%)   틀림→맞음 {w2c}회")
    # 세그 1 끝에 맞았던 빈칸이 끝까지 맞은 비율, 한 번이라도 틀린 비율
    c1 = Cs[0] & Bs[0]; stay = (Cs & Bs).all(0)[c1]; ever_wrong = (~Cs[1:]).any(0)[c1]
    print(f"  세그 1 끝에 맞은 빈칸 {c1.sum()}개 중 64세그 내내 맞음 {100*stay.mean():.1f}%, 한 번이라도 틀림 {100*ever_wrong.mean():.1f}%, 끝에 맞음 {100*Cs[-1][c1].mean():.1f}%")
    # 세그 16 끝 기준
    c16 = Cs[15] & Bs[15]; print(f"  세그 16 끝에 맞은 빈칸 {c16.sum()}개 중 이후 한 번이라도 틀림 {100*(~Cs[16:]).any(0)[c16].mean():.1f}%, 끝에 맞음 {100*Cs[-1][c16].mean():.1f}%")
    # 마진: 맞은 칸 vs 틀린 칸 (세그 16, 64)
    for sg in (15, 63):
        cm = Ms[sg][Cs[sg] & Bs[sg]]; wm = Ms[sg][~Cs[sg] & Bs[sg]]
        print(f"  세그 {sg+1} 마진: 맞은 칸 {cm.mean():.1f} (중앙값 {np.median(cm):.1f})  틀린 칸 {wm.mean() if wm.size else float('nan'):.1f} (중앙값 {np.median(wm) if wm.size else float('nan'):.1f})")
    # 마진으로 맞은 칸을 골라낼 수 있나: 마진 상위 X% 칸의 정확도 (세그 16)
    sg = 15; mm = Ms[sg][Bs[sg]]; cc = Cs[sg][Bs[sg]]; order = np.argsort(-mm)
    for q in (0.25, 0.5, 0.75, 1.0):
        k = int(len(order) * q); print(f"  세그 16, 마진 상위 {int(q*100):3d}% 빈칸의 정확도 {100*cc[order[:k]].mean():.1f}%  (마진 하한 {mm[order[k-1]]:.1f})")
