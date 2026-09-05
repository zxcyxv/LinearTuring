"""섭동 1번: 미해결 퍼즐의 '진동하는 칸' h 만 init_hidden 으로 재초기화.
- 미해결 판정은 정답 없이 규칙 검사(행·열·박스 중복 없음 + 주어진 칸 일치 + 빈칸 없음)로만 한다.
- 유효한 퍼즐은 절대 건드리지 않는다 (정답 = 흡수 상태이므로 단조 보장).
- 세그 start 부터 every 세그마다 검사·섭동. 기준선(무섭동)과 같은 배치·같은 세그 수로 비교.
사용: python perturb1_reset_cells.py --n 1024 --segs 256 --start 64 --every 8"""
import os, argparse, importlib.util, time, json, numpy as np, torch
from dataclasses import replace
ROOT = "/workspace/LinearTuring"
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1024); ap.add_argument("--segs", type=int, default=256); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--start", type=int, default=64); ap.add_argument("--every", type=int, default=8); ap.add_argument("--tag", default="p1")
args = ap.parse_args(); torch.set_grad_enabled(False)
s = importlib.util.spec_from_file_location("tk", os.path.join(ROOT, "kaggle", "train_kaggle.py")); tk = importlib.util.module_from_spec(s); s.loader.exec_module(tk)
ck = torch.load(os.path.join(ROOT, "step_182450.zip"), map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(batch_size=args.bs, seq_len=81, num_puzzle_identifiers=1, loops=args.segs + 1)
sd = {k[len("model."):]: v for k, v in tk.strip_prefix(ck["model_state_dict"]).items()}
m = tk.LT(cfg).cuda().eval(); m.load_state_dict(sd, strict=True); I = m.inner
z = np.load(os.path.join(ROOT, "kaggle/upload/sudoku_lt_1k.npz"))
inp = torch.from_numpy(z["test_inputs"][:args.n].reshape(-1, 81).astype(np.int32) + 1); lab = torch.from_numpy(z["test_labels"][:args.n].reshape(-1, 81).astype(np.int32) + 1).long()
r = torch.arange(81, device="cuda") // 9; c = torch.arange(81, device="cuda") % 9; bx = (r // 3) * 3 + c // 3
def valid(pred, x):
    """규칙 검사만 (정답 미사용). pred/x: 토큰 (숫자 = 토큰−1, 빈칸 토큰 1)."""
    d = pred - 1
    ok = (d >= 1) & (d <= 9)
    given = x != 1
    ok_g = (~given) | (pred == x)
    oh = torch.nn.functional.one_hot(d.clamp(0, 9), 10)[..., 1:]                       # [B,81,9]
    def nodup(idx):
        cnt = torch.zeros(pred.shape[0], 9, 9, device=pred.device, dtype=oh.dtype).index_add_(1, idx, oh)   # [B,group,9]
        return (cnt <= 1).all(-1).all(-1)
    return ok.all(-1) & ok_g.all(-1) & nodup(r) & nodup(c) & nodup(bx)
def run(perturb: bool):
    print(f"[{'섭동' if perturb else '기준선'}] 시작  n={args.n} segs={args.segs} bs={args.bs}", flush=True)
    ex = np.zeros(args.segs); n_inval = []; n_osc = []; n_cells = []; touched = torch.zeros(args.n, dtype=torch.bool); t0 = time.time()
    solved_end = torch.zeros(args.n, dtype=torch.bool)
    for b in range(0, args.n, args.bs):
        x = inp[b:b+args.bs].cuda(); y = lab[b:b+args.bs].cuda(); nb = len(x)
        batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(nb, dtype=torch.int32, device="cuda"))
        with torch.device("cuda"): carry = m.initial_carry(batch)
        prev = None
        for si in range(args.segs):
            carry, o = m(carry, batch); pred = o["logits"].argmax(-1)
            ex[si] += (pred == y).all(-1).sum().item()
            if perturb and si + 1 >= args.start and (si + 1 - args.start) % args.every == 0 and prev is not None:
                inval = ~valid(pred, x)                                    # 규칙 위반 퍼즐만
                osc = (pred != prev) & inval[:, None]                      # 그 중 진동하는 칸
                has = osc.any(-1)
                n_inval.append(inval.sum().item()); n_osc.append(has.sum().item()); n_cells.append(osc.sum().item() / max(has.sum().item(), 1))
                if (si + 1 - args.start) % (args.every * 4) == 0:
                    print(f"    seg {si+1:3d}: 위반 {int(inval.sum())}  진동있음 {int(has.sum())}  칸/퍼즐 {n_cells[-1]:.1f}  완답(배치) {int((pred == y).all(-1).sum())}/{nb}", flush=True)
                if has.any():
                    h = carry.current_hidden.clone()
                    h[osc] = I.init_hidden.to(h.dtype)
                    carry = replace(carry, current_hidden=h)
                    touched[b:b+nb] |= has.cpu()
            prev = pred
        solved_end[b:b+nb] = (pred == y).all(-1).cpu()
        print(f"  batch {b//args.bs+1}/{-(-args.n//args.bs)}  {time.time()-t0:.0f}s  seg16 누적 {int(ex[15])}  seg{args.segs} 누적 {int(ex[-1])}", flush=True)
    return dict(exact=ex.tolist(), n_inval=n_inval, n_osc=n_osc, cells=n_cells, touched=touched, solved_end=solved_end, sec=time.time() - t0)
base = run(False); pert = run(True)
N = args.n; eb = np.array(base["exact"]); ep = np.array(pert["exact"])
print(f"n={N} segs={args.segs} start={args.start} every={args.every}   기준선 {base['sec']:.0f}s / 섭동 {pert['sec']:.0f}s")
print(f"{'seg':>5} {'기준선':>8} {'섭동':>8} {'차':>6}")
for sg in [16, 32, 64, 96, 128, 160, 192, 224, 256]:
    if sg <= args.segs: print(f"{sg:5d} {int(eb[sg-1]):8d} {int(ep[sg-1]):8d} {int(ep[sg-1]-eb[sg-1]):+6d}")
print(f"최고: 기준선 {int(eb.max())} @seg{int(eb.argmax())+1}   섭동 {int(ep.max())} @seg{int(ep.argmax())+1}   끝 차 {int(ep[-1]-eb[-1]):+d} ({100*(ep[-1]-eb[-1])/N:+.2f}%p)")
ni, no, nc = np.array(pert["n_inval"]), np.array(pert["n_osc"]), np.array(pert["cells"])
print(f"검사 {len(ni)}회: 규칙위반 퍼즐 평균 {ni.mean():.1f} (첫 {ni[0]:.0f} → 끝 {ni[-1]:.0f}), 그중 진동 있는 퍼즐 {no.mean():.1f} (첫 {no[0]:.0f} → 끝 {no[-1]:.0f}), 재초기화 칸/퍼즐 평균 {nc.mean():.1f}")
t = pert["touched"]; sb = base["solved_end"]; sp = pert["solved_end"]
print(f"섭동 받은 퍼즐 {int(t.sum())}개 중 끝에 맞음 {int((t & sp).sum())}개  (같은 퍼즐 기준선에서 끝에 맞음 {int((t & sb).sum())}개)")
print(f"기준선엔 맞고 섭동판엔 틀린 퍼즐 {int((sb & ~sp).sum())}개 / 기준선엔 틀리고 섭동판엔 맞은 퍼즐 {int((~sb & sp).sum())}개")
json.dump(dict(cfg=vars(args), base=base["exact"], pert=pert["exact"], n_inval=pert["n_inval"], n_osc=pert["n_osc"], cells=pert["cells"]),
          open(os.path.join(ROOT, f"2026-09-05/results/json/perturb1_{args.tag}.json"), "w"), indent=1)
