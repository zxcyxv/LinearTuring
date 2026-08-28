"""경계(토큰 내 채널곱)와 어텐션 스텝(토큰 간)의 역할 분업 — 빈칸 로짓 엔트로피·argmax 뒤집힘을 단계별로 읽는다.
사용: python sharpen.py [--ckpt PATH --bilinear 0|1] [--n 256] [--out JSON]
원 결과: results/json/sharpen_R1B8_9765.json"""
import argparse, json, torch
from common import load_lt, load_test, make_batch, rollout, logits, CKPT_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT_DEFAULT); ap.add_argument("--bilinear", type=int, default=1)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--out", default=None)
args = ap.parse_args()
torch.set_grad_enabled(False)
m = load_lt(args.ckpt, bilinear=bool(args.bilinear), batch_size=args.n)
inp, lab, _ = load_test(args.n); blank = inp == 1
res, cur = {}, {}

def read(h):
    lg = logits(m, h)[..., 2:11]; p = lg.softmax(-1)
    ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
    return ent[blank].mean().item(), ((lg.argmax(-1) + 2) == lab)[blank].float().mean().item(), lg.argmax(-1)

def hook(loop, blk, stage, h, a):
    if loop not in (0, 1, 15) or blk not in (0, 3, 7): return
    e, acc, am = read(h); cur[stage] = (e, acc, am)
    if stage == "post_step":
        r = {"ent_pre": cur["pre"][0], "ent_postBnd": cur["post_bnd"][0], "ent_postInj": cur["post_inj"][0], "ent_postAttn": e,
             "acc_pre": cur["pre"][1], "acc_postBnd": cur["post_bnd"][1], "acc_postAttn": acc,
             "flip_Bnd": (cur["post_bnd"][2] != cur["pre"][2])[blank].float().mean().item(),
             "flip_Attn": (am != cur["post_inj"][2])[blank].float().mean().item()}
        res[f"L{loop}_b{blk}"] = {k: round(v, 3) for k, v in r.items()}
rollout(m, make_batch(inp, lab), hook=hook)
print(json.dumps(res, indent=1))
if args.out: json.dump(res, open(args.out, "w"), indent=1)
