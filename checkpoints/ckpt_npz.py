"""체크포인트 ↔ npz (git 보관용).

저장:  python checkpoints/ckpt_npz.py pack  <in.pt> <out.npz> [--raw] [--note "..."]
       ema_shadow (+ ema 에 없는 raw 키) 를 저장. --raw 면 raw_model_state_dict 도 저장.
       numel ≥ 1e5 텐서는 float16 (학습·평가가 bf16 autocast 라 정밀도 손실 없음), 작은 텐서(ψ, θ, gain, η 등)는 float32 유지.
로드:  from checkpoints.ckpt_npz import load ;  sd, meta = load("x.npz", which="ema")   # sd: {name: float32 tensor}
       model.load_state_dict(sd, strict=False) 로 평가. meta = {step, cfg, note, source}
"""
import json, sys, numpy as np, torch

BIG = 100_000

def _pack_sd(sd, prefix, out):
    for k, t in sd.items():
        if not torch.is_tensor(t): continue
        a = t.detach().cpu()
        if a.is_floating_point() and a.numel() >= BIG: a = a.to(torch.float16)
        out[f"{prefix}/{k}"] = a.numpy()

def pack(src, dst, with_raw=False, note=""):
    c = torch.load(src, map_location="cpu", weights_only=False)
    out = {}
    ema = c.get("ema_shadow") or {}
    raw = c.get("raw_model_state_dict") or c.get("model_state_dict") or {}
    _pack_sd(ema, "ema", out)
    _pack_sd({k: v for k, v in raw.items() if k not in ema}, "ema", out)      # ema 에 없는 키(퍼즐 임베딩 등)는 raw 로 채움
    if with_raw: _pack_sd(raw, "raw", out)
    meta = dict(step=int(c.get("step", -1)), cfg=c.get("cfg", {}), note=note or c.get("note", ""), source=src,
                ema_keys=list(ema.keys()), raw_filled_into_ema=[k for k in raw if k not in ema])
    out["__meta__"] = np.array(json.dumps(meta, default=str))
    np.savez_compressed(dst, **out)

def load(path, which="ema"):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    sd = {k.split("/", 1)[1]: torch.from_numpy(z[k]).to(torch.float32) if z[k].dtype.kind == "f" else torch.from_numpy(z[k])
          for k in z.files if k.startswith(which + "/")}
    return sd, meta

if __name__ == "__main__":
    if sys.argv[1] == "pack":
        a = sys.argv[2:]; note = ""
        if "--note" in a: i = a.index("--note"); note = a[i + 1]; del a[i:i + 2]
        raw = "--raw" in a; a = [x for x in a if x != "--raw"]
        pack(a[0], a[1], raw, note)
    else:
        sd, meta = load(sys.argv[2]); print(meta["step"], len(sd), sum(v.numel() for v in sd.values()))
