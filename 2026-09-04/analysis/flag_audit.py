"""플래그 상호작용 감사 — 다른 플래그와 섞였을 때 조용히 무효가 되는 조합 찾기.

방법: 한 플래그만 바꿔 로짓이 실제로 달라지는지 본다. 안 달라지면 그 조합에서 그 플래그는 죽은 것.
(INDEX 자기정정 15·18 의 사고 유형: 플래그가 죽었는데 곡선만 보고 결론을 냈다)
"""
import os, sys, itertools, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "refs", "URM")); sys.path.insert(0, os.path.join(ROOT, "core"))
import minimal

BASE = dict(batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, puzzle_emb_ndim=832,
            hidden_size=832, num_heads=8, loops=16, grid=9, blocks_per_seg=8, amp=False,
            stdp=True, stdp_target="faithful", stdp_window="psi",
            stdp_eta_init=0.05, stdp_gain_init=1.0, stdp_lam_fixed=-1.0, block_order="pre")

def logits(cfg, nseg=2):
    torch.manual_seed(0)
    m = minimal.LT({**BASE, **cfg}).cuda().eval()
    g = torch.Generator().manual_seed(7)           # b_down 영init → 경계가 항등. 깨워야 순서/경계 축이 보인다
    with torch.no_grad():
        w = m.inner.b_down.weight
        w.copy_((torch.randn(w.shape, generator=g) * 0.02).to(w.device, w.dtype))
    b = {"inputs": torch.randint(1, 11, (2, 81), device="cuda"),
         "puzzle_identifiers": torch.zeros(2, dtype=torch.long, device="cuda")}
    torch.manual_seed(1); b["inputs"] = torch.randint(1, 11, (2, 81), device="cuda")
    from dataclasses import replace
    c = m.initial_carry(b)
    c = replace(c, current_hidden=c.current_hidden.cuda(), steps=c.steps.cuda(),
                halted=c.halted.cuda(), current_data={k: v.cuda() for k, v in c.current_data.items()})
    out = None
    with torch.no_grad():
        for _ in range(nseg):
            c, o = m(c, b); out = o["logits"]
    return out

CASES = [
    # (설명, 맥락, 바꾸는 플래그 A, B, 기대)
    ("stdp_window", dict(stdp_target="faithful"), dict(stdp_window="beta"), dict(stdp_window="psi"), "달라야"),
    ("stdp_window", dict(stdp_target="addr"),     dict(stdp_window="beta"), dict(stdp_window="psi"), "달라야"),
    ("stdp_window", dict(stdp_target="product"),  dict(stdp_window="beta"), dict(stdp_window="psi"), "같음(무효)"),
    ("stdp_window", dict(stdp_target="value"),    dict(stdp_window="beta"), dict(stdp_window="psi"), "같음(무효)"),
    ("stdp_window", dict(stdp=False),             dict(stdp_window="beta"), dict(stdp_window="psi"), "같음(무효)"),
    ("stdp_target", dict(stdp=False),             dict(stdp_target="faithful"), dict(stdp_target="addr"), "같음(무효)"),
    ("stdp_lam_init", dict(stdp_lam_fixed=1.0),   dict(stdp_lam_init=0.25), dict(stdp_lam_init=0.9), "같음(무효)"),
    ("stdp_lam_init", dict(stdp_lam_fixed=-1.0),  dict(stdp_lam_init=0.25), dict(stdp_lam_init=0.9), "달라야"),
    ("block_order",  dict(),                      dict(block_order="pre"), dict(block_order="post"), "달라야"),
    ("block_order",  dict(stdp=False),            dict(block_order="pre"), dict(block_order="post"), "달라야"),
    ("block_order",  dict(psi_zero=True),         dict(block_order="pre"), dict(block_order="post"), "달라야"),
    ("block_order",  dict(addr_dim=416),          dict(block_order="pre"), dict(block_order="post"), "달라야"),
    ("block_order",  dict(stdp_window="beta"),    dict(block_order="pre"), dict(block_order="post"), "달라야"),
    ("psi_zero",     dict(),                      dict(psi_zero=False), dict(psi_zero=True), "달라야"),
    ("stdp_diag",    dict(),                      dict(stdp_diag="keep"), dict(stdp_diag="zero"), "달라야"),
    ("stdp_gain_init", dict(),                    dict(stdp_gain_init=1.0), dict(stdp_gain_init=3.0), "달라야"),
]

print(f"{'플래그':<16}{'맥락':<34}{'max|Δ|':>12}{'기대':>12}  판정")
bad = 0
for name, ctx, a, b_, exp in CASES:
    try:
        d = (logits({**ctx, **a}) - logits({**ctx, **b_})).abs().max().item()
    except Exception as e:
        print(f"{name:<16}{str(ctx)[:32]:<34}{'ERR':>12}{exp:>12}  {type(e).__name__}: {str(e)[:50]}"); bad += 1; continue
    same = d == 0.0
    ok = (same and "같음" in exp) or ((not same) and exp == "달라야")
    if not ok: bad += 1
    print(f"{name:<16}{str(ctx)[:32]:<34}{d:>12.3e}{exp:>12}  {'OK' if ok else '★불일치'}")

# 오타 키가 조용히 먹히는지
try:
    c = minimal.LTConfig(**{**BASE, "stdp_windwo": "psi"})
    print(f"\n★ LTConfig 는 extra='allow' — 오타 키 'stdp_windwo' 를 조용히 받아들인다 "
          f"(stdp_window 는 여전히 '{c.stdp_window}'). 발사 인자 오타가 절대 안 잡힌다.")
except Exception as e:
    print("\n오타 키 거부:", e)
print(f"\n불일치 {bad}건")
