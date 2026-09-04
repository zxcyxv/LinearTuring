"""블록 1개의 연산·활성 바이트 해석 모델 (GPU 불필요).

MAC 은 forward 기준. backward 는 대략 2배(입력·가중치 두 경로).
활성 바이트는 backward 가 필요로 해 살아 있어야 하는 텐서 (bf16 = 2 byte 가정).
"""
import argparse, json

def block_cost(B, T, d, H, inter, stdp, window):
    dh = d // H; p = dh // 2; C = dh
    tok = B * T
    mac, act = {}, {}

    # 경계 (쌍선형)
    mac["boundary.gate_up (mm)"] = tok * d * 2 * inter
    mac["boundary.down (mm)"]    = tok * inter * d
    act["boundary.g,u"]          = 2 * tok * inter      # 곱의 두 인자 모두 backward 에 필요
    act["boundary.h_in"]         = tok * d

    # 주소 사영
    mac["addr W_C (mm)"] = 2 * tok * (H * p) * d
    act["addr x,y(정규화후)"] = 2 * tok * H * p
    act["addr nrm"] = tok * H

    # 값 사영
    mac["value W (mm)"] = tok * (H * C) * d
    act["value v"] = tok * H * C

    # T×T 텐서들
    n_tt = 1                                    # a
    mac["attn a (bmm)"] = 2 * B * H * T * T * p
    if stdp:
        mac["agree (bmm)"] = B * H * T * T * C
        n_tt += 1                               # agree
        if window == "beta":
            mac["attn a_beta (bmm)"] = 2 * B * H * T * T * p
            n_tt += 1
        n_tt += 2                               # w(상태, 그래디언트 창 안), a_eff
    mac["message o (bmm)"] = B * H * T * T * C
    mac["out Wt (mm)"] = tok * (H * C) * d
    act[f"T×T 텐서 {n_tt}개"] = n_tt * B * H * T * T
    act["Phi n2"] = tok

    return mac, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--d", type=int, default=832)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--inter", type=int, default=2304)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--stdp", type=int, default=1)
    ap.add_argument("--window", default="psi")
    ap.add_argument("--bytes-per-el", type=int, default=2)
    a = ap.parse_args()
    T = a.grid * a.grid
    mac, act = block_cost(a.bs, T, a.d, a.heads, a.inter, bool(a.stdp), a.window)

    tm = sum(mac.values()); ta = sum(act.values())
    print(f"# B={a.bs} T={T} d={a.d} H={a.heads} inter={a.inter} blocks={a.blocks} "
          f"stdp={bool(a.stdp)} window={a.window}")
    print(f"\n{'연산 (forward MAC)':<28}{'G-MAC/블록':>12}{'%':>8}")
    for k, v in sorted(mac.items(), key=lambda x: -x[1]):
        print(f"{k:<28}{v/1e9:>12.2f}{100*v/tm:>8.1f}")
    print(f"{'합':<28}{tm/1e9:>12.2f}{100:>8.1f}   → 세그먼트 {tm*a.blocks/1e9:.1f} G-MAC "
          f"= {2*tm*a.blocks/1e12:.3f} TFLOP (fwd)")
    print(f"\n{'backward 저장 활성':<28}{'MB/블록':>12}{'%':>8}")
    for k, v in sorted(act.items(), key=lambda x: -x[1]):
        print(f"{k:<28}{v*a.bytes_per_el/2**20:>12.1f}{100*v/ta:>8.1f}")
    print(f"{'합':<28}{ta*a.bytes_per_el/2**20:>12.1f}{100:>8.1f}   → 세그먼트 "
          f"{ta*a.bytes_per_el*a.blocks/2**30:.2f} GB")
    print(f"\n산술강도(블록): {tm/ta:.1f} MAC/element  (L4 bf16 ~121 TFLOPS / 300 GB/s → "
          f"균형점 ≈ {121e12/2/(300e9/a.bytes_per_el):.0f} MAC/element)")

if __name__ == "__main__":
    main()
