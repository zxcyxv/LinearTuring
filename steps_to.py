"""붕괴 지점이 겹쳐도 남는 신호: 임계값 도달까지 걸린 step 수."""
import json, glob, re, sys
th = float(sys.argv[1]) if len(sys.argv)>1 else 0.95
rows={}
for f in sorted(glob.glob("runs/p*_seq.json")):
    m=re.match(r"runs/p(R|D)(\d+)_(full|frozen)(?:_s(\d+))?_seq\.json", f)
    if not m: continue
    kind,val,cond,seed=m.groups()
    d=json.load(open(f)); log=d["log"]
    hit=next((e["step"] for e in log if e["acc"]>=th), None)
    rows.setdefault((kind,int(val)),{})[cond+(f"_s{seed}" if seed else "")]=(d["final"], hit)
print(f"{'':10s} {'A 자유: 최종 / 도달step':28s} {'A 고정: 최종 / 도달step'}")
for (kind,val),v in sorted(rows.items()):
    lab=("R=%d"%val) if kind=="R" else ("d=%d"%val)
    def fmt(c):
        if c not in v: return "—"
        fin,hit=v[c]; return f"{fin:.4f} / {hit if hit else '미도달':>6}"
    print(f"{lab:10s} {fmt('full'):28s} {fmt('frozen')}")
