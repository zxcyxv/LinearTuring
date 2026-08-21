import json,glob,re
rows={}
for f in sorted(glob.glob("runs/ca*_ca.json")):
    m=re.match(r"runs/ca(\d+)_k(\d+)_(full|frozen)_ca\.json",f)
    if m:
        r_,k_,c_=m.groups(); d=json.load(open(f))
        rows.setdefault((int(r_),int(k_)),{})[c_]=(d["cell"],d["exact"],"")
for f in sorted(glob.glob("runs/ca*_k*_*.log")):
    m=re.match(r"runs/ca(\d+)_k(\d+)_(full|frozen)\.log",f)
    if not m: continue
    r_,k_,c_=m.groups()
    if c_ in rows.get((int(r_),int(k_)),{}): continue
    ls=[l for l in open(f) if "전체일치" in l]
    if ls:
        v=re.findall(r"\uc140 ([\d.]+) \uc804\uccb4\uc77c\uce58 ([\d.]+)", ls[-1])
        if v: rows.setdefault((int(r_),int(k_)),{})[c_]=(float(v[0][0]),float(v[0][1])," *")
for rule in (90,110):
    ks=sorted(k for (r,k) in rows if r==rule)
    if not ks: continue
    print(f"\nRule {rule}      k   A자유 셀/전체일치      A고정 셀/전체일치")
    for k in ks:
        v=rows[(rule,k)]
        def fmt(c):
            if c not in v: return f"{'-':>21}"
            t=v[c]; return f"{t[0]:.4f} / {t[1]:.4f}{t[2]:>2}".rjust(21)
        print(f"{'':10s}{k:>4}   {fmt('full')}  {fmt('frozen')}")
print("\n(* = 진행 중 중간값)")
