"""runs/ 전체를 훑어 단일 결과 원장 RESULTS.md 생성."""
import json, glob, re, os
rows=[]
for f in sorted(glob.glob("runs/*_ca.json")):
    d=json.load(open(f)); a=d["args"]
    rows.append(dict(exp="CA", tag=a["tag"], task=f"R{a['rule']} k{a['k']}",
        cfg=f"d={a['d']} τ={a.get('tau',1)}"+(" frz" if a["freeze_A"] else "")
            +(" noov" if a.get("no_ov") else "")+(" sheaf" if a.get("sheaf") else "")
            +(f" Λ{a.get('lam','')}" if a.get("lam") not in (None,"auto") else "")
            +(f" WO:{a.get('wo_mode')}" if a.get("boundary_wo") else ""),
        metric=f"{d['exact']:.4f}", metric2=f"{d['cell']:.4f}"))
for f in sorted(glob.glob("runs/*_seq.json")):
    d=json.load(open(f)); a=d["args"]
    rows.append(dict(exp="parity", tag=a["tag"], task=f"T={a['T']}",
        cfg=f"d={a['d']} R={a['R']}"+(" frz" if a["freeze_A"] else ""),
        metric=f"{d['final']:.4f}", metric2=""))
for f in sorted(glob.glob("runs/*_hop.json")):
    d=json.load(open(f)); a=d["args"]
    arch = f"TF L{a['L']}" if "L" in a else ("M1"+(" frz" if a.get("freeze_A") else ""))
    rows.append(dict(exp="hop", tag=a["tag"], task=f"n={a['n']} N={a['N']}",
        cfg=f"{arch} d={a['d']} τ={a.get('tau',1)} {a['steps']}st"
            +(f" WO:{a.get('wo_mode')}" if a.get("boundary_wo") else "")
            +(f" k×{a.get('kernel_lr_mult'):g}" if a.get("kernel_lr_mult") not in (None,1.0) else ""),
        metric=f"{d['query']:.4f}", metric2=f"{d['exact']:.4f}"))
for f in sorted(glob.glob("runs/*_log.json")):
    tag=os.path.basename(f)[:-9]
    try: d=json.load(open(f)); acc=d["epochs"][-1]["test_acc"]; a=d["args"]
    except Exception: continue
    lam=""
    af=f"runs/{tag}_attractor.json"
    if os.path.exists(af): lam=f"{json.load(open(af))['ftle']:+.2f}"
    rows.append(dict(exp="MNIST", tag=tag, task="분류",
        cfg=f"d={a['d']}"+(" frz" if a.get("freeze_A") else "")+(" noov" if a.get("no_ov") else "")
            +(f" γ={a['gamma']}" if a.get("fix_gamma") else "")+(" randR" if a.get("rand_R") else ""),
        metric=f"{acc:.4f}", metric2=lam))
for f in sorted(glob.glob("runs/base_*.json")):
    d=json.load(open(f))
    rows.append(dict(exp="기준선", tag=f"{d['arch']}L{d['L']}", task=f"R{d['rule']} k{d['k']}",
        cfg=f"{d['params']:,}p", metric=f"{d['exact']:.4f}", metric2=f"{d['cell']:.4f}"))
out=["# 결과 원장 (자동 생성)\n","| 실험 | 태그 | 과제 | 구성 | 주지표 | 보조 |","|---|---|---|---|---|---|"]
for r in rows: out.append(f"| {r['exp']} | {r['tag']} | {r['task']} | {r['cfg']} | {r['metric']} | {r['metric2']} |")
out.append(f"\n총 {len(rows)}건 · 주지표: CA/parity=전체일치, hop=질의, MNIST=test acc · 보조: CA=셀, MNIST=λ(FTLE)")
open("RESULTS.md","w").write("\n".join(out)+"\n")
print(f"RESULTS.md 생성: {len(rows)}건")
