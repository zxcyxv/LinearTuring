"""테스트셋을 N개로 절단 (스도쿠: 그룹=퍼즐=예제 1:1:1 구조 확인 후 일관 슬라이스)."""
import json, sys, numpy as np, glob, os
d = sys.argv[1]; N = int(sys.argv[2])
meta = json.load(open(f"{d}/dataset.json"))
print("절단 전:", meta)
for f in glob.glob(f"{d}/all__*.npy"):
    a = np.load(f)
    name = os.path.basename(f)
    if "inputs" in name or "labels" in name:
        np.save(f, a[:N])
    elif "puzzle_identifiers" in name:
        np.save(f, a[:N])
    elif "puzzle_indices" in name or "group_indices" in name:
        assert (a[:len(a)] == np.arange(len(a))).all(), f"{name} 1:1 구조 아님 — 수동 확인 필요"
        np.save(f, a[:N+1])
meta["total_groups"] = N
json.dump(meta, open(f"{d}/dataset.json","w"))
print("절단 후 total_groups:", N)
