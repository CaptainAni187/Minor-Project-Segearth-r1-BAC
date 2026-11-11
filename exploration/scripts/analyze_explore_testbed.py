"""Paired comparisons for the exploration testbed runs (exploration/results/bottleneck) against the earlier runs
(results/bottleneck = Mac, results/kaggle/bottleneck = Kaggle). Writes exploration/results/testbed_explore.json;
does not touch the main result files.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac")); sys.path.insert(0, os.path.join(ROOT, "bac", "token_bottleneck"))
from analyze import paired_bootstrap
from analyze_testbed import parse
DIRS = [os.path.join(ROOT, "results", "kaggle", "bottleneck"), os.path.join(ROOT, "results", "bottleneck"),
        os.path.join(ROOT, "exploration", "results", "bottleneck")]
runs = {}
for d in DIRS:                                           # later dirs win (Mac over Kaggle, as in the main analysis)
    for p in glob.glob(os.path.join(d, "tb_*_s*.json")):
        j = json.load(open(p)); key = parse(j["name"])
        runs[key] = pd.read_csv(p[:-5] + ".csv", dtype={"name": str}).set_index("name").iou
def per(cfg, t):
    s = {k[2]: v for k, v in runs.items() if k[0] == cfg and k[1] == t}
    return (pd.concat(s, axis=1).mean(axis=1), sorted(s)) if s else (None, [])
out = {"summary": {}, "comparisons": {}}
for (cfg, t) in sorted({(k[0], k[1]) for k in runs}):
    s = {k[2]: v.mean() for k, v in runs.items() if k[0] == cfg and k[1] == t}
    out["summary"][f"{cfg}@{t}"] = dict(mean=round(100 * np.mean(list(s.values())), 2), std=round(100 * np.std(list(s.values()), ddof=1), 2) if len(s) > 1 else None, seeds=len(s))
for t in (16, 64):
    for a, b in [("bacq_supb", "uniform"), ("bacq_supr", "uniform"), ("bacq_supb", "bacq"), ("bacq_supb", "pool1024"), ("bacq_supb", "bacq_supbrand"), ("bacq_supb", "bacq_supbr0"), ("bacq_supbr0", "uniform"), ("bacq_supbrand", "uniform"),
                 ("bac2_orb", "uniform"), ("bac2_orr", "bac2_orb"), ("bac2_r75", "bac2")]:
        A, sa = per(a, t); B, sb = per(b, 1024 if b == "pool1024" else t)
        if A is None or B is None: continue
        m = pd.concat([A.rename("a"), B.rename("b")], axis=1).dropna()
        d, lo, hi = paired_bootstrap(m.a, m.b)
        out["comparisons"][f"{a} - {b} @{t}"] = dict(delta=round(100 * d, 2), ci=[round(100 * lo, 2), round(100 * hi, 2)], seeds=[len(sa), len(sb)])
json.dump(out, open(os.path.join(ROOT, "exploration", "results", "testbed_explore.json"), "w"), indent=1)
for k, v in out["comparisons"].items(): print(f"{k:34s} {v}")
