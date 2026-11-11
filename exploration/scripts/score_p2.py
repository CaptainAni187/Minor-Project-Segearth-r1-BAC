"""Score the final fix (P2: 7B names, OWLv2 score >= 0.25) and the two 3B variants on every set, with per-category
gIoU before / after. Reproduces exploration/results/p2_results.json (the final numbers).

  python exploration/scripts/score_p2.py [--out exploration/results/p2_results.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_policy import R, arrays, score  # noqa: E402

POLICIES = [("Qwen2.5-7B-Instruct", 0.25), ("Qwen2.5-3B-Instruct", 0.35), ("Qwen2.5-3B-Instruct", 0.0)]
SETS = ["valrest", "testsplit", "calib300", "test300"]


def by_category(s, names, tau):
    m = pd.DataFrame(json.load(open(os.path.join(R, "logits", f"{s}_meta.json"))))
    *_, gb, _, _, gp, _ = arrays([s], names, tau)
    cat = m.category.where(m.category.str.startswith("held-out") | (m.category == "empty"), "seen").values
    return {c: [int((cat == c).sum()), round(100 * gb[cat == c].mean(), 1), round(100 * gp[cat == c].mean(), 1)]
            for c in sorted(set(cat))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "p2_results.json"))
    a = ap.parse_args()
    out = {}
    for s in SETS:
        for names, tau in POLICIES:
            r = score([s], names, tau)
            r.pop("models")
            r["by_category"] = by_category(s, names, tau)
            out[f"{s}|{names}|tau{tau}"] = r
            print(f"{s:9s} {names:20s} tau {tau:<4}  routed {r['routed']:3d}  dgIoU {r['d_gIoU']}  dcIoU {r['d_cIoU']}")
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)
