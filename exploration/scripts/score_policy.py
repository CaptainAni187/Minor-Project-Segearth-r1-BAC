"""Score refusal-routing policies P(names, tau): route a refusal to the OV branch only if OWLv2's top score >= tau.

  python exploration/scripts/score_policy.py --sets calib300 test300 --names Qwen2.5-3B-Instruct --taus 0 0.2 0.3 0.4
Pools the given sets (images concatenated) and prints gIoU / cIoU deltas vs the released model with paired
bootstrap CIs. Used to choose tau and the name model on the development sets (calib300 + test300) only.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac")); sys.path.insert(0, os.path.dirname(__file__))
from analyze import paired_bootstrap  # noqa: E402
from final_score import ciou_boot  # noqa: E402

R = os.path.join(ROOT, "exploration", "results")


def arrays(sets, names, tau, mode="vocab", models="base"):
    ib, ub, gb, ip, up, gp, routed = [], [], [], [], [], [], 0
    for s in sets:
        m = pd.DataFrame(json.load(open(os.path.join(R, "logits", f"{s}_meta.json"))))
        o = np.load(os.path.join(R, "ov", f"{s}_{names}_{mode}_{models}_refusals.npz"))
        j = {n: k for k, n in enumerate(o["names"])}
        for _, row in m.iterrows():
            ib.append(row.inter_full); ub.append(row.union_full); gb.append(row.iou_full)
            k = j.get(row["name"])
            if k is not None and o["owl"][k, 0] >= tau:
                ip.append(o["inter_full"][k, 0]); up.append(o["union_full"][k, 0]); gp.append(o["iou_full"][k, 0]); routed += 1
            else:
                ip.append(row.inter_full); up.append(row.union_full); gp.append(row.iou_full)
    return [np.asarray(x, float) for x in (ib, ub, gb, ip, up, gp)] + [routed]


def score(sets, names, tau, mode="vocab", models="base"):
    ib, ub, gb, ip, up, gp, routed = arrays(sets, names, tau, mode, models)
    d, lo, hi = paired_bootstrap(gp, gb)
    clo, chi = ciou_boot(ip, up, ib, ub)
    dc = ip.sum() / up.sum() - ib.sum() / ub.sum()
    return dict(names=names, models=models, tau=tau, n=len(gb), routed=routed, gIoU=round(100 * gp.mean(), 2), cIoU=round(100 * ip.sum() / up.sum(), 2),
                d_gIoU=[round(100 * d, 2), round(100 * lo, 2), round(100 * hi, 2)], d_cIoU=[round(100 * dc, 2), round(100 * clo, 2), round(100 * chi, 2)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--names", default="Qwen2.5-3B-Instruct")
    ap.add_argument("--mode", default="vocab")
    ap.add_argument("--taus", nargs="+", type=float, default=[0.0])
    ap.add_argument("--models", default="base", help="SAM size + OWL tag of the OV file, e.g. base, large, base_owllarge")
    a = ap.parse_args()
    for t in a.taus:
        print(json.dumps(score(a.sets, a.names, t, a.mode, a.models)))
