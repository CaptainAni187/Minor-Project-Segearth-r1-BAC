"""Score the pre-registered policy P* (NOTES.md) against the released model at full resolution.

  python exploration/scripts/final_score.py testsplit [test300 calib300]
Uses exploration/results/logits/{set}_meta.json (released per-image inter/union at 1024) and the refusals-only
OV file exploration/results/ov/{set}_Qwen2.5-3B-Instruct_vocab_base_refusals.npz (P*'s masks for the routed images).
Writes exploration/results/final_{set}.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac"))
from analyze import paired_bootstrap  # noqa: E402

R = os.path.join(ROOT, "exploration", "results")


def ciou_boot(inter_a, union_a, inter_b, union_b, n=5000, seed=0):
    """Paired bootstrap of the cIoU difference (images resampled jointly)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(inter_a), (n, len(inter_a)))
    d = inter_a[idx].sum(1) / union_a[idx].sum(1) - inter_b[idx].sum(1) / union_b[idx].sum(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    for s in sys.argv[1:] or ["testsplit"]:
        m = pd.DataFrame(json.load(open(os.path.join(R, "logits", f"{s}_meta.json"))))
        o = np.load(os.path.join(R, "ov", f"{s}_Qwen2.5-3B-Instruct_vocab_base_refusals.npz"))
        routed = dict(zip(o["names"], range(len(o["names"]))))
        assert set(routed) == set(m.name[m.pred_area_full == 0]), "routed set must equal the refusals"
        it_b, un_b = m.inter_full.values.astype(float), m.union_full.values.astype(float)
        iou_b = m.iou_full.values
        it_p, un_p, iou_p = it_b.copy(), un_b.copy(), iou_b.copy()
        for i, nm in enumerate(m.name):
            if nm in routed:
                j = routed[nm]
                it_p[i], un_p[i], iou_p[i] = o["inter_full"][j, 0], o["union_full"][j, 0], o["iou_full"][j, 0]
        d, lo, hi = paired_bootstrap(iou_p, iou_b)
        c_b, c_p = it_b.sum() / un_b.sum(), it_p.sum() / un_p.sum()
        clo, chi = ciou_boot(it_p, un_p, it_b, un_b)
        r = m.name.isin(routed).values
        cat = m.category.where(m.category.str.startswith("held-out") | (m.category == "empty"), "seen")
        out = dict(set=s, n=len(m), routed=int(r.sum()),
                   released=dict(gIoU=round(100 * iou_b.mean(), 2), cIoU=round(100 * c_b, 2)),
                   policy=dict(gIoU=round(100 * iou_p.mean(), 2), cIoU=round(100 * c_p, 2)),
                   delta_gIoU=[round(100 * d, 2), round(100 * lo, 2), round(100 * hi, 2)],
                   delta_cIoU=[round(100 * (c_p - c_b), 2), round(100 * clo, 2), round(100 * chi, 2)],
                   routed_mean_iou=dict(released=round(100 * iou_b[r].mean(), 1), policy=round(100 * iou_p[r].mean(), 1)),
                   routed_empty_targets=int((r & (m.gt_area_full == 0).values).sum()),
                   by_category={c: dict(n=int((cat == c).sum()), released=round(100 * iou_b[(cat == c).values].mean(), 1),
                                        policy=round(100 * iou_p[(cat == c).values].mean(), 1), routed=int((r & (cat == c).values).sum()))
                                for c in sorted(cat.unique())})
        json.dump(out, open(os.path.join(R, f"final_{s}.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
