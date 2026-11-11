"""Screen inference-time post-processing rules on dumped logits (no model runs).

Every rule's free parameter is chosen on calib300 only; the chosen setting is then reported once on test300 with a
paired bootstrap CI against the released behaviour (threshold 0, single pass). Screening runs at 512x512, which is
checked against the full-resolution baseline (meta iou_full).

  python exploration/scripts/screen_rules.py   -> exploration/results/screen_rules.json (+ printed table)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac"))
from analyze import paired_bootstrap  # noqa: E402

L = os.path.join(ROOT, "exploration", "results", "logits")


def load(name):
    z = np.load(os.path.join(L, f"{name}.npz"))
    meta = json.load(open(os.path.join(L, f"{name}_meta.json")))
    return z["plain"].astype(np.float32), z["flip"].astype(np.float32), z["gt"], meta


def iou(pred, gt):
    inter = (pred & gt).sum((1, 2)); union = (pred | gt).sum((1, 2))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def rule(logits, t=0.0, fallback=None):
    """Threshold at t; if that gives an empty mask and fallback=delta, keep pixels within delta of the maximum."""
    pred = logits > t
    if fallback is not None:
        empty = ~pred.any((1, 2))
        mx = logits.reshape(len(logits), -1).max(1)
        fb = logits > (mx - fallback)[:, None, None]
        pred = np.where(empty[:, None, None], fb, pred)
    return pred


def main():
    data = {s: load(s) for s in ["calib300", "test300"]}
    for s, (p, f, g, m) in data.items():
        full = np.mean([x["iou_full"] for x in m]); r512 = iou(p > 0, g).mean()
        print(f"{s}: baseline gIoU full-res {100 * full:.2f} | 512x512 screen {100 * r512:.2f} | "
              f"empty predictions {sum(x['pred_area_full'] == 0 for x in m)} | empty targets {sum(x['gt_area_full'] == 0 for x in m)}")
    grid = {
        "threshold": [dict(t=t) for t in [-2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 1]],
        "fallback": [dict(fallback=d) for d in [0.25, 0.5, 1, 1.5, 2, 3, 4, 6]],
        "threshold+fallback": [dict(t=t, fallback=d) for t in [-1, -0.5, -0.25, 0, 0.25] for d in [0.5, 1, 2, 3, 4]],
    }
    out = {}
    for src in ["plain", "flip"]:
        for fam, settings in grid.items():
            def score(s, kw):
                p, f, g, m = data[s]
                lg = p if src == "plain" else f
                return iou(rule(lg, **kw), g)
            best = max(settings, key=lambda kw: score("calib300", kw).mean())       # chosen on calib only
            tst, base = score("test300", best), iou(data["test300"][0] > 0, data["test300"][2])
            d, lo, hi = paired_bootstrap(tst, base)
            cal = score("calib300", best).mean() - iou(data["calib300"][0] > 0, data["calib300"][2]).mean()
            out[f"{src}/{fam}"] = dict(setting=best, calib_gain=round(100 * cal, 2), test_gIoU=round(100 * tst.mean(), 2),
                                       test_delta=round(100 * d, 2), ci=[round(100 * lo, 2), round(100 * hi, 2)])
    # per-category view of the best plain fallback on test
    p, f, g, m = data["test300"]
    kw = out["plain/fallback"]["setting"]
    new, base = iou(rule(p, **kw), g), iou(p > 0, g)
    cats = np.array([x["category"] if x["category"].startswith("held-out") or x["category"] == "empty" else "seen" for x in m])
    out["by_category_plain_fallback"] = {c: dict(n=int((cats == c).sum()), base=round(100 * base[cats == c].mean(), 1),
                                                new=round(100 * new[cats == c].mean(), 1)) for c in sorted(set(cats))}
    json.dump(out, open(os.path.join(ROOT, "exploration", "results", "screen_rules.json"), "w"), indent=1)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
