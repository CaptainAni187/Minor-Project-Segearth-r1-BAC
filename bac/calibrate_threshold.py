"""Mask-path improvement without retraining: calibrate the mask threshold on held-out validation images.

The released model binarises its mask logits at 0. Here the threshold is chosen on 300 *other* validation images
(the next 300 of the same seeded shuffle that defines our 300-image test subset, so the two sets are disjoint),
then applied unchanged to the test subset. Done for the plain logits and for flip-TTA logits.

  python bac/calibrate_threshold.py          -> results/calibration/{rows.csv, summary.json}
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analyze import paired_bootstrap
from common import DataCollector, EarthReason, ROOT, load_model, predict_logits, to_device

T = [-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
OUT = os.path.join(ROOT, "results", "calibration")


def subset(ds, idx):
    ds.images, ds.labels, ds.QAs = ([x[i] for i in idx] for x in (ds.images, ds.labels, ds.QAs))
    return ds


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    tok, model, dev = load_model()
    n_all = len(EarthReason(tok, "val").images)
    order = list(range(n_all))
    random.Random(0).shuffle(order)                     # identical to EarthReason(limit=300, seed=0)
    sets = {"test": sorted(order[:300]), "calib": sorted(order[300:600])}
    rows = []
    for name, idx in sets.items():
        ds = subset(EarthReason(tok, "val"), idx)
        for i, batch in enumerate(DataLoader(ds, batch_size=1, collate_fn=DataCollector(tok), num_workers=2)):
            b = to_device(batch, dev)
            plain = predict_logits(model, b)[0].float()
            bf = dict(b); bf["images"] = torch.flip(b["images"], (-1,))
            flip = (plain + torch.flip(predict_logits(model, bf)[0].float(), (-1,))) / 2
            gt = b["masks"][0, 0].bool()
            for mode, lg in [("plain", plain), ("flip", flip)]:
                for t in T:
                    p = lg > t
                    inter, union = (p & gt).sum().item(), (p | gt).sum().item()
                    rows.append(dict(set=name, image=batch["image_name"][0], mode=mode, t=t, inter=inter, union=union,
                                     iou=1.0 if union == 0 else inter / union))
            if i % 50 == 0:
                print(f"{name} {i + 1}/{len(ds)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "rows.csv"), index=False)

    g = df.groupby(["set", "mode", "t"]).iou.mean().unstack("t")
    summ = {"gIoU_by_threshold": {f"{s}/{m}": {str(t): round(100 * v, 2) for t, v in r.items()} for (s, m), r in g.iterrows()}}
    base = df[(df.set == "test") & (df["mode"] == "plain") & (df.t == 0.0)].set_index("image").iou
    for mode in ["plain", "flip"]:
        t_star = float(g.loc[("calib", mode)].idxmax())   # chosen on the calibration images only
        test = df[(df.set == "test") & (df["mode"] == mode) & (df.t == t_star)].set_index("image").iou
        d, lo, hi = paired_bootstrap(test.loc[base.index], base)
        summ[mode] = dict(t_star=t_star, test_gIoU=round(100 * test.mean(), 2), baseline_gIoU=round(100 * base.mean(), 2),
                          delta=round(100 * d, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2))
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "gIoU_by_threshold"}, indent=1))


if __name__ == "__main__":
    main()
