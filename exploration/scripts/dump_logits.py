"""Dump the released model's raw mask logits once, so post-processing rules can be screened offline.

Sets (all from EarthReason, same seeded shuffle that defines our 300-image test subset; disjoint by construction):
  test300  : val order[:300]   (the subset every reported number uses)
  calib300 : val order[300:600] (for choosing any rule / threshold; never used for reporting)
  testsplit: the official EarthReason *test* split (final confirmation only), optionally limited

For every image: logits of the plain pass and of the horizontally flipped pass (flipped back), stored at 512x512
(fp16, bilinear from 1024x1024), the ground truth at 512x512, and full-resolution reference numbers (IoU at 0,
predicted / GT areas, max logit) to check that the 512 approximation tracks the real metric.

  python exploration/scripts/dump_logits.py test300 calib300
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac"))
from analyze import category_of  # noqa: E402
from common import DataCollector, EarthReason, load_model, predict_logits, to_device  # noqa: E402

OUT = os.path.join(ROOT, "exploration", "results", "logits")
R = 512


def take(ds, idx):
    ds.images, ds.labels, ds.QAs = ([x[i] for i in idx] for x in (ds.images, ds.labels, ds.QAs))
    return ds


def dataset(tok, name, limit=None):
    if name == "testsplit":
        ds = EarthReason(tok, "test")
        if limit:
            order = list(range(len(ds.images)))
            random.Random(0).shuffle(order)
            ds = take(ds, sorted(order[:limit]))
        return ds
    n = len(EarthReason(tok, "val").images)
    order = list(range(n))
    random.Random(0).shuffle(order)                      # identical to EarthReason(limit=300, seed=0)
    idx = {"test300": sorted(order[:300]), "calib300": sorted(order[300:600]), "valrest": sorted(order[600:])}[name]
    return take(EarthReason(tok, "val"), idx)


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    sets = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    noflip = "--no-flip" in sys.argv
    tok, model, dev = load_model()
    for name in sets:
        ds = dataset(tok, name, limit)
        n = len(ds.images)
        keep_arrays = "--meta-only" not in sys.argv
        plain = np.zeros((n if keep_arrays else 1, R, R), np.float16)   # meta-only: no 2.4 GB of unused arrays
        flip = np.zeros_like(plain)
        gts = np.zeros(plain.shape, bool)
        meta = []
        t0 = time.time()
        for i, batch in enumerate(DataLoader(ds, batch_size=1, collate_fn=DataCollector(tok), num_workers=2)):
            b = to_device(batch, dev)
            lp = predict_logits(model, b)[0].float()
            if noflip:                                     # flip TTA was rejected on calib300 (NOTES E3)
                lf = lp
            else:
                bf = dict(b); bf["images"] = torch.flip(b["images"], (-1,))
                lf = torch.flip(predict_logits(model, bf)[0].float(), (-1,))
            gt = b["masks"][0, 0].bool()
            pred = lp > 0
            inter, union = (pred & gt).sum().item(), (pred | gt).sum().item()
            if keep_arrays:
                plain[i] = F.interpolate(lp[None, None], size=R, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                flip[i] = F.interpolate(lf[None, None], size=R, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                gts[i] = F.interpolate(gt[None, None].float(), size=R, mode="area")[0, 0].cpu().numpy() > 0.5
            nm = batch["image_name"][0]
            meta.append(dict(name=nm, category=category_of(nm, "test" if name == "testsplit" else "val"),
                             iou_full=1.0 if union == 0 else inter / union, inter_full=inter, union_full=union,
                             pred_area_full=int(pred.sum()), gt_area_full=int(gt.sum()), max_logit=float(lp.max()),
                             question=ds.QAs[i]["questions"][0] if isinstance(ds.QAs[i], dict) and "questions" in ds.QAs[i] else None))
            if i % 25 == 0:
                print(f"[{name}] {i + 1}/{n}  {(time.time() - t0) / (i + 1):.2f}s/img", flush=True)
        if "--meta-only" not in sys.argv:                  # the test split only needs the per-image metadata
            np.savez(os.path.join(OUT, f"{name}.npz"), plain=plain, flip=flip, gt=gts)
        json.dump(meta, open(os.path.join(OUT, f"{name}_meta.json"), "w"), indent=0)
        g = np.mean([m["iou_full"] for m in meta])
        print(f"[{name}] done: {n} images, full-res gIoU at threshold 0 = {100 * g:.2f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
