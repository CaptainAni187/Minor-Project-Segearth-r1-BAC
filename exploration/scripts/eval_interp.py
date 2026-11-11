"""WiSE-FT style weight interpolation of the mask path (Wortsman et al., CVPR 2022).

Fine-tuning the mask path with D4 augmentation (exploration E9) helped seen object types but hurt unseen ones.
Interpolating weights, theta = (1 - a) * released + a * fine-tuned, is a published way to keep most of the
in-distribution gain while recovering robustness. Only the mask-path parameters change; everything else is released.

  python exploration/scripts/eval_interp.py calib300 --alpha 0.5 [--ft ft_d4] [--tta 8]
  -> exploration/results/interp/{set}_{ft}_a{alpha}_tta{n}.csv  (+ .json summary)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac")); sys.path.insert(0, HERE)
from common import DataCollector, load_model, predict, predict_tta, to_device  # noqa: E402
from dump_logits import dataset  # noqa: E402

OUT = os.path.join(ROOT, "exploration", "results", "interp")


@torch.no_grad()
def main():
    a = sys.argv[1:]
    sets = [x for x in a if not x.startswith("--") and not x.replace(".", "").isdigit() and x not in ("ft_d4", "ft_ctrl")]
    alpha = float(a[a.index("--alpha") + 1])
    ft = a[a.index("--ft") + 1] if "--ft" in a else "ft_d4"
    tta = int(a[a.index("--tta") + 1]) if "--tta" in a else 1
    tok, model, dev = load_model()
    blob = torch.load(os.path.join(ROOT, "exploration", "ft_weights", "runs", ft, "connector.pt"), map_location="cpu", weights_only=False)
    ftw = blob["mask_path"]
    sd = model.state_dict()
    new = {k: ((1 - alpha) * sd[k].float().cpu() + alpha * v.float()).to(sd[k].dtype) for k, v in ftw.items()}
    res = model.load_state_dict(new, strict=False)
    assert not res.unexpected_keys
    model.to(dev)
    print(f"interpolated {len(new)} mask-path tensors with alpha={alpha} ({ft})", flush=True)
    os.makedirs(OUT, exist_ok=True)
    for s in sets:
        ds = dataset(tok, s)
        rows, t0 = [], time.time()
        for i, batch in enumerate(DataLoader(ds, batch_size=1, collate_fn=DataCollector(tok), num_workers=2)):
            b = to_device(batch, dev)
            pred = predict(model, b)[0] if tta == 1 else predict_tta(model, b, tta)[0]
            gt = b["masks"][0, 0].bool()
            it, un = (pred & gt).sum().item(), (pred | gt).sum().item()
            rows.append(dict(name=batch["image_name"][0], inter=it, union=un, iou=1.0 if un == 0 else it / un, pred_empty=int(pred.sum() == 0)))
            if i % 50 == 0:
                print(f"[{s} a={alpha} tta={tta}] {i + 1}/{len(ds)} gIoU {100 * pd.DataFrame(rows).iou.mean():.2f} {(time.time() - t0) / (i + 1):.2f}s/img", flush=True)
        df = pd.DataFrame(rows)
        tag = f"{s}_{ft}_a{alpha}_tta{tta}"
        df.to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
        summ = dict(set=s, ft=ft, alpha=alpha, tta=tta, n=len(df), gIoU=100 * df.iou.mean(), cIoU=100 * df.inter.sum() / df.union.sum())
        json.dump(summ, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
        print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
