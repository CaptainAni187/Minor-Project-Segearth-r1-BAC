"""Ceiling of an open-vocabulary grounding branch (diagnostic only, calib300).

Question: if the target's *name* were known, can an open-vocabulary detector (OWLv2) + SAM segment the unseen
categories that SegEarth-R1 misses in aerial images? The name here comes from the ground-truth category, so this is
an upper bound for the grounding stage only; it is never used as a reported method. A second pass replaces the
oracle name by a name predicted from the question text alone (predict_noun), which is what a real system can use.

  python exploration/scripts/ov_ceiling.py [--all]   (default: held-out-category images only)
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac")); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from demo_utils import letterbox  # noqa: E402
from dump_logits import dataset  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def to_dev(batch, dev):
    """Move a processor output to the device, casting float64 (unsupported on MPS) to float32."""
    return {k: (v.float() if v.dtype == torch.float64 else v).to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


class Grounder:
    def __init__(self, sam_name="base", owl_name="google/owlv2-base-patch16-ensemble"):
        self.op = Owlv2Processor.from_pretrained(owl_name)
        self.ow = Owlv2ForObjectDetection.from_pretrained(owl_name).eval()   # CPU: OWLv2 uses float64 (unsupported on MPS)
        self.sp = SamProcessor.from_pretrained(f"facebook/sam-vit-{sam_name}")
        self.sam = SamModel.from_pretrained(f"facebook/sam-vit-{sam_name}").to(DEV).eval()

    @torch.no_grad()
    def boxes(self, lb, text):
        inp = self.op(text=[[text]], images=lb, return_tensors="pt", truncation=True, max_length=16)   # OWLv2 text limit
        out = self.ow(**inp)
        # OWLv2 pads to a square; our letterboxed image is already 1024x1024 square
        res = self.op.post_process_object_detection(out, threshold=0.0, target_sizes=torch.tensor([[1024, 1024]]))[0]
        o = res["scores"].argsort(descending=True)
        return res["boxes"][o].cpu().numpy(), res["scores"][o].cpu().numpy()

    @torch.no_grad()
    def masks(self, lb, boxes):
        inp = to_dev(self.sp(lb, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt"), DEV)
        out = self.sam(**inp, multimask_output=True)
        m = self.sp.image_processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        sc = out.iou_scores[0].cpu().numpy()                       # [n_boxes, 3]
        best = sc.argmax(1)
        return np.stack([m[i, best[i]].numpy() for i in range(len(boxes))]), sc.max(1)


def main():
    allimg = "--all" in sys.argv
    meta = json.load(open(os.path.join(ROOT, "exploration", "results", "logits", "calib300_meta.json")))
    z = np.load(os.path.join(ROOT, "exploration", "results", "logits", "calib300.npz"))
    ds = dataset(None, "calib300")
    g = Grounder()
    rows = []
    t0 = time.time()
    for i, m in enumerate(meta):
        held = m["category"].startswith("held-out")
        if not (held or allimg) or m["gt_area_full"] == 0:
            continue
        noun = m["category"].split(": ")[1] if held else None
        if noun is None:
            continue
        lb = letterbox(cv2.cvtColor(cv2.imread(ds.images[i]), cv2.COLOR_BGR2RGB))
        gt = cv2.resize(z["gt"][i].astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)
        b, s = g.boxes(lb, f"a satellite image of a {noun}")
        top = b[:5]
        mk, ms = g.masks(lb, top)
        r = dict(name=m["name"], noun=noun, seg_iou=m["iou_full"], owl_top_score=float(s[0]),
                 iou_top1=float(iou(mk[0], gt)), iou_best_of5=float(max(iou(x, gt) for x in mk)),
                 iou_union_ge=float(iou(mk[s[:5] >= 0.5 * s[0]].any(0), gt)))
        rows.append(r)
        print(r, f"{(time.time() - t0) / len(rows):.1f}s/img", flush=True)
    out = os.path.join(ROOT, "exploration", "results", "ov_ceiling_calib300.json")
    json.dump(rows, open(out, "w"), indent=1)
    import pandas as pd
    d = pd.DataFrame(rows)
    print(d.groupby("noun")[["seg_iou", "iou_top1", "iou_best_of5", "iou_union_ge"]].mean().round(3))
    print("ALL held-out", d[["seg_iou", "iou_top1", "iou_best_of5", "iou_union_ge"]].mean().round(3).to_dict(), "n =", len(d))


if __name__ == "__main__":
    main()
