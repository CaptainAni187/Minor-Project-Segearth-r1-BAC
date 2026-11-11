"""Vision-language fallback, step 3: VLM box (original pixels) -> letterboxed 1024 frame -> SAM mask -> exact IoU.
Writes the same format as ov_branch.py (refusals only) so score_policy.py can score it; 'owl' holds 1 if the VLM gave
a box and 0 if it said nothing answers the question (so tau = 0.5 means "use the VLM box when it gives one").

  python exploration/scripts/vlm_to_masks.py calib300 test300 valrest testsplit [--sam base]
  -> exploration/results/ov/{set}_Qwen2.5-VL-3B_ground_{sam}_refusals.npz
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac")); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from demo_utils import letterbox  # noqa: E402
from eval_dataset.RS_val_dataset import preprocess_mask  # noqa: E402
from ov_ceiling import Grounder  # noqa: E402
from dump_logits import dataset  # noqa: E402

args = sys.argv[1:]
sam = args[args.index("--sam") + 1] if "--sam" in args else "base"
sets = [a for a in args if not a.startswith("--") and a != sam]
g = Grounder(sam)
for s in sets:
    inputs = {r["name"]: r for r in json.load(open(os.path.join(ROOT, "exploration", "results", "vlm", f"inputs_{s}.json")))}
    boxes = json.load(open(os.path.join(ROOT, "exploration", "results", "vlm", f"boxes_{s}.json")))
    ds = dataset(None, s)
    path_to_label = dict(zip([os.path.basename(p).split(".")[0] for p in ds.images], ds.labels))
    n = len(boxes)
    iou, it, un, has = np.zeros((n, 1)), np.zeros((n, 1), np.int64), np.zeros((n, 1), np.int64), np.zeros((n, 1), np.float32)
    cands = np.zeros((n, 1, 512, 512), bool)
    for j, b in enumerate(boxes):
        r = inputs[b["name"]]
        lab = cv2.imread(path_to_label[b["name"]], cv2.IMREAD_GRAYSCALE); lab[lab != 0] = 1
        gt = preprocess_mask(lab, 1024)[0].numpy().astype(bool)
        if b["box"]:
            sc = 1024 / max(r["h"], r["w"])                     # letterbox: scale the longest side to 1024, pad centred
            top, left = (1024 - int(r["h"] * sc)) // 2, (1024 - int(r["w"] * sc)) // 2
            x1, y1, x2, y2 = b["box"]
            bb = [x1 * sc + left, y1 * sc + top, x2 * sc + left, y2 * sc + top]
            lb = letterbox(cv2.cvtColor(cv2.imread(r["image"]), cv2.COLOR_BGR2RGB))
            mk = g.masks(lb, [bb])[0][0]
            has[j, 0] = 1.0
        else:
            mk = np.zeros((1024, 1024), bool)               # "nothing answers the question": keep the empty answer
        u = (mk | gt).sum(); t = (mk & gt).sum()
        iou[j, 0], it[j, 0], un[j, 0] = (1.0 if u == 0 else t / u), t, u
        cands[j, 0] = cv2.resize(mk.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA) > 0
    np.savez_compressed(os.path.join(ROOT, "exploration", "results", "ov", f"{s}_Qwen2.5-VL-3B_ground_{sam}_refusals.npz"),
                        cands=np.packbits(cands, axis=-1), owl=has, sam=has, nouns=np.array([b["label"] for b in boxes]),
                        iou_full=iou, inter_full=it, union_full=un, names=np.array([b["name"] for b in boxes]))
    print(s, n, "refusals |", int(has.sum()), "with a VLM box | mean IoU on boxed", round(float(iou[has > 0].mean()) if has.sum() else 0, 3))
