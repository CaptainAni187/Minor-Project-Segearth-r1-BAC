"""Open-vocabulary grounding branch: predicted object name -> OWLv2 boxes -> SAM masks (top 5 boxes per image).

The name comes from predict_noun.py (question text only), mode 'free' or 'vocab'; 'oracle' (ground-truth category
of held-out images) exists only for the diagnostic ceiling and is never used for reported results.
Stores, per image, the 5 candidate masks (512x512, packed), OWLv2 scores and SAM predicted IoUs, for the
policy that is chosen on calib300 (screen_hybrid.py).

  python exploration/scripts/ov_branch.py calib300 test300 --nouns Qwen2.5-3B-Instruct --mode free
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac")); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from demo_utils import letterbox  # noqa: E402
from dump_logits import dataset  # noqa: E402
from ov_ceiling import Grounder  # noqa: E402
from eval_dataset.RS_val_dataset import preprocess_mask  # noqa: E402

K = 5


def main():
    args = sys.argv[1:]
    nouns = args[args.index("--nouns") + 1]
    mode = args[args.index("--mode") + 1]
    sam = args[args.index("--sam") + 1] if "--sam" in args else "base"
    owl_size = args[args.index("--owl") + 1] if "--owl" in args else "base"
    sets = [a for a in args if not a.startswith("--") and a not in (nouns, mode, sam, owl_size)]
    g = Grounder(sam, "google/owlv2-large-patch14-ensemble" if owl_size == "large" else "google/owlv2-base-patch16-ensemble")
    out_dir = os.path.join(ROOT, "exploration", "results", "ov")
    os.makedirs(out_dir, exist_ok=True)
    for s in sets:
        meta = json.load(open(os.path.join(ROOT, "exploration", "results", "logits", f"{s}_meta.json")))
        pred = json.load(open(os.path.join(ROOT, "exploration", "results", "nouns", f"{s}_{nouns}.json")))["predictions"]
        ds = dataset(None, s)
        idx = [i for i, m in enumerate(meta) if m["pred_area_full"] == 0] if "--refusals-only" in args else list(range(len(meta)))
        n = len(idx)
        cands = np.zeros((n, K, 512, 512), bool)
        owl = np.zeros((n, K), np.float32)
        samsc = np.zeros((n, K), np.float32)
        iou_full = np.zeros((n, K), np.float32)       # exact metric at the evaluation resolution (1024 letterbox)
        inter_full = np.zeros((n, K), np.int64); union_full = np.zeros((n, K), np.int64)
        used = []
        t0 = time.time()
        for j, i in enumerate(idx):
            m = meta[i]
            noun = pred[m["name"]][mode]
            used.append(noun)
            assert os.path.basename(ds.images[i]).split(".")[0] == m["name"], (ds.images[i], m["name"])
            lb = letterbox(cv2.cvtColor(cv2.imread(ds.images[i]), cv2.COLOR_BGR2RGB))
            b, sc = g.boxes(lb, f"a satellite image of a {noun}")
            mk, ms = g.masks(lb, b[:K])
            lab = cv2.imread(ds.labels[i], cv2.IMREAD_GRAYSCALE); lab[lab != 0] = 1
            gt = preprocess_mask(lab, 1024)[0].numpy().astype(bool)          # same as the upstream evaluation
            assert gt.sum() == m["gt_area_full"], (m["name"], gt.sum(), m["gt_area_full"])
            for k in range(min(K, len(mk))):
                cands[j, k] = cv2.resize(mk[k].astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA) > 0
                u = (mk[k] | gt).sum(); it = (mk[k] & gt).sum()
                iou_full[j, k] = 1.0 if u == 0 else it / u
                inter_full[j, k], union_full[j, k] = it, u
            owl[j, :len(sc[:K])] = sc[:K]
            samsc[j, :len(ms)] = ms
            if j % 25 == 0:
                print(f"[{s}/{mode}/{sam}] {j + 1}/{n} {(time.time() - t0) / (j + 1):.2f}s/img  '{noun}'", flush=True)
        tag = "_refusals" if "--refusals-only" in args else ""
        otag = "" if owl_size == "base" else f"_owl{owl_size}"
        np.savez_compressed(os.path.join(out_dir, f"{s}_{nouns}_{mode}_{sam}{otag}{tag}.npz"), cands=np.packbits(cands, axis=-1),
                            owl=owl, sam=samsc, nouns=np.array(used), iou_full=iou_full, inter_full=inter_full,
                            union_full=union_full, names=np.array([meta[i]["name"] for i in idx]))
        print(f"[{s}/{mode}/{sam}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
