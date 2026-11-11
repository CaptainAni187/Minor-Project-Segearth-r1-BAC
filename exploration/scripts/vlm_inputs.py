"""List the refusals (SegEarth-R1 returned an empty mask) of each set with image path, size and question, for the
vision-language fallback (vlm_ground.py runs in a separate environment).

  python exploration/scripts/vlm_inputs.py calib300 test300 valrest testsplit
  -> exploration/results/vlm/inputs_{set}.json
"""
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac")); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from dump_logits import dataset  # noqa: E402

OUT = os.path.join(ROOT, "exploration", "results", "vlm")
os.makedirs(OUT, exist_ok=True)
for s in sys.argv[1:]:
    meta = json.load(open(os.path.join(ROOT, "exploration", "results", "logits", f"{s}_meta.json")))
    ds = dataset(None, s)
    rows = []
    for i, m in enumerate(meta):
        if m["pred_area_full"] != 0:
            continue
        p = ds.images[i]
        assert os.path.basename(p).split(".")[0] == m["name"]
        h, w = cv2.imread(p).shape[:2]
        rows.append(dict(name=m["name"], image=p, h=h, w=w, question=ds.QAs[i]["questions"][0].strip()))
    json.dump(rows, open(os.path.join(OUT, f"inputs_{s}.json"), "w"), indent=0)
    print(s, len(rows), "refusals")
