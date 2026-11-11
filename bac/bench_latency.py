"""Fair latency benchmark: released (64 visual tokens) vs no visual tokens, same process, same images,
interleaved so system load affects both equally. Also checks predict_tta(n=1) == predict.

  python bac/bench_latency.py [n_images]
"""
import json
import os
import sys
import time

import numpy as np
import torch

from common import DataCollector, EarthReason, ROOT, load_model, predict, predict_tta, to_device
from evaluate import NoTokens

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
tok, model, dev = load_model()
released = model.get_model().mm_projector
none = NoTokens(released)
ds = EarthReason(tok, "val", limit=n)
col = DataCollector(tok)
sync = torch.mps.synchronize if dev.type == "mps" else torch.cuda.synchronize


def timed(fn):
    model.get_model().vision_tower._key = None     # no encoder-cache hits across timed runs: every run pays full cost
    sync(); t = time.perf_counter(); out = fn(); sync()
    return out, time.perf_counter() - t


times = {"released": [], "none": [], "released_tta2": [], "none_tta2": []}
same = []
for i in range(len(ds)):
    b = to_device(col([ds[i]]), dev)
    order = ["released", "none"] if i % 2 == 0 else ["none", "released"]
    for k in order:
        model.get_model().mm_projector = released if k == "released" else none
        _, t = timed(lambda: predict(model, b))
        times[k].append(t)
    model.get_model().mm_projector = released
    p1, _ = timed(lambda: predict(model, b))
    p2, t = timed(lambda: predict_tta(model, b, 2))
    times["released_tta2"].append(t)
    same.append(bool((predict_tta(model, b, 1)[0] == p1[0]).all()))
    model.get_model().mm_projector = none
    _, t = timed(lambda: predict_tta(model, b, 2))
    times["none_tta2"].append(t)
    model.get_model().mm_projector = released

res = {k: dict(median_ms=1e3 * float(np.median(v[3:])), mean_ms=1e3 * float(np.mean(v[3:]))) for k, v in times.items()}
res["speedup_none_vs_released"] = res["released"]["median_ms"] / res["none"]["median_ms"]
res["tta1_matches_predict"] = f"{sum(same)}/{len(same)}"
res["n_timed"] = len(ds) - 3
json.dump(res, open(os.path.join(ROOT, "results", "latency_bench.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
