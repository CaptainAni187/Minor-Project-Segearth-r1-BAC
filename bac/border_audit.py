"""Where do BAC's kept tokens go? Fraction on the image border ring and on letterbox padding.

  python bac/border_audit.py bac64 bac16
Only the frozen encoder + connector run (no LLM), so this is fast.
"""
import json
import os
import sys

import numpy as np
import torch

from common import EarthReason, ROOT, load_model, set_connector

names = sys.argv[1:] or ["bac64"]
tok, model, dev = load_model()
ds = EarthReason(tok, "val", limit=300)
out = {}
for name in names:
    blob = torch.load(os.path.join(ROOT, "runs", name, "connector.pt"), map_location="cpu")
    conn = set_connector(model, blob["kind"], blob["num_tokens"], **blob.get("kwargs", {}))
    conn.load_state_dict(blob["state_dict"]); conn.eval()
    ring, pad, ring_chance, pad_chance = [], [], [], []
    for i in range(len(ds)):
        d = ds[i]
        img = d["image"][None].to(dev)
        with torch.no_grad():
            conn(model.get_model().get_vision_tower()(img)[-1])
        idx = conn.last_selected[0].cpu().numpy()
        r, c = idx // 32, idx % 32
        on_ring = (r <= 1) | (r >= 30) | (c <= 1) | (c >= 30)          # outer two cells of the 32x32 grid
        # letterbox padding: grid cells whose 32x32 pixel block is entirely padding (normalised value of 0 px)
        raw = (d["image"] * torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
               + torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)).sum(0)
        padcell = (torch.nn.functional.avg_pool2d(raw[None, None], 32)[0, 0] < 1e-3).numpy()
        on_pad = padcell[r, c]
        ring.append(on_ring.mean()); pad.append(on_pad.mean())
        grid_ring = np.zeros((32, 32), bool); grid_ring[:2] = grid_ring[-2:] = True; grid_ring[:, :2] = grid_ring[:, -2:] = True
        ring_chance.append(grid_ring.mean()); pad_chance.append(padcell.mean())
    out[name] = dict(border_ring=float(np.mean(ring)), border_ring_chance=float(np.mean(ring_chance)),
                     padding=float(np.mean(pad)), padding_chance=float(np.mean(pad_chance)), n=len(ring))
    print(name, json.dumps(out[name]))
json.dump(out, open(os.path.join(ROOT, "results", "border_audit.json"), "w"), indent=1)
