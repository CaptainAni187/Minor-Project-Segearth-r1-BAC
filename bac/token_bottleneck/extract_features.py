"""Extract the inputs of the token-bottleneck testbed once, with SegEarth-R1's own frozen components.

For every training image and every image of the fixed 300-image validation subset:
  - Swin-B v4 features [1024, 32, 32] (fp16)        -> data/bottleneck/{split}_v4.npy  (memmap)
  - ground-truth mask, letterboxed, 256x256 (uint8)  -> data/bottleneck/{split}_mask.npy
  - question embeddings: mean of Phi-1.5's last hidden states over the question (text only, no image)
                                                      -> data/bottleneck/{split}_q.pt  (list of [n_q, 2048])

  python bac/token_bottleneck/extract_features.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import DATA, EarthReason, load_model  # noqa: E402

OUT = os.path.join(DATA, "bottleneck")
os.makedirs(OUT, exist_ok=True)


@torch.no_grad()
def question_embedding(model, tok, text, dev):
    ids = tok(" " + text.strip(), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    emb = model.get_model().embed_tokens(ids)
    h = model.get_model()(inputs_embeds=emb, output_attentions=False, return_dict=True).last_hidden_state
    return h[0].mean(0).half().cpu()


def main():
    global OUT
    smoke = int(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else 0
    if smoke:      # environment check: a few val images serve as both splits, written to a separate folder
        OUT = os.path.join(DATA, "bottleneck_smoke")
        os.makedirs(OUT, exist_ok=True)
    tok, model, dev = load_model()
    tower = model.get_model().get_vision_tower()
    for split, limit in ([("val", smoke), ("train", smoke)] if smoke else [("val", 300), ("train", None)]):
        ds = EarthReason(tok, "val" if smoke else split, limit=limit)
        n = len(ds)
        v4 = np.lib.format.open_memmap(os.path.join(OUT, f"{split}_v4.npy"), mode="w+", dtype=np.float16, shape=(n, 1024, 32, 32))
        masks = np.zeros((n, 256, 256), dtype=np.uint8)
        qs, names = [], []
        for i in range(n):
            d = ds[i]
            with torch.no_grad():
                feats = tower(d["image"][None].float().to(dev))
            v4[i] = feats[-1][0].half().cpu().numpy()
            tower._key = None
            masks[i] = (F.adaptive_avg_pool2d(d["mask"].float()[None], 256)[0, 0] > 0.5).numpy().astype(np.uint8)
            questions = ds.QAs[i]["questions"] if split == "train" else ds.QAs[i]["questions"][:1]
            qs.append(torch.stack([question_embedding(model, tok, q, dev) for q in questions]))
            names.append(d["image_name"])
            if i % 200 == 0:
                print(f"{split} {i}/{n}", flush=True)
        v4.flush()
        np.save(os.path.join(OUT, f"{split}_mask.npy"), masks)
        torch.save({"q": qs, "names": names}, os.path.join(OUT, f"{split}_q.pt"))
        print(f"{split}: {n} images done")


if __name__ == "__main__":
    main()
