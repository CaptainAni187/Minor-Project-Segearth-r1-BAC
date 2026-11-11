"""SAM refinement prompted only by SegEarth-R1's own output (no ground truth at inference).

Hypothesis: SegEarth-R1's Mask2Former decoder was trained only on the seen categories, so for unseen categories it
often refuses (all logits < 0) or covers only part of the object, while its logits still peak on the target. SAM is
category-agnostic (SA-1B), and EarthReason's masks were themselves annotated with SAM-H assistance, so SAM masks
should match the ground-truth style. We prompt SAM with a box and a point taken from SegEarth-R1's logits:
  region = logits > 0, or, if that is empty, logits > max - DELTA (the model's best guess)
  box    = bounding box of the region; point = arg-max of the logits inside the region
and store SAM's three multimask candidates for box-only and box+point prompts, with SAM's predicted IoU scores.
The policy that picks between SegEarth-R1 and the SAM candidates is chosen later on calib300 only.

  python exploration/scripts/sam_refine.py calib300 test300 [--model base|large|huge] [--src plain|flip]
  -> exploration/results/sam/{set}_{model}_{src}.npz
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from transformers import SamModel, SamProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac"))
sys.path.insert(0, HERE)
import common  # noqa: E402,F401  (import paths, env vars)
from demo_utils import letterbox  # noqa: E402
from dump_logits import dataset  # noqa: E402

DELTA = 2.0          # fallback band below the maximum logit when SegEarth-R1 predicts nothing (see screen_rules)


def to_dev(batch, dev):
    """Move a processor output to the device, casting float64 (unsupported on MPS) to float32."""
    return {k: (v.float() if v.dtype == torch.float64 else v).to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
OUT = os.path.join(ROOT, "exploration", "results", "sam")


def prompts(lg):
    """lg: [512, 512] logits -> (box_xyxy at 1024, point_xy at 1024, used_fallback)."""
    region = lg > 0
    fb = not region.any()
    if fb:
        region = lg > lg.max() - DELTA
    ys, xs = np.nonzero(region)
    box = [xs.min() * 2, ys.min() * 2, xs.max() * 2 + 1, ys.max() * 2 + 1]
    masked = np.where(region, lg, -np.inf)
    py, px = np.unravel_index(np.argmax(masked), lg.shape)
    return [float(v) for v in box], [float(px * 2 + 1), float(py * 2 + 1)], fb


@torch.no_grad()
def main():
    args = sys.argv[1:]
    model_name = args[args.index("--model") + 1] if "--model" in args else "base"
    src = args[args.index("--src") + 1] if "--src" in args else "plain"
    sets = [a for a in args if not a.startswith("--") and a not in (model_name, src)]
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    proc = SamProcessor.from_pretrained(f"facebook/sam-vit-{model_name}")
    sam = SamModel.from_pretrained(f"facebook/sam-vit-{model_name}").to(dev).eval()
    os.makedirs(OUT, exist_ok=True)
    for name in sets:
        z = np.load(os.path.join(ROOT, "exploration", "results", "logits", f"{name}.npz"))
        logits = z[src].astype(np.float32)
        ds = dataset(None, name)
        n = len(ds.images)
        assert n == len(logits), (n, len(logits))
        cands = np.zeros((n, 6, 512, 512), bool)
        scores = np.zeros((n, 6), np.float32)
        used_fb = np.zeros(n, bool)
        t0 = time.time()
        for i in range(n):
            rgb = cv2.cvtColor(cv2.imread(ds.images[i]), cv2.COLOR_BGR2RGB)
            lb = letterbox(rgb)
            box, pt, fb = prompts(logits[i])
            used_fb[i] = fb
            inp = to_dev(proc(lb, return_tensors="pt"), dev)
            emb = sam.get_image_embeddings(inp["pixel_values"])
            for j, kw in enumerate([dict(input_boxes=[[box]]), dict(input_boxes=[[box]], input_points=[[[pt]]], input_labels=[[[1]]])]):
                p = to_dev(proc(lb, return_tensors="pt", **kw), dev)
                p.pop("pixel_values")
                out = sam(image_embeddings=emb, multimask_output=True, **{k: v for k, v in p.items() if k.startswith("input_")})
                masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), p["original_sizes"].cpu(),
                                                                p["reshaped_input_sizes"].cpu())[0][0]   # [3, 1024, 1024] bool
                m512 = torch.nn.functional.interpolate(masks[None].float(), size=512, mode="area")[0] > 0.5
                cands[i, 3 * j:3 * j + 3] = m512.numpy()
                scores[i, 3 * j:3 * j + 3] = out.iou_scores[0, 0].float().cpu().numpy()
            if i % 25 == 0:
                print(f"[{name}/{model_name}/{src}] {i + 1}/{n}  {(time.time() - t0) / (i + 1):.2f}s/img", flush=True)
        np.savez_compressed(os.path.join(OUT, f"{name}_{model_name}_{src}.npz"), cands=np.packbits(cands, axis=-1),
                            scores=scores, used_fallback=used_fb, delta=DELTA)
        print(f"[{name}/{model_name}/{src}] done in {(time.time() - t0) / 60:.1f} min; fallback used on {used_fb.sum()} images", flush=True)


if __name__ == "__main__":
    main()
