"""Helpers for demo.ipynb: swap connectors in one loaded model, run a query, draw figures."""
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

# load_model, predict, ... are re-exported for demo.ipynb (`from demo_utils import *`)
from common import (DataCollector, ROOT, build_sample, iou_parts, load_model, predict,  # noqa: F401
                    set_connector, to_device, boundary_grid, NO_TARGET)

VAL = os.path.join(ROOT, "data", "rs_reason_seg", "RSReasonSeg", "val")


class ConnectorBank:
    """Keeps the released connector and swaps trained ones in and out of a single model."""

    def __init__(self, model):
        self.model = model
        self.released = model.get_model().mm_projector
        self.current = "released"

    def use(self, name):
        if name == "released":
            self.model.get_model().mm_projector = self.released
        else:
            blob = torch.load(os.path.join(ROOT, "runs", name, "connector.pt"), map_location="cpu")
            conn = set_connector(self.model, blob["kind"], blob["num_tokens"], **blob.get("kwargs", {}))
            conn.load_state_dict(blob["state_dict"])
            conn.eval()
        self.current = name
        return self.model.get_model().mm_projector

    def available(self):
        runs = os.path.join(ROOT, "runs")
        return ["released"] + sorted(d for d in os.listdir(runs) if os.path.exists(os.path.join(runs, d, "connector.pt"))) \
            if os.path.isdir(runs) else ["released"]


def load_example(name):
    img = cv2.cvtColor(cv2.imread(os.path.join(VAL, "images", f"{name}.jpg")) if os.path.exists(
        os.path.join(VAL, "images", f"{name}.jpg")) else cv2.imread(os.path.join(VAL, "images", f"{name}.png")),
        cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(VAL, "labels", f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)
    qa = json.load(open(os.path.join(VAL, "QAs", f"{name}.json")))
    return img, mask, qa["questions"][0].strip(), (qa["answer"][0] if qa["answer"] else NO_TARGET)


def letterbox(img):
    """Same resize+pad as the model input, for overlaying 1024x1024 outputs."""
    h, w = img.shape[:2]
    s = 1024 / max(h, w)
    nh, nw = int(h * s), int(w * s)
    r = cv2.resize(img, (nw, nh))
    top, left = (1024 - nh) // 2, (1024 - nw) // 2
    return cv2.copyMakeBorder(r, top, 1024 - nh - top, left, 1024 - nw - left, cv2.BORDER_CONSTANT, value=0)


def run(model, tok, device, img, question, answer=NO_TARGET, gt=None):
    sample = build_sample(tok, img, question, answer, gt)
    batch = to_device(DataCollector(tok)([sample]), device)
    pred = predict(model, batch)[0].cpu()
    conn = model.get_model().mm_projector
    info = dict(pred=pred.numpy(), gt=batch["masks"][0, 0].cpu().numpy(),
                selected=None if getattr(conn, "last_selected", None) is None else conn.last_selected[0].cpu().numpy(),
                saliency=None if getattr(conn, "last_saliency", None) is None else conn.last_saliency[0].float().cpu().numpy(),
                n_boundary=getattr(conn, "num_boundary", None), num_tokens=getattr(conn, "num_tokens", 64))
    if gt is not None:
        info["iou"] = iou_parts(torch.from_numpy(info["pred"]), torch.from_numpy(info["gt"]))[2]
    return info


def overlay(ax, img, mask, color=(0.16, 0.44, 0.86), title=None, alpha=0.5):
    ax.imshow(img)
    if mask is not None and mask.any():
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask.astype(bool)] = (*color, alpha)
        ax.imshow(rgba)
        ax.contour(mask.astype(float), [0.5], colors=[color], linewidths=1.2)
    ax.set_title(title, fontsize=10); ax.axis("off")


def draw_uniform_grid(ax, img, num_tokens=64, title=None):
    ax.imshow(img)
    g = int(round(np.sqrt(num_tokens)))
    step = 1024 / g
    for i in range(1, g):
        ax.axhline(i * step, color="w", lw=0.6, alpha=.8); ax.axvline(i * step, color="w", lw=0.6, alpha=.8)
    ax.set_title(title or f"Uniform: {num_tokens} tokens, one per {1024//g}px cell", fontsize=10); ax.axis("off")


def draw_bac_allocation(ax, img, info, gt=None, title=None):
    """Saliency heatmap + the K individually kept positions (squares) + GT boundary cells."""
    ax.imshow(img)
    if info["saliency"] is not None:
        ax.imshow(cv2.resize(info["saliency"], (1024, 1024), interpolation=cv2.INTER_NEAREST),
                  cmap="magma", alpha=0.45, vmin=0, vmax=max(1e-6, info["saliency"].max()))
    if gt is not None and gt.any():
        edge = boundary_grid(torch.from_numpy(gt), dilate=0).numpy()
        ys, xs = np.nonzero(edge)
        ax.scatter(xs * 32 + 16, ys * 32 + 16, s=4, c="cyan", marker="s", alpha=.6, label="GT boundary cells")
    if info["selected"] is not None:
        for idx in info["selected"]:
            r, c = divmod(int(idx), 32)
            ax.add_patch(Rectangle((c * 32, r * 32), 32, 32, fill=False, ec="#00ff66", lw=1.4))
    k, m = info["n_boundary"], info["num_tokens"]
    ax.set_title(title or f"BAC: {k} boundary tokens (green) + {m - k} context tokens", fontsize=10)
    ax.axis("off")


def compare_figure(img, gt, results, question):
    """results: dict name -> info (from run). One row: image+GT, then each connector's mask."""
    lb = letterbox(img)
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.9))
    overlay(axes[0], lb, gt, color=(0.1, 0.8, 0.3), title="ground truth")
    for ax, (name, info) in zip(axes[1:], results.items()):
        t = f"{name}" + (f"  IoU {100*info['iou']:.1f}" if "iou" in info else "")
        overlay(ax, lb, info["pred"], title=t)
    fig.suptitle(question[:140] + ("..." if len(question) > 140 else ""), fontsize=9)
    fig.tight_layout()
    return fig
