"""Choose how to combine SegEarth-R1 with the SAM candidates (calib300 only), then report once on test300.

Candidates per image (from sam_refine.py): 0-2 = box prompt (3 multimask outputs), 3-5 = box + peak point.
Policies (all use only the model's own outputs and SAM's own predicted IoU, never the ground truth):
  seg            : SegEarth-R1 as released (threshold 0)
  sam_k          : always replace with SAM candidate k
  sam_best_score : always replace with the candidate SAM scores highest (within a prompt type)
  sam_agree      : the candidate that best agrees (IoU) with SegEarth-R1's own region
  fallback_*     : keep SegEarth-R1, use SAM only where SegEarth-R1 predicted nothing
  gated_*        : replace with SAM only if SAM's predicted IoU >= tau and agreement with SegEarth-R1 >= kappa

  python exploration/scripts/screen_sam.py base plain
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac"))
from analyze import paired_bootstrap  # noqa: E402

LG = os.path.join(ROOT, "exploration", "results", "logits")
SM = os.path.join(ROOT, "exploration", "results", "sam")


def iou(a, b):
    inter = (a & b).sum((-2, -1)); union = (a | b).sum((-2, -1))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def load(name, model, src):
    z = np.load(os.path.join(LG, f"{name}.npz"))
    s = np.load(os.path.join(SM, f"{name}_{model}_{src}.npz"))
    lg = z[src].astype(np.float32)
    cands = np.unpackbits(s["cands"], axis=-1)[..., :512].astype(bool)
    return dict(seg=lg > 0, logits=lg, gt=z["gt"], cands=cands, scores=s["scores"], fb=s["used_fallback"],
                meta=json.load(open(os.path.join(LG, f"{name}_meta.json"))))


def region(d):
    """SegEarth-R1's own region, with the same fallback band SAM was prompted with."""
    mx = d["logits"].reshape(len(d["logits"]), -1).max(1)
    fbr = d["logits"] > (mx - 2.0)[:, None, None]
    return np.where(d["fb"][:, None, None], fbr, d["seg"])


def policies(d):
    seg, c, sc = d["seg"], d["cands"], d["scores"]
    n = len(seg)
    agree = iou(c, region(d)[:, None])                                    # [n, 6] agreement with SegEarth-R1
    out = {"seg": seg}
    for k in range(6):
        out[f"sam_{k}"] = c[:, k]
    for name, sl in [("box", slice(0, 3)), ("boxpt", slice(3, 6))]:
        kk = np.argmax(sc[:, sl], 1) + sl.start
        out[f"sam_best_score_{name}"] = c[np.arange(n), kk]
        ka = np.argmax(agree[:, sl], 1) + sl.start
        out[f"sam_agree_{name}"] = c[np.arange(n), ka]
        out[f"fallback_{name}"] = np.where(d["fb"][:, None, None], c[np.arange(n), kk], seg)
        for tau in [0.7, 0.8, 0.85, 0.9, 0.95]:
            for kappa in [0.0, 0.3, 0.5, 0.7]:
                use = (sc[np.arange(n), kk] >= tau) & (agree[np.arange(n), kk] >= kappa)
                use |= d["fb"]
                out[f"gated_{name}_tau{tau}_k{kappa}"] = np.where(use[:, None, None], c[np.arange(n), kk], seg)
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "base"
    src = sys.argv[2] if len(sys.argv) > 2 else "plain"
    cal, tst = load("calib300", model, src), load("test300", model, src)
    pc, pt = policies(cal), policies(tst)
    base_c, base_t = iou(pc["seg"], cal["gt"]), iou(pt["seg"], tst["gt"])
    rows = []
    for k in pc:
        rows.append((k, 100 * (iou(pc[k], cal["gt"]).mean() - base_c.mean())))
    rows.sort(key=lambda r: -r[1])
    print(f"calib300 baseline (512 screen) {100 * base_c.mean():.2f}; top policies by calib gain:")
    for k, g in rows[:12]:
        print(f"  {k:32s} {g:+.2f}")
    best = rows[0][0]
    res = {}
    for k in dict.fromkeys([best, "sam_best_score_box", "sam_best_score_boxpt", "fallback_box", "fallback_boxpt", "sam_agree_box"]):
        t = iou(pt[k], tst["gt"])
        d, lo, hi = paired_bootstrap(t, base_t)
        cats = np.array(["held-out" if m["category"].startswith("held-out") else m["category"].replace("seen category", "seen")
                         for m in tst["meta"]])
        res[k] = dict(chosen_on_calib=k == best, calib_gain=round(dict(rows)[k], 2), test_gIoU=round(100 * t.mean(), 2),
                      test_delta=round(100 * d, 2), ci=[round(100 * lo, 2), round(100 * hi, 2)],
                      by_category={c: [round(100 * base_t[cats == c].mean(), 1), round(100 * t[cats == c].mean(), 1), int((cats == c).sum())]
                                   for c in sorted(set(cats))})
    print(f"test300 baseline (512 screen) {100 * base_t.mean():.2f}")
    for k, v in res.items():
        print(k, v)
    json.dump(dict(model=model, src=src, calib_ranking=rows[:20], test=res),
              open(os.path.join(ROOT, "exploration", "results", f"screen_sam_{model}_{src}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
