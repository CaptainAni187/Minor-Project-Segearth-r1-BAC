"""Compare a fresh run (e.g. on Kaggle) with the reference results produced on the Apple M4, and check every
headline claim of the project. Writes results/verification_report.md and prints a PASS/FAIL table.

  python bac/verify_against_reference.py [--ref reference_results] [--new results]

Tolerances: inference-only configurations use the released weights, so only floating-point differences between
devices are expected (|delta gIoU| <= 0.5). Trained connectors are re-trained from scratch on different hardware and
precision (fp32 on CUDA vs bf16 on MPS), so they are compared loosely (|delta| <= 1.5) and by the conclusions they
support rather than by exact numbers.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from analyze import paired_bootstrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFERENCE_ONLY = ["released", "zero", "wrong", "wrong_far", "none", "dproj", "dproj_zero", "wrongq", "dproj_zero_wrongq",
                  "released_tta2", "none_tta2"]
PAPER_VAL_GIOU = 68.60   # SegEarth-R1, EarthReason val, full split (Li et al., 2025, Table 5)
# Note: our 300-image subset is not the full split. On all 1,135 validation images the released model scores 70.59;
# on the official test split 70.40 (paper 70.75), which is the like-for-like reproduction check.


def load(folder, name):
    p = os.path.join(folder, f"{name}_val.csv")
    return pd.read_csv(p, dtype={"name": str}) if os.path.exists(p) else None


def giou(df):
    return 100 * df.iou.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(ROOT, "reference_results"))
    ap.add_argument("--new", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    names = sorted({f[:-8] for f in os.listdir(args.new) if f.endswith("_val.csv") and not f.startswith("boundary")})
    rows, lines = [], []
    for n in names:
        new, ref = load(args.new, n), load(args.ref, n)
        if new is None:
            continue
        tol = 0.5 if n in INFERENCE_ONLY else 1.5
        d = None if ref is None else giou(new) - giou(ref)
        rows.append(dict(config=n, kind="inference" if n in INFERENCE_ONLY else "trained", n_images=len(new),
                         gIoU_new=round(giou(new), 2), gIoU_ref=None if ref is None else round(giou(ref), 2),
                         delta=None if d is None else round(d, 2), tolerance=tol,
                         status="no reference" if d is None else ("PASS" if abs(d) <= tol else "FAIL")))
    table = pd.DataFrame(rows)

    # claim checks (on the NEW results only: they must stand on their own)
    claims = []

    def claim(text, ok, detail):
        claims.append(dict(claim=text, status="PASS" if ok else "FAIL", detail=detail))

    R = load(args.new, "released")
    if R is not None:
        g = giou(R)
        claim("Released model is close to the paper's validation score (|gIoU - 68.60| <= 1.0; sanity check only, the paper scores all 1,135 validation images)", abs(g - PAPER_VAL_GIOU) <= 1.0, f"{g:.2f}")

        def vs(n):
            X = load(args.new, n)
            if X is None:
                return None
            m = X.merge(R, on="name", suffixes=("", "_r"))
            dd, lo, hi = paired_bootstrap(m.iou, m.iou_r)
            return 100 * dd, 100 * lo, 100 * hi, np.corrcoef(m.iou, m.iou_r)[0, 1]

        z, w = vs("zero"), (vs("wrong_far") or vs("wrong"))   # wrong_far: tokens of a different-category image
        if z:
            claim("Zeroing the 64 visual tokens does not lower accuracy (CI upper bound > 0, |delta| < 1)",
                  z[2] > 0 and abs(z[0]) < 1, f"{z[0]:+.2f} [{z[1]:+.2f}, {z[2]:+.2f}], r={z[3]:.3f}")
        if w:
            claim("A different-category image's tokens leave the masks unchanged (per-image r >= 0.95)", w[3] >= 0.95,
                  f"{w[0]:+.2f}, r={w[3]:.3f}")
        nn = vs("none")
        if nn:
            claim("No visual tokens at all: accuracy within 1 gIoU", abs(nn[0]) < 1, f"{nn[0]:+.2f} [{nn[1]:+.2f}, {nn[2]:+.2f}]")
        for n in ["dproj", "dproj_zero"]:
            x = vs(n)
            if x:
                claim(f"{n}: removing the D-Projector's image input does not lower accuracy (|delta| < 1.5)",
                      abs(x[0]) < 1.5, f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}], r={x[3]:.3f}")
        q = vs("wrongq")
        if q:
            claim("Mismatched question: the question matters (drop > 1.5) but there is no collapse (drop < 15)",
                  -15 < q[0] < -1.5, f"{q[0]:+.2f} [{q[1]:+.2f}, {q[2]:+.2f}], r={q[3]:.3f}")
        trained = [n for n in names if n not in INFERENCE_ONLY and load(args.new, n) is not None]
        if trained:
            ds = {n: vs(n)[0] for n in trained}
            worst = max(ds, key=lambda k: abs(ds[k]))
            claim("Every trained connector at every budget stays within 1.5 gIoU of the released model",
                  all(abs(v) <= 1.5 for v in ds.values()), f"{len(ds)} connectors; largest |delta|: {worst} {ds[worst]:+.2f}")

    ctab = pd.DataFrame(claims)
    lines += ["# Verification against reference results", "",
              f"New results: `{args.new}`  |  Reference: `{args.ref}`", "",
              "## Claims (checked on the new run alone)", "", ctab.to_markdown(index=False) if len(ctab) else "_no claims checked_", "",
              "## Configuration by configuration", "", table.to_markdown(index=False) if len(table) else "_no results_", ""]
    out = os.path.join(args.new, "verification_report.md")
    open(out, "w").write("\n".join(lines))
    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(ctab.to_string(index=False)); print(); print(table.to_string(index=False))
    if not len(table) or not len(ctab):
        print(f"\nwrote {out}  |  FAIL: no results to verify (did the runs crash?)")
        raise SystemExit(1)
    n_fail = int((ctab.status == "FAIL").sum()) + int((table.status == "FAIL").sum())
    print(f"\nwrote {out}  |  failures: {n_fail}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
