"""Aggregate results/*.json|csv into summary tables and figures.

  python bac/analyze.py            # writes results/summary.csv, results/stratified.csv, figures/*.png
"""
import glob
import json
import os
import sys

import matplotlib
if "ipykernel" not in sys.modules:   # file-only backend for scripts; inside Jupyter keep the inline plots working
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
COLORS = {"wrong": "#a05195", "bac": "#2a6fdb", "pool": "#e8833a", "uniform": "#5b8c5a", "random": "#9a9a9a",
          "tome": "#a05195", "released": "#222222", "zero": "#c0392b"}


def load(split="val"):
    summ = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, f"*_{split}.json")))]
    per = {s["name"]: pd.read_csv(os.path.join(RES, f"{s['name']}_{split}.csv"), dtype={"name": str}) for s in summ}
    return pd.DataFrame(summ), per


def paired_bootstrap(a, b, n=5000, seed=0):
    """95% CI of mean(a - b) over paired samples (per-image IoU)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


HELD_OUT = ["basketball court", "island", "lake", "stadium"]   # withheld from EarthReason training


def category_of(name, split="val"):
    """Held-out category from the answer text (it names the target), else 'seen'/'empty'."""
    p = os.path.join(ROOT, "data", "rs_reason_seg", "RSReasonSeg", split, "QAs", f"{name}.json")
    try:
        ans = " ".join(json.load(open(p))["answer"]).lower()
    except (OSError, KeyError):
        return "unknown"
    if not ans:
        return "empty"
    hits = [c for c in HELD_OUT if c in ans]
    return "held-out: " + hits[0] if hits else "seen category"


def boundary_chance(names, split="val"):
    """Per-image fraction of 32x32 cells on the dilated GT boundary = agreement of a random selector."""
    cache = os.path.join(RES, f"boundary_chance_{split}.csv")
    have = pd.read_csv(cache, dtype={"name": str}) if os.path.exists(cache) else pd.DataFrame(columns=["name", "chance"])
    todo = [n for n in names if n not in set(have.name)]
    if todo:
        import sys
        import cv2
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from common import boundary_grid, preprocess_mask
        rows = []
        for n in todo:
            m = (cv2.imread(os.path.join(ROOT, "data", "rs_reason_seg", "RSReasonSeg", split, "labels", f"{n}.png"), 0) > 0)
            g = preprocess_mask(m.astype(np.uint8), 1024)[0]
            rows.append(dict(name=n, chance=np.nan if g.sum() == 0 else boundary_grid(g).float().mean().item()))
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(cache, index=False)
    return have


def strata(df):
    df = df.copy()
    df["category"] = [category_of(str(n)) for n in df.name]
    df["familiarity"] = np.where(df.category.str.startswith("held-out"), "held-out (unseen in training)",
                                 np.where(df.category == "seen category", "seen in training", "empty"))
    nz = df[df["empty"] == 0]
    df["size"] = pd.cut(df.area_cells, [-1, 0, 4, 16, 64, 1025],
                        labels=["empty", "tiny (<4 cells)", "small (4-16)", "medium (16-64)", "large (>64)"])
    q = nz.compactness.quantile([1 / 3, 2 / 3]).values if len(nz) else [0.3, 0.6]
    df["shape"] = pd.cut(df.compactness, [-0.01, q[0], q[1], 1.01],
                         labels=["irregular", "moderate", "compact"])
    return df


def main(split="val", reference="released"):
    os.makedirs(FIG, exist_ok=True)
    summ, per = load(split)
    if summ.empty:
        print("no results yet"); return
    cols = ["name", "kind", "num_tokens", "n", "gIoU", "cIoU", "Pr@0.5", "Pr@0.7", "Pr@0.9",
            "latency_ms", "prefill_GFLOPs", "connector_params_M", "boundary_agreement"]
    summ = summ[[c for c in cols if c in summ]].sort_values(["num_tokens", "gIoU"], ascending=[False, False])

    # paired bootstrap vs reference and vs same-budget pooling
    ci = []
    for _, r in summ.iterrows():
        row = {"name": r["name"]}
        for ref in [reference, f"pool{r.num_tokens}"]:
            if ref in per and ref != r["name"]:
                m = per[r["name"]].merge(per[ref], on="name", suffixes=("", "_ref"))
                d, lo, hi = paired_bootstrap(m.iou, m.iou_ref)
                tag = "vs_released" if ref == reference else "vs_pool"
                row[f"dgIoU_{tag}"] = f"{100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]"
        ci.append(row)
    summ = summ.merge(pd.DataFrame(ci), on="name", how="left")
    summ.to_csv(os.path.join(RES, "summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # stratified gIoU
    rows = []
    for name, df in per.items():
        df = strata(df)
        for axis in ["size", "shape", "familiarity", "category"]:
            for lvl, g in df.groupby(axis, observed=True):
                rows.append(dict(name=name, axis=axis, stratum=str(lvl), n=len(g), gIoU=g.iou.mean()))
        e = df[df["empty"] == 1]
        if len(e):
            rows.append(dict(name=name, axis="hallucination", stratum="empty target: predicted empty",
                             n=len(e), gIoU=e.pred_empty.mean()))
    strat = pd.DataFrame(rows)
    strat.to_csv(os.path.join(RES, "stratified.csv"), index=False)
    pivot = strat.pivot_table(index=["axis", "stratum"], columns="name", values="gIoU")
    print("\nStratified gIoU\n", pivot.round(4).to_string())

    # figure 1: accuracy vs budget
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for kind, g in summ[summ.kind.isin(["bac", "pool", "uniform", "random", "tome"])].groupby("kind"):
        base = g[g.name == g.kind + g.num_tokens.astype(str)]   # main runs only, not ablations
        base = base.sort_values("num_tokens")
        if len(base):
            ax.plot(base.num_tokens, 100 * base.gIoU, "o-", color=COLORS.get(kind), label=kind.upper() if kind == "bac" else kind)
    for name, ls in [("released", "--"), ("zero", ":")]:
        if name in set(summ.name):
            ax.axhline(100 * summ[summ.name == name].gIoU.iloc[0], ls=ls, color=COLORS[name],
                       label="released (strided conv, 64)" if name == "released" else "zero visual tokens")
    if "released" in set(summ.name):   # honest scale: shade +-1 gIoU around the released model
        ref = 100 * summ[summ.name == "released"].gIoU.iloc[0]
        ax.axhspan(ref - 1, ref + 1, color="#888888", alpha=.12, lw=0, label="released ±1 gIoU")
        ax.set_ylim(ref - 6, ref + 6)
    ax.set_xscale("log", base=2); ax.set_xticks([16, 32, 64, 128]); ax.set_xticklabels([16, 32, 64, 128])
    ax.set_xlabel("visual tokens passed to the LLM"); ax.set_ylabel("gIoU (%)")
    ax.set_title(f"EarthReason {split}: accuracy vs token budget"); ax.grid(alpha=.3)
    ax.legend(fontsize=7.5, ncol=4, loc="lower center", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "budget_curve.png"), dpi=200); plt.close(fig)

    # figure 2: stratified bars by target size
    sz = strat[strat.axis == "size"].pivot_table(index="stratum", columns="name", values="gIoU")
    keep = [c for c in ["released", "zero", "pool64", "bac64", "pool16", "bac16"] if c in sz]
    if keep:
        order = [s for s in ["tiny (<4 cells)", "small (4-16)", "medium (16-64)", "large (>64)"] if s in sz.index]
        sz.loc[order, keep].mul(100).plot.bar(figsize=(7.5, 4), rot=0, width=.8)
        plt.ylabel("gIoU (%)"); plt.title("gIoU by target size (32x32 feature-grid cells)")
        plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(FIG, "by_size.png"), dpi=200); plt.close()

    # figure 3 + stat: saliency-boundary agreement vs IoU for BAC runs
    for name, df in per.items():
        if df.agreement.notna().sum() > 10:
            d = df.dropna(subset=["agreement"])
            rho, p = spearmanr(d.agreement, d.iou)
            ch = d.merge(boundary_chance(list(d.name)), on="name")
            print(f"\n{name}: saliency-boundary agreement mean {d.agreement.mean():.3f} vs chance "
                  f"{ch.chance.mean():.3f} (lift x{d.agreement.mean()/ch.chance.mean():.2f}, above chance on "
                  f"{(ch.agreement > ch.chance).mean():.0%} of images); "
                  f"Spearman(agreement, IoU) = {rho:.3f} (p = {p:.3g}, n = {len(d)})")
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            ax.scatter(d.agreement, 100 * d.iou, s=10, alpha=.5, color=COLORS["bac"])
            ax.set_xlabel("fraction of boundary tokens on GT boundary"); ax.set_ylabel("IoU (%)")
            ax.set_title(f"{name}: Spearman rho = {rho:.2f} (p = {p:.2g})")
            ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(FIG, f"agreement_{name}.png"), dpi=200)
            plt.close(fig)

    # figure 4: do the LLM's visual tokens matter at all? per-image IoU, released vs controls
    ctrl = [c for c in ["zero", "wrong"] if c in per and "released" in per]
    if ctrl:
        fig, axes = plt.subplots(1, len(ctrl), figsize=(4.4 * len(ctrl), 4.2), squeeze=False)
        for ax, c in zip(axes[0], ctrl):
            m = per["released"].merge(per[c], on="name", suffixes=("", "_c"))
            d, lo, hi = paired_bootstrap(m.iou_c, m.iou)
            r = np.corrcoef(m.iou, m.iou_c)[0, 1]
            ax.scatter(100 * m.iou, 100 * m.iou_c, s=12, alpha=.55, color=COLORS[c] if c in COLORS else "#a05195")
            ax.plot([0, 100], [0, 100], color="#888", lw=1, ls="--")
            ax.set_xlabel("IoU with real visual tokens (%)")
            ax.set_ylabel({"zero": "IoU with zeroed visual tokens (%)", "wrong": "IoU with another image's tokens (%)"}[c])
            ax.set_title(f"r = {r:.3f};  dgIoU = {100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]", fontsize=9)
            ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "token_ablation.png"), dpi=200); plt.close(fig)

    # figure 5: training curves
    logs = {os.path.basename(os.path.dirname(p)): pd.DataFrame(json.load(open(p)))
            for p in glob.glob(os.path.join(ROOT, "runs", "*", "log.json"))}
    if logs:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for name, lg in sorted(logs.items()):
            ax.plot(lg.step, (lg["mask"] + lg["dice"]).rolling(20, min_periods=1).mean(), label=name)
        ax.set_xlabel("step"); ax.set_ylabel("mask + dice loss (smoothed)"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2); ax.set_title("Connector-only training")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "training_curves.png"), dpi=200); plt.close(fig)
    print(f"\nwrote {RES}/summary.csv, {RES}/stratified.csv and figures in {FIG}")


if __name__ == "__main__":
    main()
