"""Consolidate the token-bottleneck testbed (Mac + Kaggle), the RRSIS-D token controls and the cross-hardware
replication into tables and figures.

  .venv/bin/python bac/token_bottleneck/analyze_testbed.py

Inputs : results/bottleneck/*.json|csv (Apple M4), results/kaggle/bottleneck/*.json|csv (Kaggle T4),
         results/rrsisd/*, results/kaggle/{diagnostics.json, repro_verification.md}, results/diagnostics.json
Outputs: results/testbed_summary.csv, results/testbed_comparisons.csv, results/testbed_replication.csv,
         results/rrsisd_summary.csv, results/final_numbers.json, figures/testbed_*.png
Rule   : one run per (config, seed); the Mac run is used when both exist (Kaggle duplicates of the Mac's seed-0
         16-token runs serve only as the cross-hardware replication check). Differences are paired bootstrap
         95% CIs over the 300 validation images of the per-image IoU averaged over the available seeds.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac"))
from analyze import paired_bootstrap  # noqa: E402

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
SRC = {"mac": os.path.join(RES, "bottleneck"), "kaggle": os.path.join(RES, "kaggle", "bottleneck")}
LABEL = {"uniform": "Strided conv", "pool": "Average pool", "bac": "BAC (proposed)", "bac2": "BAC-v2",
         "bac2_randsel": "BAC-v2, random selection", "bacq": "BAC-q (question-aware)", "random": "Random select",
         "tome": "ToMe", "pool1024": "No compression (1024)", "bac_learned": "BAC, learned score only",
         "bac2_orb": "Oracle: GT boundary", "bac2_orr": "Oracle: GT region", "bacq_supb": "BAC-q, boundary-supervised",
         "bacq_supr": "BAC-q, region-supervised", "bac2_r75": "BAC-v2, 75% boundary"}
COL = {"uniform": "#5b8c5a", "pool": "#e8833a", "bac": "#9db7e6", "bac2": "#2a6fdb", "bac2_randsel": "#7f8c8d",
       "bacq": "#8e44ad", "random": "#b0b0b0", "tome": "#a05195", "bac2_orb": "#c0392b", "bac2_orr": "#e74c3c",
       "bacq_supb": "#16a085", "bacq_supr": "#1abc9c", "bac2_r75": "#5dade2"}


def parse(name):
    """tb_bac2_16_randsel_s1 -> ('bac2_randsel', 16, 1); tb_val_pool16_s0 -> ('pool', 16, 0); tb_bacq64_supr_s2 -> ('bacq_supr', 64, 2)."""
    m = re.match(r"tb_(?:val_)?(bac2|bacq|uniform|pool|bac|random|tome)_?(\d+)(?:_([a-z0-9]+))?_s(\d)$", name)
    kind, tokens, suffix, seed = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    if kind == "pool" and tokens == 1024:
        kind = "pool1024"
    return kind + (f"_{suffix}" if suffix else ""), tokens, seed


def collect():
    runs = {}
    for hw in ["kaggle", "mac"]:                       # mac last: it wins when both exist
        for p in glob.glob(os.path.join(SRC[hw], "tb_*_s*.json")):
            j = json.load(open(p))
            cfg, tokens, seed = parse(j["name"])
            df = pd.read_csv(p[:-5] + ".csv", dtype={"name": str})
            runs[(cfg, tokens, seed)] = dict(hw=hw, gIoU=j["gIoU"], cIoU=j["cIoU"], minutes=j["minutes"],
                                             iou=df.set_index("name").iou)
    return runs


def per_image(runs, cfg, tokens):
    s = {seed: r["iou"] for (c, t, seed), r in runs.items() if c == cfg and t == tokens}
    return (pd.concat(s, axis=1).mean(axis=1), sorted(s)) if s else (None, [])


def main():
    runs = collect()
    # summary per configuration
    rows = []
    for (cfg, tokens) in sorted({(c, t) for c, t, _ in runs}, key=lambda x: (x[1], x[0])):
        rs = {s: r for (c, t, s), r in runs.items() if c == cfg and t == tokens}
        g = np.array([100 * rs[s]["gIoU"] for s in sorted(rs)])
        c = np.array([100 * rs[s]["cIoU"] for s in sorted(rs)])
        rows.append(dict(config=cfg, label=LABEL.get(cfg, cfg), tokens=tokens, n_seeds=len(rs), seeds=",".join(map(str, sorted(rs))),
                         hardware="+".join(sorted({rs[s]["hw"] for s in rs})), gIoU=g.mean(), gIoU_std=g.std(ddof=1) if len(g) > 1 else np.nan,
                         cIoU=c.mean(), per_seed=" / ".join(f"{v:.2f}" for v in g),
                         minutes=np.mean([rs[s]["minutes"] for s in rs])))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(RES, "testbed_summary.csv"), index=False)

    # pre-declared paired comparisons
    plan = []
    for t in (16, 64):
        plan += [("bac2", "bac2_randsel", t, "saliency vs random selection (core test of the boundary idea)"),
                 ("bac2", "uniform", t, "BAC-v2 vs the published strided conv"),
                 ("bacq", "uniform", t, "question-aware BAC vs strided conv"),
                 ("bacq", "bac2", t, "does question conditioning help?"),
                 ("bacq", "bac2_randsel", t, "question-aware saliency vs random selection"),
                 ("bac2", "pool", t, "BAC-v2 vs average pooling"),
                 ("bac", "pool", t, "proposed BAC vs average pooling"),
                 ("uniform", "pool", t, "learned vs fixed downsampling"),
                 ("random", "pool", t, "random selection vs pooling"),
                 ("tome", "pool", t, "ToMe vs pooling")]
        plan += [("bac2_orb", "uniform", t, "ceiling: perfect boundary allocation vs strided conv"),
                 ("bac2_orr", "uniform", t, "ceiling: perfect region allocation vs strided conv"),
                 ("bac2_orr", "bac2_orb", t, "region vs boundary allocation (oracle)"),
                 ("bac2_orb", "bac2_randsel", t, "perfect vs random allocation, same network"),
                 ("bacq_supb", "uniform", t, "boundary-supervised BAC-q vs strided conv"),
                 ("bacq_supr", "uniform", t, "region-supervised BAC-q vs strided conv"),
                 ("bacq_supb", "bacq", t, "does supervising the saliency help? (boundary)"),
                 ("bacq_supr", "bacq", t, "does supervising the saliency help? (region)"),
                 ("bac2_r75", "bac2", t, "75% vs 50% of the budget on boundary tokens")]
    for t in (32, 128):
        plan += [("bacq", "uniform", t, "budget curve"), ("bac2", "uniform", t, "budget curve"), ("pool", "uniform", t, "budget curve")]
    comps = []
    for a, b, t, why in plan:
        A, sa = per_image(runs, a, t)
        B, sb = per_image(runs, b, t)
        if A is None or B is None:
            continue
        m = pd.concat([A.rename("a"), B.rename("b")], axis=1).dropna()
        d, lo, hi = paired_bootstrap(m.a, m.b)
        hw = {runs[(x, t, s)]["hw"] for x, ss in [(a, sa), (b, sb)] for s in ss}
        comps.append(dict(tokens=t, a=a, b=b, question=why, seeds_a=len(sa), seeds_b=len(sb), delta=100 * d, lo=100 * lo, hi=100 * hi,
                          significant=bool(lo > 0 or hi < 0), cross_hardware=len(hw) > 1))
    # upper bound: no compression vs strided conv at 64
    A, sa = per_image(runs, "pool1024", 1024)
    B, sb = per_image(runs, "uniform", 64)
    m = pd.concat([A.rename("a"), B.rename("b")], axis=1).dropna()
    d, lo, hi = paired_bootstrap(m.a, m.b)
    comps.append(dict(tokens=1024, a="pool1024", b="uniform", question="headroom: no compression vs 64 strided-conv tokens",
                      seeds_a=len(sa), seeds_b=len(sb), delta=100 * d, lo=100 * lo, hi=100 * hi, significant=bool(lo > 0 or hi < 0),
                      cross_hardware=True))
    comps = pd.DataFrame(comps)
    comps.to_csv(os.path.join(RES, "testbed_comparisons.csv"), index=False)

    # cross-hardware replication (same config + seed)
    rep = []
    for p in glob.glob(os.path.join(SRC["kaggle"], "tb_*_s*.json")):
        j = json.load(open(p))
        cfg, t, s = parse(j["name"])
        mac = [q for q in glob.glob(os.path.join(SRC["mac"], "tb_*_s*.json")) if parse(json.load(open(q))["name"]) == (cfg, t, s)]
        if mac:
            jm = json.load(open(mac[0]))
            rep.append(dict(config=cfg, tokens=t, seed=s, mac=100 * jm["gIoU"], kaggle=100 * j["gIoU"], diff=100 * (j["gIoU"] - jm["gIoU"])))
    rep = pd.DataFrame(rep).sort_values(["tokens", "config"])
    rep.to_csv(os.path.join(RES, "testbed_replication.csv"), index=False)

    # RRSIS-D token controls
    D = os.path.join(RES, "rrsisd")
    R = pd.read_csv(os.path.join(D, "rrsisd_released_val.csv"), dtype={"name": str})
    rr = [dict(control="released", gIoU=100 * R.iou.mean(), cIoU=100 * R.inter.sum() / R.union.sum(), delta=0.0, lo=np.nan, hi=np.nan, r=1.0)]
    for n in ["zero", "wrong_far", "none", "wrongq"]:
        X = pd.read_csv(os.path.join(D, f"rrsisd_{n}_val.csv"), dtype={"name": str}).merge(R, on="name", suffixes=("", "_r"))
        d, lo, hi = paired_bootstrap(X.iou, X.iou_r)
        rr.append(dict(control=n, gIoU=100 * X.iou.mean(), cIoU=100 * X.inter.sum() / X.union.sum(), delta=100 * d, lo=100 * lo, hi=100 * hi,
                       r=float(np.corrcoef(X.iou, X.iou_r)[0, 1])))
    rr = pd.DataFrame(rr)
    rr.to_csv(os.path.join(RES, "rrsisd_summary.csv"), index=False)

    # one JSON with every headline number
    def cmp(a, b, t):
        r = comps[(comps.a == a) & (comps.b == b) & (comps.tokens == t)].iloc[0]
        return dict(delta=round(r.delta, 2), lo=round(r.lo, 2), hi=round(r.hi, 2), sig=bool(r.significant))

    def g(cfg, t):
        r = summ[(summ.config == cfg) & (summ.tokens == t)]
        return None if not len(r) else dict(gIoU=round(r.gIoU.iloc[0], 2), std=None if np.isnan(r.gIoU_std.iloc[0]) else round(r.gIoU_std.iloc[0], 2),
                                             n=int(r.n_seeds.iloc[0]), per_seed=r.per_seed.iloc[0])
    diag = {k: json.load(open(p)) for k, p in [("mac", os.path.join(RES, "diagnostics.json")), ("kaggle", os.path.join(RES, "kaggle", "diagnostics.json"))] if os.path.exists(p)}
    rv = open(os.path.join(RES, "kaggle", "repro_verification.md")).read() if os.path.exists(os.path.join(RES, "kaggle", "repro_verification.md")) else ""
    out = dict(
        testbed={f"{c}{t}": g(c, t) for c, t in {(c, t) for c, t, _ in runs}},
        comparisons={f"{a}-{b}@{t}": cmp(a, b, t) for a, b, t in comps[["a", "b", "tokens"]].itertuples(index=False)},
        replication=dict(n=len(rep), max_abs=round(rep["diff"].abs().max(), 2), mean_abs=round(rep["diff"].abs().mean(), 2)),
        rrsisd={r.control: dict(gIoU=round(r.gIoU, 2), cIoU=round(r.cIoU, 2), delta=round(r.delta, 2),
                                lo=None if np.isnan(r.lo) else round(r.lo, 2), hi=None if np.isnan(r.hi) else round(r.hi, 2), r=round(r.r, 3))
                for r in rr.itertuples()},
        diagnostics={hw: {k: dict(median=round(100 * v["median"], 2), max=round(100 * v["max"], 2)) for k, v in d["sensitivity"].items()}
                     for hw, d in diag.items()},
        weight_audit={k.replace("model.", ""): round(100 * v["moved_fraction"], 1) for k, v in diag.get("kaggle", diag.get("mac", {})).get("weight_audit", {}).items()},
        kaggle_repro=dict(claims_pass=len(re.findall(r"\|\s*PASS\s*\|", rv.split("## Configuration")[0])),
                          claims_total=len(re.findall(r"\|\s*(?:PASS|FAIL)\s*\|", rv.split("## Configuration")[0])),
                          configs_pass=len(re.findall(r"\|\s*PASS\s*\|", rv.split("## Configuration")[-1])),
                          configs_total=len(re.findall(r"\|\s*(?:PASS|FAIL)\s*\|", rv.split("## Configuration")[-1]))),
    )
    json.dump(out, open(os.path.join(RES, "final_numbers.json"), "w"), indent=1)

    # figures
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # (1) 16 and 64 tokens, mean +- std over seeds, per-seed dots
    order = ["uniform", "bacq", "bac2", "bac2_randsel", "pool", "bac", "random", "tome"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    up = summ[summ.config == "pool1024"].gIoU.iloc[0]
    for ax, t in zip(axes, (16, 64)):
        sub = summ[summ.tokens == t].set_index("config")
        cs = [c for c in order if c in sub.index]
        y = np.arange(len(cs))[::-1]
        ax.barh(y, [sub.loc[c, "gIoU"] for c in cs], xerr=[0 if np.isnan(sub.loc[c, "gIoU_std"]) else sub.loc[c, "gIoU_std"] for c in cs],
                color=[COL[c] for c in cs], height=0.68, capsize=2)
        for yy, c in zip(y, cs):
            vals = [float(v) for v in sub.loc[c, "per_seed"].split(" / ")]
            ax.scatter(vals, [yy] * len(vals), s=9, color="k", zorder=3)
            ax.text(sub.loc[c, "gIoU"] + 1.2, yy, f"{sub.loc[c, 'gIoU']:.1f}", va="center", fontsize=9)
        ax.axvline(up, color="k", ls=":", lw=1)
        ax.text(up - 0.4, y[0] + 0.55, "no compression", ha="right", fontsize=8)
        ax.set_yticks(y, [LABEL[c] for c in cs])
        ax.set_xlim(20, 58)
        ax.set_title(f"{t} tokens (3 seeds)")
        ax.set_xlabel("testbed gIoU (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "testbed_bars.png"), dpi=200)
    plt.close()
    # (2) budget curve
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for c in ["uniform", "bacq", "bac2", "pool"]:
        sub = summ[summ.config == c].sort_values("tokens")
        ax.plot(sub.tokens, sub.gIoU, "o-", color=COL[c], label=LABEL[c], lw=2)
    ax.axhline(up, color="k", ls=":", lw=1)
    ax.text(16, up + 0.4, "no compression (1024 tokens)", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128], ["16", "32", "64", "128"])
    ax.set_xlabel("tokens M (16 and 64: mean of 3 seeds; 32, 128: seed 0)")
    ax.set_ylabel("testbed gIoU (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "testbed_budget.png"), dpi=200)
    plt.close()

    with pd.option_context("display.width", 200):
        print(summ[["config", "tokens", "n_seeds", "hardware", "gIoU", "gIoU_std", "per_seed"]].round(2).to_string(index=False))
        print(comps[["tokens", "a", "b", "delta", "lo", "hi", "significant", "cross_hardware"]].round(2).to_string(index=False))
        print(rep.round(2).to_string(index=False))
        print(rr.round(3).to_string(index=False))
    print(json.dumps({k: out[k] for k in ["replication", "diagnostics", "weight_audit", "kaggle_repro"]}, indent=0))


if __name__ == "__main__":
    main()
