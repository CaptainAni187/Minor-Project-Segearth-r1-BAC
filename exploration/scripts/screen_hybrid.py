"""Hybrid SegEarth-R1 + open-vocabulary branch: choose candidate selection and routing on calib300, report test300.

Inputs: dumped SegEarth-R1 logits (dump_logits.py) and OV candidates (ov_branch.py). Nothing here sees the
ground truth except to score policies on calib300 (selection) and test300 (one report per chosen policy).

Selection of one OV candidate per image:
  owl      : highest OWLv2 score          sam   : highest SAM predicted IoU
  owlxsam  : highest product              agree : best agreement with SegEarth-R1's mask (owl if it is empty)
Routing (use OV instead of SegEarth-R1):
  refusal  : SegEarth-R1 predicted nothing
  conf<tau : SegEarth-R1's maximum logit below tau (refusals included)
  router   : logistic regression on [max logit, log area, refusal, owl, sam, agreement] fitted on calib300 with
             5-fold CV for the decision threshold

  python exploration/scripts/screen_hybrid.py Qwen2.5-3B-Instruct free base
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bac"))
from analyze import paired_bootstrap  # noqa: E402

R = os.path.join(ROOT, "exploration", "results")


def iou(a, b):
    inter = (a & b).sum((-2, -1)); union = (a | b).sum((-2, -1))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def load(s, nouns, mode, sam):
    z = np.load(os.path.join(R, "logits", f"{s}.npz"))
    o = np.load(os.path.join(R, "ov", f"{s}_{nouns}_{mode}_{sam}.npz"))
    meta = json.load(open(os.path.join(R, "logits", f"{s}_meta.json")))
    lg = z["plain"].astype(np.float32)
    seg = lg > 0
    c = np.unpackbits(o["cands"], axis=-1)[..., :512].astype(bool)
    d = dict(seg=seg, gt=z["gt"], cands=c, owl=o["owl"], sam=o["sam"], nouns=o["nouns"], meta=meta,
             maxlog=lg.reshape(len(lg), -1).max(1), refusal=~seg.any((1, 2)))
    d["agree"] = iou(c, seg[:, None])
    return d


def select(d, how):
    n = len(d["seg"])
    if how == "owl":
        k = np.zeros(n, int)
    elif how == "sam":
        k = d["sam"].argmax(1)
    elif how == "owlxsam":
        k = (d["owl"] * d["sam"]).argmax(1)
    else:  # agree
        k = np.where(d["refusal"], 0, d["agree"].argmax(1))
    return k


def features(d, k):
    n = np.arange(len(k))
    area = d["seg"].sum((1, 2))
    return np.stack([d["maxlog"], np.log1p(area), d["refusal"].astype(float), d["owl"][n, k], d["sam"][n, k],
                     d["agree"][n, k]], 1)


def fit_router(X, y_gain, seed=0):
    """Logistic regression (numpy) predicting 'OV beats SegEarth-R1'; threshold by 5-fold CV on the gain."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    y = (y_gain > 0).astype(float)

    def train(Zt, yt, l2=1.0, it=2000, lr=0.1):
        w = np.zeros(Zt.shape[1] + 1)
        A = np.hstack([Zt, np.ones((len(Zt), 1))])
        for _ in range(it):
            p = 1 / (1 + np.exp(-A @ w))
            w -= lr * (A.T @ (p - yt) / len(yt) + l2 * np.r_[w[:-1], 0] / len(yt))
        return w

    def prob(w, Zt):
        return 1 / (1 + np.exp(-np.hstack([Zt, np.ones((len(Zt), 1))]) @ w))

    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(Z)) % 5
    best = None
    for l2 in [0.1, 1.0, 10.0]:
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            g = 0.0
            for f in range(5):
                w = train(Z[folds != f], y[folds != f], l2)
                g += (y_gain[folds == f] * (prob(w, Z[folds == f]) > thr)).sum()
            if best is None or g > best[0]:
                best = (g, l2, thr)
    w = train(Z, y, best[1])
    return lambda Xn: prob(w, (Xn - mu) / sd) > best[2], dict(l2=best[1], thr=best[2], cv_gain=float(best[0]))


def main():
    nouns, mode, sam = (sys.argv[1:4] + ["Qwen2.5-3B-Instruct", "free", "base"][len(sys.argv[1:4]):])
    cal, tst = load("calib300", nouns, mode, sam), load("test300", nouns, mode, sam)
    base_c, base_t = iou(cal["seg"], cal["gt"]), iou(tst["seg"], tst["gt"])
    results = []
    for how in ["owl", "sam", "owlxsam", "agree"]:
        kc, kt = select(cal, how), select(tst, how)
        ovc = iou(cal["cands"][np.arange(300), kc], cal["gt"]); ovt = iou(tst["cands"][np.arange(300), kt], tst["gt"])
        rules = {"always_ov": (np.ones(300, bool), np.ones(300, bool)), "refusal": (cal["refusal"], tst["refusal"])}
        for tau in [0, 2, 4, 6, 8, 10, 12, 15]:
            rules[f"conf<{tau}"] = (cal["maxlog"] < tau, tst["maxlog"] < tau)
        router, info = fit_router(features(cal, kc), ovc - base_c)
        rules["router"] = (router(features(cal, kc)), router(features(tst, kt)))
        for r, (uc, ut) in rules.items():
            gc = np.where(uc, ovc, base_c).mean() - base_c.mean()
            results.append(dict(select=how, route=r, calib_gain=100 * gc, routed_calib=int(uc.sum()),
                                _t=np.where(ut, ovt, base_t), routed_test=int(ut.sum()), router=info if r == "router" else None))
    results.sort(key=lambda x: -x["calib_gain"])
    print(f"calib300 baseline {100 * base_c.mean():.2f} | test300 baseline {100 * base_t.mean():.2f} (512 screen)")
    print("top policies by calib gain:")
    for x in results[:10]:
        print(f"  {x['select']:8s} {x['route']:10s} calib {x['calib_gain']:+.2f} (routed {x['routed_calib']})")
    best = results[0]
    cats = np.array(["held-out" if m["category"].startswith("held-out") else ("empty" if m["category"] == "empty" else "seen")
                     for m in tst["meta"]])
    out = []
    for x in [best] + [y for y in results if y["route"] == "refusal" and y["select"] == best["select"]]:
        t = x["_t"]
        dlt, lo, hi = paired_bootstrap(t, base_t)
        rec = dict(select=x["select"], route=x["route"], chosen=x is best, calib_gain=round(x["calib_gain"], 2),
                   test_gIoU=round(100 * t.mean(), 2), test_delta=round(100 * dlt, 2), ci=[round(100 * lo, 2), round(100 * hi, 2)],
                   routed_test=x["routed_test"], router=x["router"],
                   by_category={c: [int((cats == c).sum()), round(100 * base_t[cats == c].mean(), 1), round(100 * t[cats == c].mean(), 1)]
                                for c in ["seen", "held-out", "empty"]})
        out.append(rec)
        print(rec)
    json.dump(dict(nouns=nouns, mode=mode, sam=sam, test=out,
                   calib_ranking=[{k: v for k, v in x.items() if k != "_t"} for x in results[:25]]),
              open(os.path.join(R, f"screen_hybrid_{nouns}_{mode}_{sam}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
