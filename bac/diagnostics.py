"""Mechanistic diagnostics behind the token controls. Writes results/diagnostics.json.

1. Token sensitivity: how much does the question state that feeds the mask decoder (pooled hidden state over the
   question tokens) change when the 64 visual tokens are replaced by (a) another, different-category image's tokens,
   (b) zeros, (c) nothing? Reference scale: a different question on a different image.
2. Weight audit: how far did the released connector move from its initialisation, compared with a module that was
   certainly trained (the D-Projector)? Weights are stored in bf16, whose resolution near 1.0 is 2^-7 = 0.0078, so
   "moved" means "differs from the initial value after rounding".

  python bac/diagnostics.py [n_images]
"""
import json
import os
import sys

import numpy as np
import torch

from common import CKPT, DataCollector, EarthReason, ROOT, load_model, model_inputs, to_device
from evaluate import NoTokens, ZeroTokens


class FixedTokens(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x):
        return self.t


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    tok, model, dev = load_model()
    ds = EarthReason(tok, "val", limit=300)
    col = DataCollector(tok)
    released = model.get_model().mm_projector
    cap = {}
    orig_get = model.get_SEG_embedding

    def hook(h, idx, return_all=False):
        out = orig_get(h, idx, return_all)
        cap["q"] = out[0][0].detach().float().clone()          # pooled question state of the (single) sample
        return out

    model.get_SEG_embedding = hook

    def state(b):
        with torch.no_grad():
            model.eval_seg(**model_inputs(b), output_attentions=False)
        return cap["q"]

    rel = lambda a, r: ((a - r).norm() / r.norm()).item()
    rows = []
    for i in np.linspace(0, len(ds) - 1, n).astype(int):
        j = (i + len(ds) // 2) % len(ds)                         # files are sorted by category: j is a different scene type
        b, bo = to_device(col([ds[i]]), dev), to_device(col([ds[j]]), dev)
        with torch.no_grad():
            far = released(model.get_model().get_vision_tower()(bo["images"].float())[-1])
        model.get_model().vision_tower._key = None
        model.get_model().mm_projector = released; s_real = state(b)
        model.get_model().mm_projector = FixedTokens(far); s_far = state(b)
        model.get_model().mm_projector = ZeroTokens(released); s_zero = state(b)
        model.get_model().mm_projector = NoTokens(released); s_none = state(b)
        model.get_model().mm_projector = released
        s_other = state(bo)                                      # different image AND different question
        rows.append(dict(image=ds.images[i].split("/")[-1], other=ds.images[j].split("/")[-1],
                         far_tokens=rel(s_far, s_real), zero_tokens=rel(s_zero, s_real),
                         no_tokens=rel(s_none, s_real), other_question=rel(s_other, s_real)))
    sens = {k: dict(median=float(np.median([r[k] for r in rows])), max=float(np.max([r[k] for r in rows])))
            for k in ["far_tokens", "zero_tokens", "no_tokens", "other_question"]}

    from safetensors import safe_open
    f = safe_open(os.path.join(CKPT, "model.safetensors"), "pt")
    audit = {}
    for k in f.keys():
        if ("mm_projector.norm" in k or ("d_layers" in k and "norm" in k)) and k.endswith("weight"):
            t = f.get_tensor(k).float()
            audit[k] = dict(moved_fraction=float((t != 1).float().mean()), mean_abs_change=float((t - 1).abs().mean()))

    out = dict(n_images=len(rows), sensitivity=sens, per_image=rows, weight_audit=audit)
    json.dump(out, open(os.path.join(ROOT, "results", "diagnostics.json"), "w"), indent=1)
    print(json.dumps(dict(sensitivity=sens, weight_audit=audit), indent=1))


if __name__ == "__main__":
    main()
