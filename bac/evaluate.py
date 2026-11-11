"""Evaluate a connector on EarthReason; writes per-sample CSV + summary JSON.

  python bac/evaluate.py --name released                     # published Compression_Connector
  python bac/evaluate.py --name bac64 --weights runs/bac64/connector.pt
  python bac/evaluate.py --name zero  --kind zero            # control: visual tokens zeroed
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (DataCollector, EarthReason, ROOT, add_lora, load_model, mask_geometry, model_inputs, phi_prefill_flops,
                    predict, predict_tta, saliency_agreement, set_connector, to_device)


class ZeroTokens(torch.nn.Module):
    """Control: keep the 64 image positions but feed the LLM zeros (no visual evidence)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return torch.zeros_like(self.inner(x))


class WrongImageTokens(torch.nn.Module):
    """Control: give the LLM the visual tokens of the *previous* image (content mismatch)."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.prev = inner, None

    def forward(self, x):
        out = self.inner(x)
        prev = torch.zeros_like(out) if self.prev is None else self.prev
        self.prev = out.detach()
        return prev


class ZeroOutput(torch.nn.Module):
    """Wrap a module so its output is replaced by zeros (same shape)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return torch.zeros_like(self.inner(x))


class FixedTokens(torch.nn.Module):
    """Returns tokens set from outside (used for the far-image control)."""

    def __init__(self):
        super().__init__()
        self.tokens = None

    def forward(self, x):
        return self.tokens


class NoTokens(torch.nn.Module):
    """Efficiency test: drop the visual tokens entirely (0 image tokens enter the LLM)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        return out[:, :0]


def load_connector(model, weights):
    blob = torch.load(weights, map_location="cpu", weights_only=False)
    if blob["kind"] == "released":                    # fine-tuned copy of the released connector
        conn = model.get_model().mm_projector.float()
    else:
        conn = set_connector(model, blob["kind"], blob["num_tokens"], **blob.get("kwargs", {}))
    conn.load_state_dict(blob["state_dict"])
    conn.eval()
    if blob.get("mask_path"):                          # fine-tuned mask path
        res = model.load_state_dict({k: v.float() for k, v in blob["mask_path"].items()}, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        model.to(next(conn.parameters()).device)
    if blob.get("lora"):                               # jointly fine-tuned LLM (LoRA)
        add_lora(model, blob["lora_r"])
        missing = model.load_state_dict(blob["lora"], strict=False)
        assert not [k for k in missing.unexpected_keys], missing.unexpected_keys
        model.to(next(conn.parameters()).device)
    return blob


@torch.no_grad()
def answer_nll(model, b):
    """Teacher-forced cross-entropy of the LLM on the reference answer tokens (and on all supervised text tokens)."""
    cap = {}
    orig = model.prepare_inputs_labels_for_multimodal
    def prep(*a, **k):
        out = orig(*a, **k); cap["labels"] = out[4]; return out
    h = model.lm_head.register_forward_hook(lambda m, i, o: cap.__setitem__("logits", o))
    model.prepare_inputs_labels_for_multimodal = prep
    try:
        model(**model_inputs(b), masks=[m.float() for m in b["masks"]], dataset_type=b["dataset_type"], output_attentions=False)
    finally:
        h.remove(); del model.prepare_inputs_labels_for_multimodal
    logits, labels = cap["logits"][0, :-1].float(), cap["labels"][0, 1:]
    ce = torch.nn.functional.cross_entropy(logits, labels.clamp_min(0), reduction="none")
    keep = labels >= 0
    ans = b["token_answer_id"][0].tolist()
    lab = labels.tolist()
    pos = next((j for j in range(len(lab) - len(ans) + 1) if lab[j:j + len(ans)] == ans), None)
    if pos is None:                                          # answer ids not found verbatim: fall back to all text
        sel = keep
    else:
        sel = torch.zeros_like(keep); sel[pos:pos + len(ans)] = True
    top1 = (logits.argmax(-1) == labels)[sel].float().mean().item()
    return dict(answer_nll=ce[sel].mean().item(), answer_top1=top1, n_answer=int(sel.sum()),
                text_nll=ce[keep].mean().item(), found=pos is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default=None, help="connector.pt from train.py; omit for released")
    ap.add_argument("--kind", default=None, help="'zero' control, or an untrained connector kind")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4, 8], help="flip test-time augmentation passes")
    ap.add_argument("--dataset", default="earthreason", choices=["earthreason", "rrsisd"],
                    help="rrsisd: referring segmentation; set SEGEARTH_CKPT to the RRSIS-D checkpoint")
    ap.add_argument("--subset", default="test300", choices=["test300", "calib300"],
                    help="calib300: the next 300 images of the same shuffle (with --limit 300)")
    ap.add_argument("--answer-nll", action="store_true",
                    help="instead of masks, measure how well the LLM predicts the reference answer text (teacher-forced "
                         "cross-entropy on the answer tokens): does the language path use the visual tokens?")
    ap.add_argument("--shuffle-questions", action="store_true",
                    help="control: give each image the question of the image half the list away (files are sorted by "
                         "category, so this is a different target); the ground-truth mask stays the image's own")
    args = ap.parse_args()

    tok, model, dev = load_model()
    meta = {"kind": "released", "num_tokens": 64}
    if args.weights:
        meta = load_connector(model, args.weights)
        meta = {"kind": meta["kind"], "num_tokens": meta["num_tokens"]}
    if args.weights and args.kind not in ("zero", "none"):
        pass
    elif args.kind == "zero":
        model.get_model().mm_projector = ZeroTokens(model.get_model().mm_projector)
        meta = {"kind": "zero", "num_tokens": 64}
    elif args.kind in ("dproj", "dproj_zero"):
        # bypass ablation: the D-Projector cross-attends the description embedding with projected Swin v4
        # features (model.local_project); zeroing them removes its image input. dproj_zero also zeroes the LLM tokens.
        model.local_project = ZeroOutput(model.local_project)
        if args.kind == "dproj_zero":
            model.get_model().mm_projector = ZeroTokens(model.get_model().mm_projector)
        meta = {"kind": args.kind, "num_tokens": 64}
    elif args.kind == "none":
        model.get_model().mm_projector = NoTokens(model.get_model().mm_projector)
        meta = {"kind": "none", "num_tokens": 0}
    elif args.kind == "wrong":
        model.get_model().mm_projector = WrongImageTokens(model.get_model().mm_projector)
        meta = {"kind": "wrong", "num_tokens": 64}
    elif args.kind == "wrong_far":
        # tokens of the image half the subset away: files are sorted by category, so this is a different scene type
        # (the plain "wrong" control uses the previous image, which is usually the same category)
        far_conn = model.get_model().mm_projector
        model.get_model().mm_projector = FixedTokens()
        meta = {"kind": "wrong_far", "num_tokens": 64}
    elif args.kind:
        set_connector(model, args.kind, args.tokens).eval()
        meta = {"kind": args.kind + "-untrained", "num_tokens": args.tokens}
    conn = model.get_model().mm_projector
    conn_params = sum(p.numel() for p in conn.parameters())

    if args.dataset == "rrsisd":
        from common import DATA
        from eval_dataset.RS_val_dataset import RRSISDDataset
        ds = RRSISDDataset(DATA, tok, args.split)
        keep = sorted(np.random.RandomState(0).permutation(len(ds.images))[: args.limit].tolist())
        ds.images, ds.refs, ds.masks = ([x[k] for k in keep] for x in (ds.images, ds.refs, ds.masks))
        if args.shuffle_questions:                       # each image gets the expression of a sample half the subset away
            k = len(ds.refs) // 2
            ds.refs = ds.refs[k:] + ds.refs[:k]
    else:
        ds = EarthReason(tok, args.split, limit=None if args.limit == 0 else args.limit)
        if args.subset == "calib300":                   # the next 300 of the seed-0 shuffle (disjoint from test300)
            full = EarthReason(tok, args.split)
            order = list(range(len(full.images))); random.Random(0).shuffle(order)
            keep = sorted(order[300:600])
            ds.images, ds.labels, ds.QAs = ([x[i] for i in keep] for x in (full.images, full.labels, full.QAs))
        if args.shuffle_questions:
            k = len(ds.QAs) // 2
            ds.QAs = ds.QAs[k:] + ds.QAs[:k]
    dl = DataLoader(ds, batch_size=1, collate_fn=DataCollector(tok), num_workers=2)
    rows = []
    sync = torch.mps.synchronize if dev.type == "mps" else (torch.cuda.synchronize if dev.type == "cuda" else (lambda: None))
    for i, batch in enumerate(dl):
        b = to_device(batch, dev)
        if args.kind == "wrong_far":
            far = to_device(DataCollector(tok)([ds[(i + len(ds) // 2) % len(ds)]]), dev)
            with torch.no_grad():
                conn.tokens = far_conn(model.get_model().get_vision_tower()(far["images"].float())[-1])
            model.get_model().vision_tower._key = None      # the real image's features must be recomputed
        if args.answer_nll:
            rows.append(dict(name=batch["image_name"][0], **answer_nll(model, b)))
            if i % 25 == 0:
                print(f"[{args.name}] {i+1}/{len(ds)}  answer NLL {pd.DataFrame(rows).answer_nll.mean():.4f}", flush=True)
            continue
        sync()
        t0 = time.perf_counter()
        pred = predict(model, b)[0] if args.tta == 1 else predict_tta(model, b, args.tta)[0]
        sync()
        dt = time.perf_counter() - t0
        gt = b["masks"][0, 0]
        inter = (pred & gt.bool()).sum().item()
        union = (pred | gt.bool()).sum().item()
        n_answer = len(b["token_answer_id"][0]) - 1 if "token_answer_id" in b else 0
        seq_len = b["input_ids"].shape[1] - 1 + meta["num_tokens"] + len(b["token_refer_id"][0]) - 1 + n_answer
        row = dict(name=f"{batch['image_name'][0]}#{i}" if args.dataset == "rrsisd" else batch["image_name"][0], inter=inter, union=union,
                   iou=1.0 if union == 0 else inter / union, empty=int(gt.sum().item() == 0),
                   pred_empty=int(pred.sum().item() == 0), latency_s=dt, seq_len=seq_len,
                   agreement=saliency_agreement(getattr(conn, "last_selected", None)[0]
                                                if getattr(conn, "last_selected", None) is not None else None, gt),
                   **mask_geometry(gt))
        rows.append(row)
        if i % 25 == 0:
            df = pd.DataFrame(rows)
            print(f"[{args.name}] {i+1}/{len(ds)}  gIoU {df.iou.mean():.4f}  "
                  f"cIoU {df.inter.sum()/max(df.union.sum(),1):.4f}  {dt:.2f}s/img", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, f"{args.name}_{args.split}.csv"), index=False)
    if args.answer_nll:
        summary = dict(name=args.name, kind=meta["kind"], num_tokens=meta["num_tokens"], n=len(df),
                       answer_nll=float(df.answer_nll.mean()), answer_ppl=float(np.exp(df.answer_nll.mean())),
                       answer_top1=float(df.answer_top1.mean()), text_nll=float(df.text_nll.mean()))
        json.dump(summary, open(os.path.join(args.out, f"{args.name}_{args.split}.json"), "w"), indent=2)
        print(json.dumps(summary, indent=2))
        return
    summary = dict(name=args.name, kind=meta["kind"], tta=args.tta, num_tokens=meta["num_tokens"], split=args.split,
                   n=len(df), gIoU=df.iou.mean(), cIoU=df.inter.sum() / df.union.sum(),
                   **{f"Pr@{t}": float((df.iou > t).mean()) for t in (0.5, 0.6, 0.7, 0.8, 0.9)},
                   latency_ms=1e3 * df.latency_s.iloc[1:].median(),
                   prefill_GFLOPs=phi_prefill_flops(int(df.seq_len.median())) / 1e9,
                   connector_params_M=conn_params / 1e6,
                   peak_mem_GB=(torch.mps.driver_allocated_memory() if dev.type == "mps" else
                                torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0) / 1e9,
                   boundary_agreement=float(np.nanmean(df.agreement)) if df.agreement.notna().any() else None)
    json.dump(summary, open(os.path.join(args.out, f"{args.name}_{args.split}.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
