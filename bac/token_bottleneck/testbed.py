"""Token-bottleneck testbed: the M compressed tokens are the ONLY route from the image to the mask.

SegEarth-R1 lets image evidence reach the mask through an uncompressed path, so its visual tokens can be ignored
(see the token controls). Here a compressor (BAC, strided conv, pool, ToMe, random; or all 1024 positions as an
upper bound) turns SegEarth-R1's own frozen Swin-B v4 features into M tokens, and a small question-conditioned
decoder must predict the target mask from those tokens alone. Compressor and decoder are trained together;
everything else (features, question embeddings, decoder, schedule, data) is identical across runs.
Decoder: ScatterTestbed (default; every grid cell takes its nearest token, then a FiLM-conditioned CNN). The first
design, Testbed (cross-attention from a 64x64 query grid), failed its validation check (all 1024 tokens scored below
16 pooled tokens) and is kept only as --decoder attn for the record.

  python bac/token_bottleneck/testbed.py --name tb_bac64 --kind bac --tokens 64 [--seed 0] [--alpha 1]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import DATA, ROOT, pick_device  # noqa: E402
from segearth_r1.model.multimodal_projector.token_compressors import build_connector, sincos_pos_embed_2d  # noqa: E402

FEAT = os.path.join(DATA, "bottleneck")
D = 256          # token / decoder width
OUT_RES = 64     # decoder output grid (upsampled to the 256x256 ground truth)


def pe(rows, cols, dim=D):
    return sincos_pos_embed_2d(dim, rows.flatten(), cols.flatten(), rows.device, torch.float32).view(*rows.shape, dim)


class Testbed(nn.Module):
    def __init__(self, kind, tokens, bac_kw):
        super().__init__()
        self.comp = build_connector(kind, 1024, D, num_tokens=tokens, **bac_kw)
        self.tok_pos = nn.Linear(D, D)
        self.q_proj = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, D))
        r = (torch.arange(OUT_RES).float() + 0.5) * 32 / OUT_RES - 0.5
        rr, cc = torch.meshgrid(r, r, indexing="ij")
        self.register_buffer("query_pe", pe(rr, cc))                       # [64, 64, D]
        self.query_in = nn.Linear(D, D)
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            attn=nn.MultiheadAttention(D, 8, batch_first=True), n1=nn.LayerNorm(D),
            ff=nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)), n2=nn.LayerNorm(D))) for _ in range(2)])
        self.refine = nn.Sequential(nn.Conv2d(D, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(64, 1, 1))

    def forward(self, v4, q, target=None):
        B = v4.shape[0]
        tok = self.comp(v4)                                              # [B, M, D]
        pos = self.comp.last_positions                                   # [B, M, 2]
        qv = self.q_proj(q)                                              # [B, D]
        tok = tok + self.tok_pos(pe(pos[..., 0], pos[..., 1])) + qv[:, None]
        x = self.query_in(self.query_pe).reshape(1, OUT_RES * OUT_RES, D).expand(B, -1, -1) + qv[:, None]
        for L in self.layers:
            x = L["n1"](x + L["attn"](x, tok, tok, need_weights=False)[0])
            x = L["n2"](x + L["ff"](x))
        x = x.transpose(1, 2).reshape(B, D, OUT_RES, OUT_RES)
        return self.refine(x)[:, 0]                                      # [B, 64, 64] logits


class FiLMBlock(nn.Module):
    def __init__(self, c, qdim=D):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)
        self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
        self.film = nn.Linear(qdim, 2 * c)

    def forward(self, x, qv):
        g, b = self.film(qv)[:, :, None, None].chunk(2, 1)
        h = F.gelu(self.n1(self.conv1(x)) * (1 + g) + b)
        return F.gelu(x + self.n2(self.conv2(h)))


class ScatterTestbed(nn.Module):
    """Each of the 32x32 grid cells takes the token whose position is nearest (Voronoi unpooling); a question-
    conditioned CNN decodes the resulting map. With all 1024 tokens this reproduces the full feature map."""

    def __init__(self, kind, tokens, bac_kw):
        super().__init__()
        self.comp = build_connector(kind, 1024, D, num_tokens=tokens, **bac_kw)
        self.q_proj = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, D))
        r = torch.arange(32).float()
        self.register_buffer("cells", torch.stack(torch.meshgrid(r, r, indexing="ij"), -1).reshape(1024, 2))
        self.blocks32 = nn.ModuleList([FiLMBlock(D) for _ in range(3)])
        self.up1, self.b64 = nn.Conv2d(D, 128, 3, padding=1), FiLMBlock(128)
        self.up2, self.b128 = nn.Conv2d(128, 64, 3, padding=1), FiLMBlock(64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, v4, q, target=None):
        B = v4.shape[0]
        if hasattr(self.comp, "set_query"):
            self.comp.set_query(q)
        if hasattr(self.comp, "set_target"):
            self.comp.set_target(target)          # oracle selection / saliency supervision only (see BAC2Connector)
        tok = self.comp(v4)                                              # [B, M, D]
        pos = self.comp.last_positions                                   # [B, M, 2]
        nearest = torch.cdist(self.cells[None].expand(B, -1, -1), pos.float()).argmin(-1)   # [B, 1024]
        grid = torch.gather(tok, 1, nearest[..., None].expand(-1, -1, D))                   # [B, 1024, D]
        x = grid.transpose(1, 2).reshape(B, D, 32, 32)
        qv = self.q_proj(q)
        for blk in self.blocks32:
            x = blk(x, qv)
        x = self.b64(F.gelu(self.up1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))), qv)
        x = self.b128(F.gelu(self.up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))), qv)
        return self.head(x)[:, 0]                                        # [B, 128, 128] logits


def cells(m):
    """256x256 masks -> fraction of each 32x32 feature-grid cell covered by the target."""
    return F.adaptive_avg_pool2d(m[:, None].float(), 32)[:, 0]


def dice_loss(logits, target):
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()


def load(split, feat=None):
    feat = feat or FEAT
    v4 = np.load(os.path.join(feat, f"{split}_v4.npy"), mmap_mode="r")
    masks = np.load(os.path.join(feat, f"{split}_mask.npy"))
    qd = torch.load(os.path.join(feat, f"{split}_q.pt"))
    return v4, masks, qd["q"], qd["names"]


def batch(v4, masks, qs, names, idx, dev, train, gen):
    x = torch.from_numpy(np.stack([v4[i] for i in idx])).float()
    m = torch.from_numpy(masks[idx]).float()
    q = torch.stack([qs[i][torch.randint(len(qs[i]), (1,), generator=gen).item()] if train else qs[i][0] for i in idx]).float()
    if train:
        flip = torch.rand(len(idx), generator=gen) < 0.5                 # same augmentation for every compressor
        x[flip] = x[flip].flip(-1)
        m[flip] = m[flip].flip(-1)
    return x.to(dev), m.to(dev), q.to(dev)


@torch.no_grad()
def evaluate(model, data, dev):
    v4, masks, qs, names = data
    model.eval()
    rows = []
    for s in range(0, len(names), 16):
        idx = list(range(s, min(s + 16, len(names))))
        x, m, q = batch(v4, masks, qs, names, idx, dev, False, None)
        pred = F.interpolate(model(x, q, cells(m))[:, None], size=256, mode="bilinear", align_corners=False)[:, 0] > 0
        for k, i in enumerate(idx):
            p, g = pred[k].cpu(), m[k].cpu().bool()
            inter, union = (p & g).sum().item(), (p | g).sum().item()
            rows.append(dict(name=names[i], inter=inter, union=union, iou=1.0 if union == 0 else inter / union,
                             area_cells=g.sum().item() * 16 / 1024, empty_target=int(g.sum().item() == 0)))
    model.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", required=True, choices=["uniform", "pool", "random", "tome", "bac", "bac2", "bacq"])
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--sal-weight", type=float, default=1.0)
    ap.add_argument("--decoder", default="scatter", choices=["scatter", "attn"])
    ap.add_argument("--select", default="saliency", choices=["saliency", "random", "oracle_boundary", "oracle_region"],
                    help="bac2 / bacq; oracle_*: boundary tokens placed with the ground-truth target (upper bound)")
    ap.add_argument("--sup", default=None, choices=["boundary", "region"],
                    help="bac2 / bacq: supervise the saliency with the GT target boundary / region (training only)")
    ap.add_argument("--sup-weight", type=float, default=1.0)
    ap.add_argument("--feat-dir", default=None, help="feature folder (default data/bottleneck)")
    ap.add_argument("--max-steps", type=int, default=0, help="cap the number of steps (smoke tests)")
    ap.add_argument("--out-dir", default=None, help="results folder (default results/bottleneck)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    dev = pick_device()
    bac_kw = dict(rho=args.rho, alpha_init=args.alpha, sal_weight=args.sal_weight,
                  use_edge=args.alpha < 1.0, use_learned=args.alpha > 0.0) if args.kind in ("bac", "bac2", "bacq") else {}
    if args.kind in ("bac2", "bacq"):
        bac_kw.update(select=args.select, sup=args.sup, sup_weight=args.sup_weight)
    model = (ScatterTestbed if args.decoder == "scatter" else Testbed)(args.kind, args.tokens, bac_kw).to(dev)
    train, val = load("train", args.feat_dir), load("val", args.feat_dir)
    n = len(train[3])
    steps = max(1, args.epochs * (n // args.batch))
    if args.max_steps:
        steps = min(steps, args.max_steps) if steps > 1 else args.max_steps
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    print(f"{args.name}: {args.kind} M={args.tokens} | params {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(compressor {sum(p.numel() for p in model.comp.parameters())/1e6:.2f}M) | {steps} steps on {dev}", flush=True)
    t0, log = time.time(), []
    for step in range(steps):
        idx = torch.randint(n, (args.batch,), generator=gen).tolist()
        x, m, q = batch(*train, idx, dev, True, gen)
        logits = F.interpolate(model(x, q, cells(m))[:, None], size=256, mode="bilinear", align_corners=False)[:, 0]
        loss = F.binary_cross_entropy_with_logits(logits, m) + dice_loss(logits, m)
        aux = getattr(model.comp, "aux_loss", None)
        if aux is not None:
            loss = loss + aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == steps - 1:
            rec = dict(step=step, loss=round(loss.item(), 4), sec=round(time.time() - t0, 1))
            if args.kind in ("bac", "bac2", "bacq"):
                rec["alpha"] = round(float(model.comp.alpha), 4)
            log.append(rec); print(rec, flush=True)

    rows = evaluate(model, val, dev)
    import pandas as pd
    df = pd.DataFrame(rows)
    out = args.out_dir or os.path.join(ROOT, "results", "bottleneck")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{args.name}.csv"), index=False)
    summ = dict(name=args.name, kind=args.kind, tokens=args.tokens, seed=args.seed, alpha=args.alpha, rho=args.rho, decoder=args.decoder, select=args.select,
                sup=args.sup, sup_weight=args.sup_weight,
                gIoU=df.iou.mean(), cIoU=df.inter.sum() / df.union.sum(), minutes=(time.time() - t0) / 60, log=log)
    json.dump(summ, open(os.path.join(out, f"{args.name}.json"), "w"), indent=1)
    print(f"{args.name}: gIoU {100*summ['gIoU']:.2f}  cIoU {100*summ['cIoU']:.2f}  ({summ['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
