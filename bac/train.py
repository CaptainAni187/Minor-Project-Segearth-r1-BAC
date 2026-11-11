"""Train a drop-in connector with everything else frozen (frozen-LLM protocol).

  python bac/train.py --name bac64  --kind bac  --tokens 64
  python bac/train.py --name pool64 --kind pool --tokens 64
  python bac/train.py --name bac64_lam0 --kind bac --tokens 64 --sal-weight 0

Only model.mm_projector is trainable; the Swin-B encoder, Phi-1.5, D-Projector and mask
generator keep the released SegEarth-R1 weights, so any accuracy difference between runs
is attributable to the compression strategy alone. Loss = LLM CE + mask (focal/BCE + dice
+ class) as in the reference recipe, plus the BAC budget prior when applicable.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (DataCollector, EarthReason, ROOT, add_lora, connector_aux_loss, load_model, lora_state,
                    model_inputs, set_connector, to_device)


MASK_PATH = ("pixel_decoder", "predictor", "d_layers", "local_project", "SEG_token_projector", "origin_SEG_token_projector",
             "text_projector", "seg_query")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", required=True, choices=["released", "uniform", "pool", "random", "tome", "bac", "bac2", "bacq"],
                    help="released: keep (and fine-tune) the released connector")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the frozen model (faster)")
    # BAC ablation knobs
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5, help="initial alpha (1 = learned only, 0 = Sobel only)")
    ap.add_argument("--fixed-alpha", action="store_true")
    ap.add_argument("--no-pe", action="store_true")
    ap.add_argument("--no-nms", action="store_true")
    ap.add_argument("--sal-weight", type=float, default=1.0, help="lambda of the budget prior")
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank on every Phi-1.5 linear layer (0 = LLM frozen)")
    ap.add_argument("--mask-path", action="store_true",
                    help="fine-tune the mask path (pixel decoder, mask decoder, D-Projector, projectors); connector frozen")
    ap.add_argument("--aug", default="hflip", choices=["none", "hflip", "d4"], help="training augmentation")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation (micro-batches per optimiser step)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # frozen model in bf16 (its native release dtype): half the memory of fp32 + autocast copies
    tok, model, dev = load_model(dtype=torch.bfloat16 if args.amp else torch.float32)
    kwargs = {}
    if args.kind == "bac":
        kwargs = dict(rho=args.rho, alpha_init=args.alpha, learn_alpha=not args.fixed_alpha,
                      use_pe=not args.no_pe, use_nms=not args.no_nms, sal_weight=args.sal_weight,
                      use_edge=args.alpha < 1.0, use_learned=args.alpha > 0.0)
    if args.kind == "released":                                            # joint fine-tuning of the released system
        conn = model.get_model().mm_projector.float().train()
        for p in conn.parameters():
            p.requires_grad_(not args.mask_path)
    else:
        conn = set_connector(model, args.kind, args.tokens, **kwargs).train()   # connector stays fp32
    lora_params = add_lora(model, args.lora) if args.lora else []
    mask_params = []
    if args.mask_path:
        for n, p in model.named_parameters():
            if n.split(".")[0] in MASK_PATH:
                p.data = p.data.float(); p.requires_grad_(True); mask_params.append(p)
        print(f"mask path: {sum(p.numel() for p in mask_params)/1e6:.2f}M parameters")
    if args.amp:
        conn.register_forward_pre_hook(lambda m, a: (a[0].float(),))
        conn.register_forward_hook(lambda m, a, out: out.to(torch.bfloat16))
    params = [p for p in conn.parameters() if p.requires_grad] + lora_params + mask_params
    print(f"trainable params: connector {sum(p.numel() for p in conn.parameters() if p.requires_grad)/1e6:.2f}M, "
          f"LoRA {sum(p.numel() for p in lora_params)/1e6:.2f}M")

    ds = EarthReason(tok, "train", train=True)
    ds.aug = args.aug
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=DataCollector(tok),
                    num_workers=2, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, int(0.05 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))

    log, step, t0, micro = [], 0, time.time(), 0
    opt.zero_grad(set_to_none=True)
    while step < args.steps:
        for batch in dl:
            b = to_device(batch, dev)
            inp = model_inputs(b)
            if args.amp:
                inp["images"] = inp["images"].to(torch.bfloat16)
            out = model(**inp, masks=[m.float() for m in b["masks"]],
                        dataset_type=b["dataset_type"], output_attentions=False)
            loss = out.loss.float()
            aux = connector_aux_loss(model)
            if aux is not None:
                loss = loss + aux
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum:                       # gradient accumulation: optimiser step every `accum` batches
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            opt.zero_grad(set_to_none=True)
            rec = dict(step=step, loss=loss.item(), llm=out.loss_llm.item(), mask=out.loss_mask.item(),
                       dice=out.loss_dice.item(), aux=float(aux) if aux is not None else 0.0,
                       lr=sched.get_last_lr()[0], sec=time.time() - t0)
            if args.kind == "bac":
                rec["alpha"] = float(conn.alpha)
                rec["mean_a"] = float(conn.last_saliency.mean())
            log.append(rec)
            if dev.type == "mps":
                rec["mem_gb"] = torch.mps.driver_allocated_memory() / 1e9
            if step % 10 == 0 or step == 1:
                eta = (time.time() - t0) / step * (args.steps - step) / 60
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
                      f"eta={eta:.1f}min", flush=True)
            if step >= args.steps:
                break

    blob = dict(kind=args.kind, num_tokens=args.tokens, kwargs=kwargs, args=vars(args),
                state_dict={k: v.cpu() for k, v in conn.state_dict().items()},
                lora=lora_state(model) if args.lora else None, lora_r=args.lora,
                mask_path={n: p.detach().cpu() for n, p in model.named_parameters() if n.split(".")[0] in MASK_PATH} if args.mask_path else None)
    torch.save(blob, os.path.join(out_dir, "connector.pt"))
    json.dump(log, open(os.path.join(out_dir, "log.json"), "w"))
    print(f"saved {out_dir}/connector.pt  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
