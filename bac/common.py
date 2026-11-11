"""Shared utilities: load SegEarth-R1 on Apple Silicon / CUDA, swap the connector, data, metrics."""
import json
import math
import os
import random
import sys
import warnings

# PyTorch only: stop transformers from importing JAX / TensorFlow (on Kaggle, JAX needs a different NumPy)
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")
import cv2
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.join(ROOT, "SegEarth-R1")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "segearth_r1", "eval_and_test"))

from segearth_r1 import conversation as conversation_lib  # noqa: E402
from segearth_r1.model.builder import load_pretrained_model  # noqa: E402
from segearth_r1.model.multimodal_projector.token_compressors import build_connector  # noqa: E402
from eval_dataset.RS_val_dataset import (  # noqa: E402,F401  (DataCollector is re-exported)
    DataCollector, ReasonSegDataset, preprocess_image, preprocess_mask, preprocess_llama2,
    preprocess_referring_instruction, REFER_TOKEN_INDEX, ANSWER_TOKEN_INDEX)

CKPT = os.environ.get("SEGEARTH_CKPT", os.path.join(ROOT, "checkpoints", "SegEarth-R1-EarthReason"))
DATA = os.environ.get("SEGEARTH_DATA", os.path.join(ROOT, "data"))
MASK_CFG = os.path.join(REPO, "segearth_r1", "mask_config", "maskformer2_swin_base_384_bs16_50ep.yaml")
PREFIX = 'This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:'
NO_TARGET = "There is no target object in the image."
# set at import so spawned DataLoader workers (macOS) see the same template
conversation_lib.default_conversation = conversation_lib.conv_templates["llava_phi"]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Args:
    model_map_name = "segearth_r1"
    seg_task = "referring"


class FrozenCachedTower(nn.Module):
    """Frozen Swin-B that memoises its last output.

    SegEarth-R1 calls the vision tower twice per forward (once for the connector, once
    for the mask generator). With a frozen encoder both calls are identical, so we run it
    once under no_grad and return the cached features the second time.
    """

    def __init__(self, tower):
        super().__init__()
        self.tower = tower
        self._key, self._out = None, None

    def forward(self, images):
        # pointer alone is unsafe: a freed batch's address is reused by the next one, so add a
        # cheap content fingerprint
        flat = images.detach().flatten()
        key = (images.data_ptr(), tuple(images.shape), images._version,
               float(flat[:: max(1, flat.numel() // 4096)].float().sum()))
        if key != self._key:
            with torch.no_grad():
                self._out = [f.detach() for f in self.tower(images)]
            self._key = key
        return self._out


def load_model(device=None, dtype=torch.float32):
    device = device or pick_device()
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_phi"]
    tokenizer, model, _ = load_pretrained_model(CKPT, None, "segearth_r1", model_args=_Args(),
                                                mask_config=MASK_CFG, device="cpu")
    model.config.output_attentions = False
    model.get_model().vision_tower = FrozenCachedTower(model.get_model().vision_tower)
    model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model, device


def set_connector(model, kind, num_tokens=64, **kw):
    """Replace model.mm_projector with a fresh drop-in connector (trainable)."""
    ref = next(model.parameters())
    conn = build_connector(kind, 1024, model.config.hidden_size, num_tokens=num_tokens, **kw)
    conn.to(device=ref.device, dtype=torch.float32)
    model.get_model().mm_projector = conn
    return conn


class LoRALinear(torch.nn.Module):
    """y = W x + (B A x) * alpha / r, with W frozen and A, B trainable (B starts at 0: identical model at step 0)."""

    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.lora_A = torch.nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        return y + ((x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()).to(y.dtype) * self.scale


LORA_TARGETS = ("query_key_value", "dense", "fc1", "fc2")


def add_lora(model, r=16, alpha=32):
    """Wrap every linear layer of the Phi-1.5 blocks with LoRA; returns the new trainable parameters."""
    params = []
    for name, mod in list(model.named_modules()):
        if not name.startswith("model.layers."):
            continue
        for child, sub in list(mod.named_children()):
            if child in LORA_TARGETS and isinstance(sub, torch.nn.Linear):
                wrapped = LoRALinear(sub, r, alpha)
                setattr(mod, child, wrapped)
                params += [wrapped.lora_A, wrapped.lora_B]
    return params


def lora_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}


def connector_aux_loss(model):
    return getattr(model.get_model().mm_projector, "aux_loss", None)


# data
class EarthReason(ReasonSegDataset):
    """EarthReason split. Train: random question/answer per visit. Eval: first question."""

    def __init__(self, tokenizer, split="val", train=False, limit=None, seed=0):
        self.train_mode = train
        sp = split
        # the upstream class only whitelists val/test, so load train files ourselves
        super().__init__(DATA, tokenizer, "val" if split == "train" else split)
        if split == "train":
            base = os.path.join(DATA, "rs_reason_seg/RSReasonSeg/train")
            self.images = self.load_file_paths(os.path.join(base, "images"), ('.jpg', '.jpeg', '.png'))
            self.labels = self.load_file_paths(os.path.join(base, "labels"), ('.png',))
            paths = self.load_file_paths(os.path.join(base, "QAs"), ('.json', '.txt'))
            self.QAs = [json.load(open(p)) for p in paths]
        self.split = sp
        if limit is not None and limit < len(self.images):
            order = list(range(len(self.images)))
            random.Random(seed).shuffle(order)
            keep = sorted(order[:limit])
            self.images = [self.images[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            self.QAs = [self.QAs[i] for i in keep]

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1
        qa = self.QAs[idx]
        if self.train_mode:
            aug = getattr(self, "aug", "hflip")
            if aug in ("hflip", "d4") and random.random() < 0.5:   # horizontal flip augmentation
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            if aug == "d4":                                          # + vertical flip and 90-degree rotations
                if random.random() < 0.5:
                    image, mask = image[::-1].copy(), mask[::-1].copy()
                k = random.randrange(4)
                image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
            question = random.choice(qa["questions"])
            answer = random.choice(qa["answer"]) if qa["answer"] else NO_TARGET
        else:
            question = qa["questions"][0]
            answer = qa["answer"][0] if qa["answer"] else NO_TARGET
        return build_sample(self.tokenizer, image, question, answer, mask,
                            os.path.basename(self.images[idx]).split(".")[0])


PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


def build_sample(tokenizer, image_rgb, question, answer=NO_TARGET, mask=None, name="query"):
    """Same tokenisation and preprocessing as the upstream ReasonSegDataset."""
    d = {}
    img = preprocess_image(image_rgb, 1024)
    d["image"] = (torch.tensor(img) - PIXEL_MEAN) / PIXEL_STD
    d["image_name"] = name
    if mask is None:
        mask = np.zeros(image_rgb.shape[:2], np.uint8)
    d["mask"] = preprocess_mask(mask, 1024)
    sources = [[{"from": "human", "value": PREFIX + "\n<refer>"},
                {"from": "gpt", "value": "Sure, It is <seg>. \n<answer>."}]]
    text = preprocess_llama2(sources, tokenizer)
    d["input_ids"], d["labels"] = text["input_ids"][0], text["labels"][0]
    d["dataset_type"] = "reason_seg"
    d["token_refer_id"] = preprocess_referring_instruction(" " + question, tokenizer)
    d["token_answer_id"] = preprocess_referring_instruction(answer, tokenizer)
    d["refer_embedding_indices"] = (d["input_ids"] == REFER_TOKEN_INDEX).long()
    d["answer_embedding_indices"] = (d["input_ids"] == ANSWER_TOKEN_INDEX).long()
    return d


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def model_inputs(batch):
    return dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                images=batch["images"].float(), token_refer_id=batch["token_refer_id"],
                refer_embedding_indices=batch["refer_embedding_indices"], labels=batch["labels"],
                token_answer_id=batch.get("token_answer_id"),                 # absent for referring (RRSIS-D)
                answer_embedding_indices=batch.get("answer_embedding_indices"))


@torch.no_grad()
def predict(model, batch):
    """Returns list of binary [1024,1024] masks (top-scoring query, as in upstream eval)."""
    outs = model.eval_seg(**model_inputs(batch), output_attentions=False)
    preds = []
    for r in outs:
        pm = r["pred_masks"]
        if r["scores"] is not None:
            pm = pm[torch.tensor(r["scores"]).argmax()]
        preds.append(pm.reshape(pm.shape[-2:]).bool())
    return preds


# metrics
def iou_parts(pred, gt):
    """Foreground intersection/union exactly as upstream (empty union counts as IoU 1)."""
    pred, gt = pred.bool(), gt.bool()
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return inter, union, 1.0 if union == 0 else inter / union


def mask_geometry(gt):
    """Target area in 32x32 feature-grid cells and compactness 4*pi*A/P^2 (largest contour)."""
    g = gt.cpu().numpy().astype(np.uint8)
    area_px = int(g.sum())
    cells = area_px / (32 * 32)            # one cell of the 32x32 grid is 32x32 pixels at 1024^2
    if area_px == 0:
        return dict(area_px=0, area_cells=0.0, compactness=np.nan, n_components=0)
    contours, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in contours)
    comp = float(4 * np.pi * area_px / (per ** 2 + 1e-6))
    return dict(area_px=area_px, area_cells=cells, compactness=min(comp, 1.0),
                n_components=len(contours))


def boundary_grid(gt, dilate=1):
    """Ground-truth boundary on the 32x32 token grid (dilated by `dilate` cells)."""
    g = gt.float()[None, None]
    small = torch.nn.functional.adaptive_avg_pool2d(g, 32)[0, 0]   # fraction of target per cell
    edge = (small > 0) & (small < 1)
    k = 2 * dilate + 1
    edge = torch.nn.functional.max_pool2d(edge.float()[None, None], k, 1, dilate)[0, 0] > 0
    return edge


def saliency_agreement(selected_idx, gt):
    """Fraction of BAC boundary picks that land on the (dilated) GT boundary grid cells."""
    if selected_idx is None or gt.sum() == 0:
        return np.nan
    edge = boundary_grid(gt.cpu()).flatten()
    return edge[selected_idx.cpu()].float().mean().item()


def phi_prefill_flops(seq_len, n_layers=24, d=2048, d_ff=8192):
    """Approximate Phi-1.5 prefill FLOPs (2 x MACs) for a sequence of seq_len tokens."""
    per_tok = n_layers * (2 * (4 * d * d + 2 * d * d_ff))
    attn = n_layers * 2 * (2 * seq_len * seq_len * d)
    return seq_len * per_tok + attn


@torch.no_grad()
def predict_logits(model, batch):
    """Mask logits [1024,1024] per sample (before the >0 threshold), via the upstream eval_seg path."""
    store = []
    orig = model.SEG_instance_inference

    def capture(seg_cls, mask_pred):
        store.append(mask_pred.detach())
        return orig(seg_cls, mask_pred)

    model.SEG_instance_inference = capture
    try:
        model.eval_seg(**model_inputs(batch), output_attentions=False)
    finally:
        del model.SEG_instance_inference          # restore the class method
    return [m.reshape(-1, *m.shape[-2:])[0] for m in store]


FLIPS = {1: [()], 2: [(), (-1,)], 4: [(), (-1,), (-2,), (-2, -1)]}


@torch.no_grad()
def predict_tta(model, batch, n=2):
    """Flip test-time augmentation in the mask path: average un-flipped logits, then threshold.
    n = 8: the full dihedral group D4 (4 rotations x optional horizontal flip)."""
    if n == 8:
        acc = None
        for k in range(4):
            for fl in (False, True):
                b = dict(batch)
                im = torch.rot90(batch["images"], k, (-2, -1))
                b["images"] = torch.flip(im, (-1,)) if fl else im
                lg = predict_logits(model, b)
                lg = [torch.rot90(torch.flip(l, (-1,)) if fl else l, -k, (-2, -1)) for l in lg]
                acc = lg if acc is None else [a + l for a, l in zip(acc, lg)]
        return [(a / 8) > 0 for a in acc]
    acc = None
    for dims in FLIPS[n]:
        b = dict(batch)
        if dims:
            b["images"] = torch.flip(batch["images"], dims)
        logits = predict_logits(model, b)
        if dims:
            logits = [torch.flip(l, dims) for l in logits]
        acc = logits if acc is None else [a + l for a, l in zip(acc, logits)]
    return [(a / len(FLIPS[n])) > 0 for a in acc]
