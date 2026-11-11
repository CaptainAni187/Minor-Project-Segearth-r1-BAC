"""Vision-language fallback, step 2: Qwen2.5-VL-3B sees the image AND the question and returns a box for the region that
answers it, or null if nothing does. Runs in exploration/.venv-vlm (transformers 4.51).

  exploration/.venv-vlm/bin/python exploration/scripts/vlm_ground.py calib300 test300 valrest testsplit
  -> exploration/results/vlm/boxes_{set}.json   (box in original-image pixels, or null)
"""
import json
import os
import re
import sys
import time

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "exploration", "results", "vlm")
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = ("{q}\nThis is a satellite image. Find the one object or area in the image that answers the question. "
          "Reply only with JSON: {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"<object type>\"}}. "
          "If nothing in the image answers the question, reply {{\"bbox_2d\": null}}.")


def parse(text):
    m = re.search(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
    lab = re.search(r'"label"\s*:\s*"([^"]*)"', text)
    return ([float(v) for v in m.groups()] if m else None), (lab.group(1) if lab else "")


@torch.no_grad()
def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    for s in sys.argv[1:]:
        rows = json.load(open(os.path.join(OUT, f"inputs_{s}.json")))
        out, t0 = [], time.time()
        for i, r in enumerate(rows):
            msgs = [{"role": "user", "content": [{"type": "image", "image": r["image"]},
                                                 {"type": "text", "text": PROMPT.format(q=r["question"])}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, _ = process_vision_info(msgs)
            inp = proc(text=[text], images=imgs, return_tensors="pt").to(dev)
            gen = model.generate(**inp, max_new_tokens=64, do_sample=False)
            ans = proc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
            box, label = parse(ans)
            # Qwen2.5-VL answers in the pixel space of the resized input image: map back to the original image
            th, tw = inp["image_grid_thw"][0, 1].item() * 14, inp["image_grid_thw"][0, 2].item() * 14
            if box:
                box = [box[0] * r["w"] / tw, box[1] * r["h"] / th, box[2] * r["w"] / tw, box[3] * r["h"] / th]
            out.append(dict(name=r["name"], raw=ans, box=box, label=label, resized=[th, tw]))
            if i % 10 == 0:
                print(f"[{s}] {i + 1}/{len(rows)} {(time.time() - t0) / (i + 1):.1f}s/img  {ans[:90]!r}", flush=True)
        json.dump(out, open(os.path.join(OUT, f"boxes_{s}.json"), "w"), indent=0)
        print(f"[{s}] done: {sum(o['box'] is not None for o in out)}/{len(out)} with a box", flush=True)


if __name__ == "__main__":
    main()
