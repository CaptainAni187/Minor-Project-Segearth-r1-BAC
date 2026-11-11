"""Infer the target's object type from the implicit question alone (no image, no reference answer).

Runs in the separate environment exploration/.venv-llm (newer transformers for Qwen2.5). Two modes per question:
  free  : a short noun phrase usable as an open-vocabulary detection query
  vocab : one choice from a generic remote-sensing vocabulary = the 45 NWPU-RESISC45 scene classes + the DOTA
          object classes (public taxonomies, not derived from EarthReason annotations)
Which mode is used downstream is decided on calib300 only.

  exploration/.venv-llm/bin/python exploration/scripts/predict_noun.py calib300 test300 [--model Qwen/Qwen2.5-3B-Instruct]
  -> exploration/results/nouns/{set}_{model-short}.json
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NWPU = ["airplane", "airport", "baseball diamond", "basketball court", "beach", "bridge", "chaparral", "church",
        "circular farmland", "cloud", "commercial area", "dense residential", "desert", "forest", "freeway",
        "golf course", "ground track field", "harbor", "industrial area", "intersection", "island", "lake", "meadow",
        "medium residential", "mobile home park", "mountain", "overpass", "palace", "parking lot", "railway",
        "railway station", "rectangular farmland", "river", "roundabout", "runway", "sea ice", "ship", "snowberg",
        "sparse residential", "stadium", "storage tank", "tennis court", "terrace", "thermal power station", "wetland"]
DOTA = ["plane", "large vehicle", "small vehicle", "helicopter", "helipad", "soccer ball field", "swimming pool",
        "container crane", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor",
        "bridge", "roundabout", "storage tank", "ship", "airport"]
VOCAB = sorted(set(NWPU + DOTA))

SYSTEM = ("You help a segmentation system for satellite and aerial images. Given a question about an aerial image, "
          "name the type of object or area in the image that answers it.")


def ask(tok, model, user, max_new=10):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().strip(".").lower()


def main():
    args = sys.argv[1:]
    name = args[args.index("--model") + 1] if "--model" in args else "Qwen/Qwen2.5-3B-Instruct"
    sets = [a for a in args if not a.startswith("--") and a != name]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    out_dir = os.path.join(ROOT, "exploration", "results", "nouns")
    os.makedirs(out_dir, exist_ok=True)
    short = name.split("/")[-1]
    for s in sets:
        meta = json.load(open(os.path.join(ROOT, "exploration", "results", "logits", f"{s}_meta.json")))
        if "--refusals-only" in args:                    # the policy only needs a name where SegEarth-R1 predicts nothing
            meta = [m for m in meta if m["pred_area_full"] == 0]
        if "--refusals-or-heldout" in args:              # + held-out questions, only to measure name accuracy (diagnostic)
            meta = [m for m in meta if m["pred_area_full"] == 0 or m["category"].startswith("held-out")]
        res, t0 = {}, time.time()
        for i, m in enumerate(meta):
            q = m["question"]
            free = ask(tok, model, f"Question: {q}\nAnswer with only the object or area type, in 1 to 3 words "
                                   f"(for example: helipad, parking lot, river).")
            voc = ask(tok, model, f"Question: {q}\nChoose the single best answer from this list and reply with it exactly:\n"
                                  f"{', '.join(VOCAB)}")
            if voc not in VOCAB:                                  # tolerate small formatting deviations
                hits = [v for v in VOCAB if v in voc]
                voc = max(hits, key=len) if hits else voc
            res[m["name"]] = dict(question=q, free=free, vocab=voc, vocab_valid=voc in VOCAB)
            if i % 50 == 0:
                print(f"[{s}] {i + 1}/{len(meta)} {(time.time() - t0) / (i + 1):.2f}s/q | {q[:60]} -> {free} | {voc}", flush=True)
        json.dump(dict(model=name, vocab=VOCAB, predictions=res), open(os.path.join(out_dir, f"{s}_{short}.json"), "w"), indent=1)
        print(f"[{s}] done ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
