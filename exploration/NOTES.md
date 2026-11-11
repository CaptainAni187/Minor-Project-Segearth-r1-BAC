# Exploration log: improving SegEarth-R1's segmentation accuracy

This is our working log for the second half of the project, where we tried to improve SegEarth-R1's accuracy
directly. Everything here lives in `exploration/` (scripts and results), plus a few small additions to `bac/`
and the connector file that do not change any earlier result.

## Protocol (fixed before any tuning)
- **test300**: the 300-image validation subset used for every number so far (val shuffle seed 0, first 300).
- **calib300**: the next 300 images of the same shuffle (disjoint). Any rule, threshold or hyper-parameter is
  chosen here only.
- **testsplit**: the official EarthReason *test* split (1,928 images) for the final confirmation of whatever is
  chosen, run once.
- Metric: upstream gIoU / cIoU at the released 1024x1024 letterboxed resolution; paired bootstrap 95% CIs over
  images (5,000 resamples, `bac/analyze.paired_bootstrap`). Screening may use 512x512 logits; claims use full res.

## What is already known (from earlier work)
- The connector / visual tokens do not affect the masks (controls, two datasets, two machines), so compression
  is not a lever on accuracy in this model.
- Flip TTA: +0.5 gIoU, not significant (300 images).

## Bottleneck diagnosis (2026-10-17, from results/released_val.csv)
Released model, test300: gIoU 68.40, i.e. 31.6 points lost.

| IoU bin | n | points lost |
|---|---|---|
| miss (< 0.05) | 33 | **11.00** |
| 0.05 - 0.3 | 14 | 3.84 |
| 0.3 - 0.5 | 27 | 5.33 |
| 0.5 - 0.7 | 35 | 4.60 |
| 0.7 - 0.85 | 54 | 3.66 |
| > 0.85 | 137 | 3.17 |

- **Half of the lost accuracy is localisation failure, not boundary quality.**
- 26 of the 33 misses are in the four categories withheld from training. Miss rate: lake 55%, island 64%,
  basketball court 47%, stadium 6%, seen categories 1.7%. Held-out categories are 64 of the 300 images.
- 17 of the 33 misses are **empty predictions** (all mask logits < 0); 3 misses are the empty-target images.
- So the dominant failure is generalisation to unseen categories, often as a refusal (empty mask).

## Open questions and candidate directions (ranked by expected value / cost)
1. Do the logits of empty predictions still peak on the target? -> a principled non-empty fallback
   (inference only; risk: the ~1.3% empty-target queries). Screen on calib300 from dumped logits.
2. Class-agnostic refinement with SAM (prompted from SegEarth-R1's logits): may recover whole objects for unseen
   categories (SAM is category-agnostic, trained on SA-1B) and fix boundaries in the 0.3-0.85 bins. Hypothesis,
   to be tested; prompts derived from the model's own output only (no GT).
3. Mask threshold calibration / flip TTA (screen from the same dump).
4. Joint fine-tuning (LoRA on Phi-1.5 + connector): could make the LLM use vision; risk of overfitting
   (training answers are memorised: text loss 0.20 on train vs ~3.1 on val). Cannot use held-out categories.
5. Answer likelihood (does the LLM's text use the tokens?): relevant to the BAC contribution, not to accuracy.

## Running / done
- Kaggle `bac-explore` (testbed): oracle and supervised allocation, rho 0.75, 16/64 tokens x 3 seeds.
- Mac: `exploration/scripts/dump_logits.py test300 calib300` -> `exploration/results/logits/`.

## Results log

### E1. Logit dump (exploration/scripts/dump_logits.py) - done
- test300 full-res gIoU at threshold 0 = **68.40** (identical to the released number: sanity check passed);
  calib300 = 71.68. 512x512 screening tracks full res within 0.13 gIoU.

### E2. Do refusals still point at the target? - **No** (test300 diagnostic)
- 18 empty predictions; max logit -2.7 to -16.3 (deep negatives); arg-max inside the ground truth in 2/18.
- => the decoder has no signal for these unseen-category queries; logit-level rescue cannot work.

### E3. Logit post-processing (screen_rules.py; chosen on calib300, reported on test300) - **all negative**
| rule (chosen on calib) | calib gain | test300 delta [95% CI] |
|---|---|---|
| threshold sweep | best t = 0 (released) | 0 |
| flip TTA (+ threshold) | -1.68 | -0.00 [-1.48, +1.55] |
| empty -> top band fallback | -0.24 | -0.23 [-0.95, +0.19] |
- Flip TTA hurts on calib300 although it gave +0.5 (n.s.) on test300 earlier: that earlier gain was noise.

### E4. SAM refinement prompted by SegEarth-R1's output (sam_refine.py, screen_sam.py; SAM ViT-B) - **negative**
- Replacing SegEarth-R1 masks by SAM masks hurts seen categories badly (77.0 -> 62-69): SegEarth-R1's own masks
  are better than box-prompted SAM masks, so boundary refinement is rejected.
- SAM only on refusals (prompt from the logit band): calib +0.39, test300 +1.24 [-0.02, +2.57] (held-out 39.7 ->
  47.1, but the one correctly-empty target becomes wrong). Weak, not significant; superseded by E6 if E6 works.

### E5. Open-vocabulary ceiling (ov_ceiling.py; calib300 held-out images, n = 58; **oracle names, diagnostic only**)
| | mean IoU |
|---|---|
| SegEarth-R1 | 0.511 |
| OWLv2 top box + SAM | 0.576 |
| best of OWLv2 top-5 boxes + SAM | 0.767 |
- court 0.48 -> 0.68, island 0.31 -> 0.62; lake 0.64 -> 0.45, stadium 0.58 -> 0.52 (top box).
- Routing to OV on refusal: held-out 0.511 -> 0.584; on max logit < 10: 0.628 (oracle names).
- => an OV branch is worth building if the name can be inferred from the question alone (E6).

### E6. Hybrid: LLM-inferred name -> OWLv2 -> SAM, routed (predict_noun.py, ov_branch.py, screen_hybrid.py)
- Names from Qwen2.5-3B-Instruct (separate env exploration/.venv-llm), question text only.
- Name accuracy on held-out categories: calib300 vocab 30/58, free 23/58; test300 vocab 34/64, free 26/64
  (stadium ~all right; lakes called wetland/river; islands hardest).
- Free-mode names, calib300 (512 screen): OV top box on held-out 45.1 vs SegEarth-R1 51.1 (worse); routing on
  refusal +0.45 gIoU (10 routed); routing on max logit < 10: -2.86 (seen images get routed).
- => with imperfect names the branch only helps on refusals (~3% of queries); ceiling with perfect names ~ +1.4.
  Vocab-mode run pending.

### E7. Where the partial masks lose (test300, IoU 0.05-0.85: 130 images, 17.4 points)
- under-segmentation (prediction inside GT, median area ratio 0.75): 67 images, 7.8 points (24 multi-part GT)
- over-segmentation (GT inside prediction, ratio 1.46): 43 images, 6.1 points; both wrong: 20 images, 3.5
- errors go both ways, consistent with threshold 0 being optimal (E3).

### E8. Testbed: supervised / oracle allocation (Kaggle bac-explore; 3 seeds; testbed gIoU)
| | 16 tok | 64 tok |
|---|---|---|
| strided conv | 44.20 | 48.28 |
| no compression | 52.83 | 52.83 |
| **BAC-q, saliency supervised by GT boundary (training only)** | **45.70** | **53.42** |
| BAC-q, supervised by GT region | 45.11 | 51.02 |
| BAC-v2, rho 0.75 | 39.94 | 46.04 |
| oracle boundary / region (leak: positions reveal the target) | 55.51 / 58.24 | 69.21 / 70.80 |
- boundary-supervised BAC-q - strided conv: +1.50 [+0.17, +2.86] @16, **+5.15 [+4.01, +6.30] @64**; vs no
  compression @64: +0.60 [-1.23, +2.45] (tie); vs unsupervised BAC-q +3.18 / +5.99. rho 0.75 hurts.
- Open: is it allocation or the auxiliary loss? Controls queued (bac-explore2): same supervision + random
  placement (supbrand); same supervised head with all tokens uniform (supbr0).

### E9. Mask-path fine-tuning with D4 augmentation (Kaggle bac-ft) - running
- Upstream trained 15 epochs, full fine-tuning, **no geometric augmentation**; the model is not flip-invariant
  (flip TTA hurts, E3) and memorises training answers.
- ft_d4: mask path (pixel decoder, mask decoder, D-Projector, projectors; 23M params) from released weights, D4
  augmentation, 2000 steps x batch 4, lr 3e-5 cosine; ft_ctrl: same without augmentation. Full val + 8-way TTA.


### E8b. Allocation controls (Kaggle bac-explore2; 3 seeds; paired 95% CIs, testbed gIoU)
| comparison | 16 tok | 64 tok |
|---|---|---|
| supervised placement - same model, random placement | +4.49 [+3.34, +5.68] | +5.76 [+4.65, +6.92] |
| supervised placement - same supervised head, all tokens uniform (rho 0) | -1.08 [-2.34, +0.23] | **+2.83 [+1.67, +4.01]** |
| auxiliary supervision alone (rho 0) - strided conv | +2.58 [+1.28, +3.83] | +2.32 [+1.15, +3.49] |
- At 64 tokens the +5.15 over the strided conv decomposes into ~+2.3 from the auxiliary boundary task and ~+2.8
  from boundary-aware **allocation** itself. At 16 tokens allocation adds nothing beyond the supervision (too few
  tokens to split). Testbed result only: SegEarth-R1's masks ignore the visual tokens.

### E9 result. Mask-path fine-tuning (Kaggle bac-ft v2; 1500 steps, batch 1 x accum 4, lr 3e-5)
| (paired vs released, full res) | calib300 | test300 | seen | held-out |
|---|---|---|---|---|
| ft_ctrl (no aug) | +0.05 [-1.65, +1.65] | +0.10 [-1.45, +1.68] | +0.2..+1.2 | -0.2..-4.8 |
| ft_d4 | -1.21 [-3.18, +0.64] | -0.25 [-1.86, +1.23] | +0.6..+0.9 | -4.4..-8.6 |
| ft_d4 + 8-way TTA | -0.47 [-2.44, +1.43] | +0.74 [-1.13, +2.59] | +1.5..+2.1 | -4.3..-8.5 |
| released + 8-way TTA | -1.67 [-3.62, +0.06] | +0.10 [-1.29, +1.45] | | |
- No reliable gain. D4 augmentation (+TTA) helps the seen categories but further training on them hurts the
  unseen ones: the limiting factor is generalisation to new categories, not segmentation of known ones.
- v1 of this kernel failed in its smoke test (CUDA OOM at batch 2 fp32) -> batch 1 with accumulation.

### E6 result (calib300). Name mode and routing chosen on calib300
- vocab-mode names: refusal routing **+1.40** gIoU (full res; 10 routed); free-mode +0.45. Gating on the OWLv2
  score does not help (+1.07 / +1.26 at 0.2 / 0.3). OV top-1 box = OV best (owl x sam) here.

## Pre-registered final protocol (written before any test300 / test-split outcome of the hybrid)
**Policy P\*:** SegEarth-R1 exactly as released. If its mask is empty (all logits < 0), replace it by:
Qwen2.5-3B-Instruct vocab-mode target name from the question only -> OWLv2 (base, ensemble) with the text
"a satellite image of a {name}" -> top-1 box -> SAM ViT-B, the multimask output with the highest predicted IoU.
No other parameter. Evaluation: full-resolution gIoU / cIoU (upstream definitions, 1024 letterbox) on
(1) test300 and (2) the official EarthReason **test split** (1,928 images, never used before), paired bootstrap
95% CI vs the released model; report the number routed and the result on routed images.

### E10. Policy P* on test300 (first held-out check, policy fixed on calib300)
- calib300 (selection set): 71.68 -> 73.08, +1.40 [-0.00, +2.92], 10 routed (8 held-out, 1 empty target).
- **test300: 68.40 -> 70.53, +2.12 [+0.50, +3.87]**, 18 routed (16 held-out, 1 empty target, 1 seen);
  mean IoU on routed images 0 -> 40.9 (the one correctly-empty target is lost: 1.0 -> 0.0).
- Failures on routed images are mostly wrong names ('beach' / 'golf course' for lakes, 'lake' for an island);
  correct names usually give IoU 0.7-0.99 (basketball courts, lakes).
- Official test split (1,928 images) run started: exploration/scripts/run_testsplit.sh.

### E11. Pre-registered P* on the official test split (1,928 images) - **mixed**
| | released | P* | delta [95% CI] |
|---|---|---|---|
| gIoU | 70.40 (paper 70.75) | 71.36 | **+0.96 [+0.36, +1.57]** |
| cIoU | 68.05 (paper 68.25) | 66.51 | **-1.54 [-2.91, -0.24]** |
- 88 routed (4.6%): mean IoU 13.6 -> 34.6. basketball court 45.0 -> 61.5, lake 51.7 -> 58.4, stadium 47.6 -> 49.1;
  islands never routed (the model is confidently wrong on them, not refusing); empty targets 30.8 -> 0.0 (12 lost).
- Post-hoc diagnosis (test split; used for nothing but motivating P2): cIoU effect by routed group:
  wrong object (IoU < 0.1, n = 33) -2.48, empty targets (12) -0.38, partial (11) -0.10, good (32) +1.43.
  OWLv2 top score separates them: median 0.55 good vs 0.10 wrong vs 0.19 empty.
- Conclusion: P* improves the per-image metric but hurts the pixel-cumulative one (paper prefers cIoU): **not a
  clean improvement**. The test split is now spent for P*; anything after this is evaluated elsewhere.

## Pre-registered P2 protocol (written before any val-rest outcome)
- P2 = P* + gate: route a refusal only if OWLv2's top score >= tau; names from Qwen2.5-{3B,7B}-Instruct (vocab mode).
- **Selection** on dev = calib300 + test300 (600 images, 28 refusals): maximise gIoU subject to delta cIoU >= 0.
  3B result: tau = 0.35 -> gIoU +1.39 [+0.59, +2.34], cIoU +0.38 [-1.21, +1.92] (no gate: +1.76 / -1.48).
- **Confirmation** on val-rest = val minus test300/calib300 (535 images, never used for anything before).
- Test split reported only as secondary for P2 (its gate was motivated by test-split diagnosis).

### E12. P2 = gated hybrid (selected on dev, confirmed on val-rest) - **best supported improvement**
Selection on dev (calib300 + test300): 7B names, tau = 0.25 -> gIoU +1.82 [+0.92, +2.86], cIoU +0.34 [-1.53, +2.18].
| set | role | n | routed | delta gIoU [95% CI] | delta cIoU [95% CI] |
|---|---|---|---|---|---|
| **val-rest** | **clean confirmation** | 535 | 19 | **+1.62 [+0.75, +2.66]** | **+1.16 [-0.13, +2.66]** |
| test split | secondary (gate motivated by its P* diagnosis) | 1,928 | 49 | +1.00 [+0.49, +1.53] | +0.83 [-0.11, +1.77] |
- Released on val-rest 71.20 / 66.77 -> P2 72.83 / 67.93. Gate is essential: ungated on val-rest gIoU +0.72 (n.s.),
  cIoU -1.10. 3B names with tau 0.35 behave almost identically (test split +1.04 / +0.95, both significant).
- Per category (test split): basketball court 45.0 -> 61.5, lake 51.7 -> 54.7, stadium 47.6 -> 49.4, seen 77.7 ->
  77.8, island unchanged (never refused), empty targets 30.8 -> 15.4 (6 of 12 correct empties lost).
- Cost: only gated refusals (~3-4% of queries) pay extra (~3 s 7B name + ~2 s OWLv2 on CPU + ~0.5 s SAM on M4);
  ~+0.2 s per query on average; ~17 GB extra model weights (Qwen2.5-7B fp16, OWLv2, SAM-B).
- Results file: exploration/results/p2_results.json.

## Summary of the exploration (2026-10-18)
- Bottleneck: generalisation to the 4 held-out categories (half of the lost accuracy is complete misses, mostly
  refusals on unseen categories); the decoder has no signal there.
- Work: gated open-vocabulary fallback (P2) - consistent ~+1.0-1.6 gIoU and ~+0.8-1.2 cIoU on two held-out sets.
- Work (methodological, testbed only): question-aware boundary allocation **with saliency supervision** beats the
  strided conv by +5.15 at 64 tokens; ~+2.8 of it is allocation itself (controls E8b).
- Do not work: threshold calibration, flip / D4 TTA, logit-peak fallback, SAM replacement/refinement, mask-path
  fine-tuning (seen up, unseen down), BAC-v2 rho 0.75, unsupervised question-aware saliency.
- Not tested (and why): joint LoRA fine-tuning of Phi-1.5 (mask-path FT already showed that more training on seen
  categories hurts unseen ones; the LoRA path is implemented and smoke-tested in bac/train.py --lora); answer-text
  likelihood check (dropped: memory contention; relevant to BAC, not to accuracy); VLM grounding (e.g. Qwen2-VL)
  instead of LLM name + OWLv2; routing for confidently-wrong unseen cases (islands are never refused);
  OWLv2-large / SAM-L/H; supervised allocation in a model whose masks depend on the tokens.
- Next steps (ranked): (1) better grounding for refusals: VLM box grounding or OWLv2-large + reranking (best-of-5
  boxes reaches 0.77 IoU vs 0.58 top-1 with correct names, E5); (2) a confidence signal for confidently-wrong
  unseen predictions (islands); (3) protect empty targets (6 of 12 still lost); (4) evaluate supervised allocation
  inside an MLLM that has no spatial bypass.

### E13. Better box selection inside P2 - **no gain, stopped** (2026-10-18)
- Headroom (perfect choice among the 5 OWLv2+SAM candidates, routed images only), as overall gIoU:
  calib300 +0.22, test300 +0.94, val-rest +0.84, test split +0.41 (top-1 IoU 0.46-0.63 vs best-of-5 0.68-0.83).
- Parameter-free rules vs the current top-1 (overall gIoU): owl x sam = identical; best SAM among owl >= 0.5 top:
  -0.58 / +0.30 / -0.13 / -0.10; largest: 0.00 / +0.04 / -0.03 / +0.02; smallest: -0.70 / +0.13 / -0.38 / -0.29.
- => none beats the top OWLv2 box consistently; the best-of-5 headroom needs information the system does not have.
  A learned reranker was not attempted: 18 routed dev images are too few to fit it without overfitting.
  P2 (E12) stays the final configuration. Exploration closed here.

## Final best shot (2026-10-20)

### F1. Audit findings
- **Reproduction wording was misleading.** test300 + calib300 + val-rest = all 1,135 validation images: the released
  model scores **70.59 gIoU / 66.00 cIoU on the full validation set** (paper 68.60 / 64.13) and 70.40 / 68.05 on the
  official test split (paper 70.75 / 68.25). Our 68.40 on test300 vs the paper's 68.60 was a coincidence of the subset.
  Metric and protocol match upstream (first question per image, foreground IoU, empty-vs-empty = 1).
- **Training-data artifact:** 60 training samples have empty targets; 8 explicitly ask about basketball courts
  (hoops) and most others about other sports courts. The model was taught "court question -> nothing", which matches
  courts having the highest empty-output rate (29%) on the test split.

### F2. Novelty-aware routing (route non-empty answers too when the target type looks unseen) - **rejected**
- Ceiling on dev held-out images the model did NOT refuse (n = 98, 3B names): SegEarth-R1 0.562, OV top box 0.525,
  per-image best 0.724, but no available signal picks the winner (corr with OWLv2 score 0.06, with SegEarth max logit
  -0.08). A known-type list from training answers flags 243 seen images as unknown (-20 gIoU if routed).

### F3. WiSE-FT weight interpolation of the fine-tuned mask path - **rejected (partial screen)**
- calib300: alpha 1.0 = 70.47 (matches Kaggle ft_d4: code sanity), alpha 0.5 = 71.14 vs released 71.68;
  seen 77.8 -> 78.3, unseen 51.1 -> 46.4. alpha 0.25 / 0.75 not run (stopped: memory contention, and alpha 0.5 already
  shows the unseen loss dominating; smaller alpha can only trade less of both).

### F4. Vision-language fallback (Qwen2.5-VL-3B sees image + question, returns a box; SAM-B) - **rejected on dev**
- Separate env exploration/.venv-vlm (transformers 4.51). It gave a box for all 28 dev refusals (never "null"), so its
  built-in "nothing here" check never fired; mean IoU on boxed 0.35-0.36.
- dev (calib300 + test300): VLM alone gIoU +1.34 / cIoU -3.24 vs current fix +1.82 / +0.34. Agreement gating
  (use the fallback only if OWLv2 and VLM masks overlap >= 0.3-0.7) is worse in every setting (+0.5..+1.2 gIoU, cIoU
  negative). Not taken to val-rest. (Ran under heavy memory contention; speed only, not results, affected.)

### F5. Two name models, pick the name whose OWLv2 box scores higher (7B and 3B names) - **no gain**
- dev (calib300 + test300), tau 0.25: +1.82 [+0.78, +2.95] gIoU / +0.31 cIoU (3B name chosen 5 times) vs P2 +1.82 /
  +0.34. tau 0.3 / 0.35: +1.66 / +1.67. Not taken to val-rest.

### F6. Larger fallback models (7B names, refusals of all three validation sets)
- SAM ViT-L instead of ViT-B (same OWLv2-B boxes): dev tau 0.25 +1.82 / +0.35 (P2 +1.82 / +0.34); per-image masks
  change slightly both ways. Mask quality is not the bottleneck: routed failures are wrong names / wrong boxes (IoU 0).
- OWLv2-L instead of OWLv2-B (SAM-B, 7B names), tau re-chosen on dev by the same rule (max gIoU with dcIoU >= 0):
  tau 0.3, dev +1.88 / +0.71 vs P2 +1.82 / +0.34 (+0.06 gIoU = less than one image's worth). Confirmed on val-rest as
  the rule requires: **+1.55 [+0.42, +2.76] / +0.58 vs P2 +1.62 [+0.75, +2.66] / +1.16 - slightly worse.** SAM-L + OWLv2-L
  not run (SAM-L alone = SAM-B). OWLv2-L also runs ~5x slower (CPU, float64).

### F7. Conclusion of the final best shot
No variant beats P2, so P2 stays the final configuration. While auditing we also fixed a few claims: the
reproduction check is the official test split (not the 300-image subset), the gradient through the original
connector design is practically zero (about 15,000x smaller, checked numerically) rather than exactly zero, and
the example figure uses validation images, not test images. `exploration/scripts/score_p2.py` regenerates
`p2_results.json` with 0 differences.
