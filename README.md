# VLM Defect Detection: an Evaluation Harness on MVTec AD

Can a vision-language model replace a classic unsupervised anomaly detector for
industrial defect detection? This repo tests that question head-to-head on
MVTec AD: **GPT-4o-mini** (a VLM, zero-shot, JSON output) vs **PatchCore**
(the standard unsupervised baseline), on the same 350 test images, scored
with the same evaluator on detection accuracy, hallucination rate,
localization IoU, latency, and cost.

**Short answer: no — not with a plain zero-shot prompt.** With the plain
prompt (v1) the VLM only wins on hallucination rate: it is far more
conservative than PatchCore, misses most cable defects entirely, and its
boxes are nearly useless for localization. A revised prompt (v2) that
asks the model to describe what it sees first and to flag a defect when
in doubt flips the operating point — recall jumps from 0.594 to 0.930 —
but hallucination explodes from 0.038 to 0.604. A middle-ground prompt
(v3) that keeps the describe-first step but requires the model to point
to something specific lands between them — recall 0.676, hallucination
0.142. A hybrid that lets PatchCore decide and calls the VLM only on
images whose PatchCore score sits near the threshold ties PatchCore on
accuracy (0.923 vs 0.920) while cutting hallucination in half
(0.057 vs 0.132) — an operating-point change, not a win. Prompting
moves the trade-off; it does not fix the capability gap. The failure
analysis below is the interesting part.

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
(CC BY-NC-SA 4.0, research use only). The train split is normal images only,
so this is an unsupervised setting — the same setting for both arms.

Three starter categories, 350 test images total:

| Category | Test images | Defective | Good |
|---|---|---|---|
| bottle | 83 | 63 | 20 |
| carpet | 117 | 89 | 28 |
| cable | 150 | 92 | 58 |
| **Total** | **350** | **244** | **106** |

Bottle = clear object defects, carpet = texture, cable = complex structure.
The other 12 MVTec categories were removed from `data/` to keep this run
focused (they can be re-downloaded).

## Methods

**PatchCore (`src/run_patchcore.py`)** — implemented directly in
torch/torchvision (not via the anomalib API). Frozen ImageNet-pretrained
WideResNet50, layer2+layer3 patch features, a memory bank of normal patch
features (coreset-subsampled to 2,000), image score = max patch distance to
the nearest normal patch. Per-category threshold = 95th percentile of normal
train scores (bottle 0.672, carpet 0.599, cable 0.768). Whole image resized
to 224×224 (no center crop, so edge defects survive). Runs locally, $0.

**GPT-4o-mini (`src/run_vlm_batch.py`, prompt v1 in
`prompts/defect_v1.txt`)** — zero-shot, one image per API call, downscaled
to max 1024px. Fixed prompt, JSON only:

> Is there a defect in this image? If yes, return JSON only, no other text:
> `{"has_defect": true, "box": [x_min, y_min, x_max, y_max], "type": "...", "severity": "minor|moderate|severe"}`
> Coordinates are 0-1000, origin top-left. If no defect, return JSON only:
> `{"has_defect": false}`

**Prompt v2 (`prompts/defect_v2.txt`)** — same model, three changes:
(1) the model must first describe what it sees in one sentence
(recorded in an `observation` field) — for cables, the number, order,
bends, and insulation condition of the wires; (2) subtle structural
defects are explicitly called out (a bent, missing, or swapped wire
counts as a defect); (3) when unsure, the model is told to say there is
a defect rather than none — missing a defect is worse than a false
alarm.

**Prompt v3 (`prompts/defect_v3.txt`)** — keeps the describe-first
step and the cable checklist from v2, but drops the "flag when in
doubt" instruction. Instead: only call it a defect if you can point to
something specific that is wrong; do not guess, and do not dismiss
something you can see.

Parse failures: 0 / 350 for all three prompts. Both arms are scored by the same `src/evaluate.py`:
accuracy, hallucination rate (FP / good images), mean IoU of predicted box
vs ground-truth mask bbox, IoU ≥ 0.5 hit rate, latency, cost per 1,000 images.

**Hybrid (`src/hybrid.py`)** — PatchCore decides; the VLM is the
second opinion only where PatchCore is unsure, i.e. where
`|score − threshold| ≤ band`. The PatchCore run must record `score`
and `threshold` per image (`src/run_patchcore.py` writes them).
No new API calls: the VLM judgments come from an already-finished run
file, so this evaluation is free and offline. Bands tried: ±0.03,
±0.05, ±0.08, ±0.12, each with v1 (conservative judge), v2
(aggressive judge), and v3 (balanced judge).

## Results (2026-11-12, full 350 images)

| run | n | accuracy | hallucination | mean IoU | IoU ≥ 0.5 | latency (s) | cost / 1k |
|---|---|---|---|---|---|---|---|
| PatchCore | 350 | **0.920** | 0.132 | **0.506** | **0.590** | **0.06** | **$0.00** |
| gpt-4o-mini v1 | 350 | 0.706 | **0.038** | 0.054 | 0.029 | 1.80 | $3.85 |
| gpt-4o-mini v2 | 350 | 0.769 | 0.604 | 0.136 | 0.037 | 2.63 | $3.90 |
| gpt-4o-mini v3 | 350 | 0.731 | 0.142 | 0.084 | 0.025 | 2.57 | $3.89 |

PatchCore: TP=230, FP=14, TN=92, FN=14, precision=recall=F1=0.943,
miss rate 0.057. GPT-4o-mini v1: TP=145, FP=4, TN=102, FN=99,
precision=0.973, recall=0.594, F1=0.738, miss rate 0.406.
GPT-4o-mini v2: TP=227, FP=64, TN=42, FN=17, precision=0.780,
recall=0.930, F1=0.849, miss rate 0.070.
GPT-4o-mini v3: TP=165, FP=15, TN=91, FN=79, precision=0.917,
recall=0.676, F1=0.778, miss rate 0.324.
Total VLM cost: v1 **$1.3478**, v2 **$1.3634**, v3 **$1.3624**
(measured ~$0.00386/image).

### Hybrid: PatchCore decides, VLM second opinion near the threshold

| run | VLM called on | accuracy | hallucination | recall | F1 | mean IoU |
|---|---|---|---|---|---|---|
| PatchCore alone | 0 | 0.920 | 0.132 | 0.943 | 0.943 | 0.506 |
| hybrid, v1 judge, band ±0.03 | 59 | **0.923** | **0.057** | 0.914 | 0.943 | 0.503 |
| hybrid, v3 judge, band ±0.03 | 59 | **0.923** | 0.066 | 0.918 | 0.943 | 0.506 |
| hybrid, v2 judge, band ±0.03 | 59 | 0.906 | 0.217 | **0.959** | 0.934 | 0.512 |
| hybrid, v1 judge, band ±0.05 | 109 | 0.889 | 0.047 | 0.861 | 0.915 | 0.485 |

Best hybrid (v1 judge, ±0.03): TP=223, FP=6, TN=100, FN=21.
Wider bands are worse for every judge; the narrowest band is best.

### Per category

| run | n | accuracy | hallucination | mean IoU | IoU ≥ 0.5 |
|---|---|---|---|---|---|
| PatchCore — bottle | 83 | 0.976 | 0.100 | 0.653 | 0.794 |
| PatchCore — carpet | 117 | 0.966 | 0.071 | 0.549 | 0.663 |
| PatchCore — cable | 150 | 0.853 | 0.172 | 0.363 | 0.380 |
| gpt-4o-mini — bottle | 83 | 0.807 | 0.200 | 0.148 | 0.095 |
| gpt-4o-mini — carpet | 117 | 0.949 | 0.000 | 0.019 | 0.000 |
| gpt-4o-mini — cable | 150 | 0.460 | 0.000 | 0.024 | 0.011 |
| gpt-4o-mini v2 — bottle | 83 | 0.771 | 0.950 | 0.197 | 0.016 |
| gpt-4o-mini v2 — carpet | 117 | 0.889 | 0.464 | 0.044 | 0.000 |
| gpt-4o-mini v2 — cable | 150 | 0.673 | 0.552 | 0.183 | 0.087 |
| gpt-4o-mini v3 — bottle | 83 | 0.783 | 0.550 | 0.175 | 0.032 |
| gpt-4o-mini v3 — carpet | 117 | 0.957 | 0.000 | 0.036 | 0.000 |
| gpt-4o-mini v3 — cable | 150 | 0.527 | 0.069 | 0.067 | 0.043 |

## Failure analysis

**PatchCore's errors sit on the threshold edge.** Its 14 false negatives
score 0.728–0.763 against the cable threshold of 0.768, and its 14 false
positives score 0.768–0.795 — good and defective cable images are
interleaved in the same score band, so no threshold separates them.
Threshold tuning moves along the FP/FN trade-off; it cannot fix cable.
(Galleries: `results/fn_gallery.png`, `results/fp_gallery.png`.)

**The VLM's errors are structural, not threshold noise.** Of its 99 false
negatives, 81 are cable. Recall by defect type:

- cable: bent_wire 0/13, cable_swap **0/12**, missing_cable 0/12,
  missing_wire 0/10, poke_insulation 0/10, cut_inner_insulation 4/14,
  combined 2/11, cut_outer_insulation 5/10
- bottle: broken_large 20/20, broken_small 22/22, contamination 9/21
- carpet: cut 17/17, hole 17/17, metal_contamination 17/17,
  color 18/19, thread 14/19

The VLM sees obvious, visible defects perfectly and is blind to structural
ones. The original hypothesis — that the VLM would win on cable_swap,
a logical anomaly (wires swapped, locally normal) that patch-wise PatchCore
cannot see — is **rejected**: the VLM caught 0 of 12 cable_swap images,
worse than PatchCore's 3 of 12. Without a reference for the correct wire
arrangement, the small model defaults to "no defect" when uncertain.

**Prompt v2 moves the operating point; it does not fix the model.**
Asking the model to describe the image first and to flag a defect when
in doubt raised recall from 0.594 to 0.930 (FN 99 → 17) and cable
accuracy from 0.460 to 0.673 (cable recall roughly 12% → 82%). Mean IoU
also improved, 0.054 → 0.136, on every category. But the price was the
hallucination advantage v1 had: hallucination rate 0.038 → 0.604
(FP 4 → 64 of 106 good images). Bottle is the worst case — 19 of its 20
good images were flagged; carpet 13 of 28. v1 was too conservative and
v2 is too trigger-happy; the sweet spot is between them. The lesson is
that a prompt can slide the FP/FN trade-off, like a threshold, but it
cannot raise the underlying capability ceiling: even at its best
operating point the VLM's accuracy (0.769) and localization (0.136)
remain far below PatchCore's (0.920 / 0.506).

**The three prompts trace one trade-off curve, and the best F1 is not
in the middle.** v1 (recall 0.594, hallucination 0.038, F1 0.738) →
v3 (recall 0.676, hallucination 0.142, F1 0.778) → v2 (recall 0.930,
hallucination 0.604, F1 0.849). The describe-first step alone, without
the aggressive bias, buys only a modest gain over v1 — most of v2's
recall jump came from the "flag when in doubt" instruction itself. v3's
cable recall falls back to roughly 27% (from v2's ~82%): asking the
model to "point to something specific" is hard when the defect *is* the
arrangement, not a visible spot. Carpet is v3's best category (0.957
accuracy, 0.000 hallucination, recall ~0.944). The cleanest statement
of the gap: at v3's hallucination rate (0.142, essentially PatchCore's
0.132), PatchCore's recall is 0.943 against the VLM's 0.676.

**The hybrid ties; it does not win.** On PatchCore's 28 errors the
VLM (v1) would correct 16 — all 14 false positives and 2 false
negatives — which suggests an oracle hybrid of 338/350 = 0.966.
But the hybrid cannot know which images are errors: it hands the
whole ambiguous band to the VLM, including correctly classified images
near the threshold that the VLM flips the wrong way. At the best band
(±0.03, 59 images called) the VLM corrects 8 false positives and adds
7 new false negatives, for a net gain of one image: accuracy 0.920 →
0.923, F1 unchanged at 0.943. Wider bands only hand the VLM more
images to get wrong. The hybrid's real value is operating-point
control, not accuracy: with a v1 judge at ±0.03, hallucination is more
than halved (0.132 → 0.057) at the cost of recall (0.943 → 0.914) —
worth it where a false alarm is expensive, not where a miss is. And
cable_swap stays unfixable by this route: the VLM misses the same 9
cable_swap images PatchCore misses.

**Localization is the VLM's weakest axis.** Even where it detects the
defect, its boxes are coarse: mean IoU per defect type is 0.001–0.055 on
carpet and 0.264 at best (bottle broken_large). A VLM returns a box as
text tokens from a coarse patch grid — it describes a location, it does not
measure one. Its 4 false positives are all bottle good images, each boxed
as the whole image `[0, 0, 1000, 1000]` with an invented defect type
("scratches", "coating issue") — hallucination looks like a confident
whole-image box, not a small slip.

## Conclusion

For this task, a specialized unsupervised detector beats a general-purpose
VLM on every axis except hallucination rate, and costs nothing and runs
30× faster. The VLM's value is not detection or localization here; it is
judgment on visible defects and a low false-alarm rate. The honest
engineering conclusion is a hybrid: PatchCore (or a segmentation model)
for detection and precise localization, VLM as a second opinion or
explainer — not a replacement. Measured, that hybrid ties PatchCore on
accuracy (0.923 vs 0.920) with hallucination more than halved
(0.057 vs 0.132): worth it where a false alarm is expensive, not
where a miss is.

Next steps: the prompt curve is now complete and the hybrid is
measured, so the remaining experiments should change the information
the model gets, not the wording — a few-shot arm that shows one or two
normal cable images as a reference for the correct wire arrangement,
or a larger VLM to test whether the capability ceiling itself moves.

## Reproduce

```bash
cd vlm-defect-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt               # openai, pillow, numpy
pip install -r requirements-baseline.txt      # torch, torchvision (PatchCore)

# 1. Download MVTec AD (form at the link above), then:
python download_mvtec.py --tar ~/Downloads/mvtec_ad.tar.xz

# 2. Build the manifest (3 starter categories)
python -m src.mvtec --root data --categories bottle carpet cable \
    --out results/manifest.jsonl

# 3a. PatchCore baseline
python -m src.run_patchcore --root data --categories bottle carpet cable \
    --out results/runs/patchcore_v1.jsonl
python -m src.evaluate --run results/runs/patchcore_v1.jsonl \
    --out results/comparison_patchcore.md

# 3b. VLM arm (needs an OpenAI API key; API is prepaid, min. $5 credit)
export OPENAI_API_KEY="sk-..."   # env var only, never commit it
python -m src.run_vlm_batch --manifest results/manifest.jsonl \
    --out results/runs/gpt-4o-mini_v1.jsonl
python -m src.evaluate --run results/runs/gpt-4o-mini_v1.jsonl \
    --out results/comparison_gpt-4o-mini_v1.md

# 3c. VLM arm, prompt v2 (describe first, flag when in doubt)
python -m src.run_vlm_batch --manifest results/manifest.jsonl \
    --out results/runs/gpt-4o-mini_v2.jsonl \
    --prompt-file prompts/defect_v2.txt
python -m src.evaluate --run results/runs/gpt-4o-mini_v2.jsonl \
    --out results/comparison_gpt-4o-mini_v2.md

# 3d. VLM arm, prompt v3 (describe first, point to something specific)
python -m src.run_vlm_batch --manifest results/manifest.jsonl \
    --out results/runs/gpt-4o-mini_v3.jsonl \
    --prompt-file prompts/defect_v3.txt
python -m src.evaluate --run results/runs/gpt-4o-mini_v3.jsonl \
    --out results/comparison_gpt-4o-mini_v3.md

# 4. Hybrid: re-run PatchCore with scores saved, then evaluate offline
python -m src.run_patchcore --root data --categories bottle carpet cable \
    --out results/runs/patchcore_v2.jsonl
python -m src.hybrid --patchcore results/runs/patchcore_v2.jsonl \
    --vlm results/runs/gpt-4o-mini_v3.jsonl
```

Re-running `run_vlm_batch` resumes where it left off (completed images
are skipped). Costs measured 2026-11-12: GPT-4o-mini $0.00386/image
($1.35 for the full 350), ~1.8 s/image (v1) and ~2.6 s/image (v2/v3,
longer output).

## Notes

- Do not upload proprietary production images to any third-party API;
  this repo uses only the public MVTec AD set. Check your employer's
  policy before uploading anything else.
- `data/` and `results/` are gitignored; the dataset and run outputs do
  not go into git. MVTec AD itself is CC BY-NC-SA 4.0, research use only.
- A supervised YOLO-seg arm was deliberately **not** used as the main
  baseline: MVTec AD trains on normal images only, so a supervised model
  is not an apples-to-apples comparison. It could serve as a labeled
  upper-bound reference, clearly marked as such.
