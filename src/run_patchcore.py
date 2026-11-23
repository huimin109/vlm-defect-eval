"""PatchCore-style unsupervised baseline for MVTec AD.

This is the fair baseline for the VLM comparison (see README.md):
MVTec AD's train split contains normal images only, so a supervised
YOLO-seg model trained on defect masks is NOT a fair comparison.
PatchCore trains only on normal images, exactly like the VLM arm
which sees no defect labels either.

Method (Bergmann et al., PatchCore, CVPR 2022, core idea):
pretrained WideResNet50 features -> patch-level memory bank built
from normal training images -> anomaly score = distance to the
nearest normal patch. The reference implementation is `anomalib`
(`pip install anomalib`); this script implements the same core
directly in torch/torchvision so the baseline runs with the same
requirements as the rest of the repo, with no anomalib API to chase.

Output is the same JSONL format as src/run_vlm_batch.py, so the
same src/evaluate.py scores it:

    python -m src.run_patchcore --root data/mvtec_ad \
        --categories bottle carpet cable \
        --out results/runs/patchcore_v1.jsonl

Not run on real data yet: the dataset has not been downloaded.
Threshold note: the image-level threshold is calibrated per
category from the normal training images (95th percentile of
their max patch distances). Tune it on train scores only, never
on the test set, or the comparison is no longer honest.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

from src.mvtec import load_manifest

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
# Patch features: layer2 (28x28) + layer3 pooled to 28x28 -> 512 + 1024 dims.
BANK_SAMPLES = 2000


def build_extractor():
    weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
    backbone = models.wide_resnet50_2(weights=weights)
    backbone.eval().to(DEVICE)
    feats = {}

    def hook(name):
        def _hook(_m, _i, out):
            feats[name] = out
        return _hook

    backbone.layer2.register_forward_hook(hook("layer2"))
    backbone.layer3.register_forward_hook(hook("layer3"))
    # NOTE: do NOT use weights.transforms() here: it centre-crops to
    # 224x224, which would cut off defects near the image edges and
    # make the 0-1000 bbox coordinates wrong. Resize the whole image
    # instead so the bbox still maps to the full image.
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return backbone, feats, preprocess


@torch.no_grad()
def extract_patches(backbone, feats, preprocess, image_path):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    _ = backbone(x)
    l2 = feats["layer2"]  # [1, 512, 28, 28]
    l3 = feats["layer3"]  # [1, 1024, 14, 14]
    l3_up = torch.nn.functional.interpolate(
        l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3_up], dim=1)  # [1, 1536, 28, 28]
    f = torch.nn.functional.normalize(f, dim=1)
    return f[0].permute(1, 2, 0).reshape(-1, f.shape[1])  # [784, 1536]


def distance_map(patches, bank):
    # patches [P, D], bank [B, D] -> min distance per patch, [28, 28]
    dists = torch.cdist(patches, bank)  # [P, B]
    return dists.min(dim=1).values.reshape(28, 28)


def map_to_bbox(dist_map, threshold):
    """BBox of the above-threshold region, in 0-1000 coordinates."""
    mask = dist_map > threshold
    if not mask.any():
        return None
    ys, xs = torch.nonzero(mask, as_tuple=True)
    return [
        int(xs.min().item() / 28 * 1000),
        int(ys.min().item() / 28 * 1000),
        int(xs.max().item() / 28 * 1000),
        int(ys.max().item() / 28 * 1000),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/mvtec_ad")
    parser.add_argument("--categories", nargs="+",
                        default=["bottle", "carpet", "cable"])
    parser.add_argument("--out",
                        default="results/runs/patchcore_v1.jsonl")
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--train-images", type=int, default=50,
                        help="Normal train images per category for the bank")
    args = parser.parse_args()

    backbone, feats, preprocess = build_extractor()
    root = Path(args.root)
    rows = load_manifest(root, args.categories)
    if args.limit_per_category:
        kept = []
        for cat in args.categories:
            kept += [r for r in rows if r["category"] == cat][:args.limit_per_category]
        rows = kept

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    with open(out, "w") as f:
        for cat in args.categories:
            train_dir = root / cat / "train" / "good"
            train_imgs = sorted(train_dir.glob("*.png"))[:args.train_images]
            if not train_imgs:
                raise FileNotFoundError(f"No train images: {train_dir}")
            print(f"{cat}: building memory bank from {len(train_imgs)} "
                  f"normal images...")
            bank_patches = torch.cat(
                [extract_patches(backbone, feats, preprocess, p)
                 for p in train_imgs]).cpu()
            idx = torch.randperm(len(bank_patches))[:BANK_SAMPLES]
            bank = bank_patches[idx].to(DEVICE)

            # Calibrate the image-level threshold on normal train images.
            train_scores = []
            for p in train_imgs:
                patches = extract_patches(backbone, feats, preprocess, p)
                train_scores.append(distance_map(patches, bank).max().item())
            train_scores = torch.tensor(train_scores)
            threshold = float(torch.quantile(train_scores, 0.95))
            print(f"{cat}: threshold (95th pct of train max distance) "
                  f"= {threshold:.3f}")

            cat_rows = [r for r in rows if r["category"] == cat]
            for i, row in enumerate(cat_rows, 1):
                t0 = time.time()
                patches = extract_patches(backbone, feats, preprocess,
                                          row["image"])
                dmap = distance_map(patches, bank)
                score = float(dmap.max().item())
                pred_defect = score > threshold
                bbox = (map_to_bbox(dmap, threshold)
                        if pred_defect else None)
                record = {
                    **row,
                    "model": "patchcore",
                    "prompt_version": Path(args.out).stem,
                    "score": round(score, 4),
                    "threshold": round(threshold, 4),
                    "raw": f"anomaly_score={score:.4f} "
                           f"threshold={threshold:.4f}",
                    "parsed": {"has_defect": pred_defect, "box": bbox},
                    "latency_s": round(time.time() - t0, 2),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "cost_usd": 0.0,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                if i % 20 == 0 or i == len(cat_rows):
                    print(f"  [{i}/{len(cat_rows)}] {row['category']} "
                          f"score={score:.3f} pred={pred_defect}")
    print(f"Done. Results: {out}")


if __name__ == "__main__":
    main()
