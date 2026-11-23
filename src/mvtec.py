"""MVTec AD loader: test images + ground-truth mask -> bbox (0-1000 coords).

Layout (one folder per category):
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/<stem>_mask.png

Only the test split is used for evaluation. The train split contains
normal images only (`train/good/`), which is why a supervised YOLO-seg
baseline trained on defect masks is NOT a fair comparison here — see
README.md. The fair unsupervised baseline is PatchCore (anomalib).

Usage:
    python -m src.mvtec --root data/mvtec_ad \
        --categories bottle carpet cable --out results/manifest.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Starter trio from the project plan: object / texture / complex structure.
STARTER_CATEGORIES = ["bottle", "carpet", "cable"]

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


def mask_to_bbox(mask_path: Path) -> list[int] | None:
    """Bounding box of the defect mask, in 0-1000 normalised coordinates.

    Returns [x_min, y_min, x_max, y_max] or None if the mask is empty.
    """
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return [
        int(xs.min() / w * 1000),
        int(ys.min() / h * 1000),
        int(xs.max() / w * 1000),
        int(ys.max() / h * 1000),
    ]


def load_manifest(root: Path, categories: list[str]) -> list[dict]:
    """All test images for the given categories, with GT labels and bboxes."""
    rows = []
    for category in categories:
        test_dir = root / category / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(f"No test dir for {category}: {test_dir}")
        for img_path in sorted(test_dir.rglob("*.png")):
            defect_type = img_path.parent.name
            has_defect = defect_type != "good"
            mask_path = None
            gt_bbox = None
            if has_defect:
                candidate = (root / category / "ground_truth" / defect_type
                             / f"{img_path.stem}_mask.png")
                if candidate.is_file():
                    mask_path = str(candidate)
                    gt_bbox = mask_to_bbox(candidate)
            rows.append({
                "category": category,
                "image": str(img_path),
                "defect_type": defect_type,
                "has_defect": has_defect,
                "mask": mask_path,
                "gt_bbox": gt_bbox,  # [x_min, y_min, x_max, y_max] in 0-1000
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/mvtec_ad")
    parser.add_argument("--categories", nargs="+", default=STARTER_CATEGORIES)
    parser.add_argument("--out", default="results/manifest.jsonl")
    parser.add_argument("--limit-per-category", type=int, default=0,
                        help="0 = all test images in the category")
    args = parser.parse_args()

    rows = load_manifest(Path(args.root), args.categories)
    if args.limit_per_category:
        kept = []
        for cat in args.categories:
            kept += [r for r in rows if r["category"] == cat][:args.limit_per_category]
        rows = kept

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    for cat in args.categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        n_def = sum(r["has_defect"] for r in cat_rows)
        print(f"{cat:10s} {len(cat_rows):4d} test images "
              f"({n_def} defective, {len(cat_rows) - n_def} good)")
    print(f"Total: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
