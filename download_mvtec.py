"""Download / unpack MVTec AD and verify it.

MVTec AD is NOT a direct download: the official site asks you to fill in a
form first, then emails / shows a download link for `mvtec_ad.tar.xz`.

    1. Go to https://www.mvtec.com/company/research/datasets/mvtec-ad
       and fill in the form (research use only).
    2. Save the downloaded file, then run:
           python download_mvtec.py --tar ~/Downloads/mvtec_ad.tar.xz
       or, if you already extracted it by hand:
           python download_mvtec.py --extracted-dir data/mvtec_ad

The script extracts to data/mvtec_ad/ and prints the test-image count per
category, so you can check the download is complete before running anything.

Expected layout after extraction (verified against the MVTec AD format):
    data/mvtec_ad/<category>/{train,test,ground_truth}/
      train/good/*.png                 (normal images only)
      test/good/*.png                  (normal test images)
      test/<defect_type>/*.png         (defective test images)
      ground_truth/<defect_type>/*_mask.png
15 categories: bottle, cable, capsule, carpet, grid, hazelnut, leather,
metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper.
"""

import argparse
import sys
import tarfile
from pathlib import Path

EXPECTED_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mvtec_ad"


def count_test_images(root: Path) -> dict[str, int]:
    counts = {}
    for cat in EXPECTED_CATEGORIES:
        test_dir = root / cat / "test"
        counts[cat] = len(list(test_dir.rglob("*.png"))) if test_dir.is_dir() else 0
    return counts


def verify(root: Path) -> bool:
    counts = count_test_images(root)
    missing = [c for c, n in counts.items() if n == 0]
    total = sum(counts.values())
    print(f"Root: {root}")
    for cat, n in counts.items():
        print(f"  {cat:12s} test images: {n}")
    print(f"Total test images: {total} (full MVTec AD test set is 1,725)")
    if missing:
        print(f"MISSING categories (0 test images): {', '.join(missing)}")
        return False
    if total != 1725:
        print("WARNING: total is not 1,725 — the download may be incomplete.")
    else:
        print("OK: all 15 categories present, test total matches 1,725.")
    return not missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", help="Path to the downloaded mvtec_ad.tar.xz")
    parser.add_argument("--extracted-dir",
                        help="Path to an already-extracted mvtec_ad folder")
    parser.add_argument("--dest", default=str(DATA_DIR),
                        help="Where to extract to (default: data/mvtec_ad)")
    args = parser.parse_args()

    if args.extracted_dir:
        root = Path(args.extracted_dir)
        if not verify(root):
            sys.exit(1)
        return

    if not args.tar:
        print("Give --tar <path to mvtec_ad.tar.xz> or --extracted-dir <folder>.")
        print("Download first at https://www.mvtec.com/company/research/"
              "datasets/mvtec-ad (form required).")
        sys.exit(2)

    tar_path = Path(args.tar)
    if not tar_path.is_file():
        print(f"Not found: {tar_path}")
        sys.exit(1)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path} -> {dest} (this takes a while)...")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dest)
    # The tarball may contain a top-level mvtec_ad/ folder; normalise to dest.
    nested = dest / "mvtec_ad"
    root = nested if nested.is_dir() and (nested / "bottle").is_dir() else dest
    if not verify(root):
        sys.exit(1)


if __name__ == "__main__":
    main()
