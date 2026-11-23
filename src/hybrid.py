"""Hybrid evaluation: PatchCore decides, VLM is the second opinion.

PatchCore's errors sit on the threshold edge, where its score is close
to the per-category threshold and good/defective images are interleaved.
For those ambiguous images, let the VLM decide instead; everywhere
else, keep PatchCore's call (and its box).

    python -m src.hybrid --patchcore results/runs/patchcore_v2.jsonl \
        --vlm results/runs/gpt-4o-mini_v3.jsonl

The PatchCore run must record `score` and `threshold` per image
(src/run_patchcore.py writes them). No API calls are made here: the
VLM judgments come from an already-finished run file, so this
evaluation is free and offline. Try several bands and several VLM
arms (v1 = conservative judge, v2 = aggressive judge, v3 = balanced)
and compare.

Metrics match src/evaluate.py: accuracy, hallucination rate
(FP / good images), mean IoU of the deciding arm's box vs the
ground-truth mask bbox, precision / recall / F1.
"""

import argparse
import json


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (max(0, a[2] - a[0]) * max(0, a[3] - a[1])
             + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter)
    return inter / union if union else 0.0


def score(rows):
    tp = fp = tn = fn = 0
    ious = []
    for r in rows:
        pred = r["pred"]
        gt = r["has_defect"]
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and not gt:
            tn += 1
        else:
            fn += 1
        if gt:
            box = r.get("box")
            if isinstance(box, list) and len(box) == 4 and r.get("gt_bbox"):
                ious.append(iou([float(x) for x in box],
                                [float(x) for x in r["gt_bbox"]]))
            else:
                ious.append(0.0)
    n = len(rows)
    good = tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "n": n, "accuracy": (tp + tn) / n if n else 0.0,
        "hallucination": fp / good if good else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patchcore",
                        default="results/runs/patchcore_v2.jsonl")
    parser.add_argument("--vlm",
                        default="results/runs/gpt-4o-mini_v3.jsonl")
    parser.add_argument("--bands", type=float, nargs="+",
                        default=[0.03, 0.05, 0.08, 0.12],
                        help="|score - threshold| <= band -> VLM decides")
    args = parser.parse_args()

    pc = {json.loads(l)["image"]: json.loads(l)
          for l in open(args.patchcore) if l.strip()}
    vl = {json.loads(l)["image"]: json.loads(l)
          for l in open(args.vlm) if l.strip()}
    common = sorted(set(pc) & set(vl))
    print(f"{len(common)} images in both runs "
          f"({len(pc) - len(common)} PatchCore-only, "
          f"{len(vl) - len(common)} VLM-only).")

    def call(row):
        parsed = row.get("parsed") or {}
        return bool(parsed.get("has_defect")), parsed.get("box")

    base = []
    for img in common:
        p, _ = call(pc[img])
        base.append({**pc[img], "pred": p, "box": pc[img]["parsed"].get("box")})
    s = score(base)
    print(f"PatchCore alone: acc {s['accuracy']:.3f} hall {s['hallucination']:.3f} "
          f"recall {s['recall']:.3f} F1 {s['f1']:.3f} IoU {s['mean_iou']:.3f}")

    for band in args.bands:
        rows = []
        n_called = 0
        for img in common:
            pc_row, vl_row = pc[img], vl[img]
            p_pred, p_box = call(pc_row)
            v_pred, v_box = call(vl_row)
            if abs(pc_row["score"] - pc_row["threshold"]) <= band:
                pred, box = v_pred, v_box
                n_called += 1
            else:
                pred, box = p_pred, p_box
            rows.append({**pc_row, "pred": pred, "box": box})
        s = score(rows)
        print(f"band +-{band:.2f} (VLM called on {n_called}): "
              f"acc {s['accuracy']:.3f} hall {s['hallucination']:.3f} "
              f"recall {s['recall']:.3f} F1 {s['f1']:.3f} "
              f"IoU {s['mean_iou']:.3f} "
              f"TP={s['tp']} FP={s['fp']} TN={s['tn']} FN={s['fn']}")


if __name__ == "__main__":
    main()
