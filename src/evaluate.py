"""Score a VLM run JSONL: accuracy, hallucination, IoU, latency, cost.

    python -m src.evaluate --run results/runs/gpt-4o-mini_v1.jsonl \
        --out results/comparison.md

Metrics (per the project plan):
- accuracy / precision / recall / F1 on image-level has_defect
- hallucination rate: good images the model called defective / all good
- localisation IoU: predicted box vs GT mask bbox, mean over defective
  images with a parsed box; IoU >= 0.5 counts as a hit. A missing or
  unparsable box scores 0 for that image.
- latency: mean seconds per image; cost: total USD and per 1,000 images.
"""

import argparse
import json
from pathlib import Path


def iou(a: list, b: list) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def score(rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    parse_fail = 0
    ious = []
    hits = 0
    latencies = []
    total_cost = 0.0
    for r in rows:
        parsed = r.get("parsed")
        if not isinstance(parsed, dict) or "has_defect" not in parsed:
            parse_fail += 1
            pred = False  # unparsable reply counts as "no defect found"
        else:
            pred = bool(parsed["has_defect"])
        gt = bool(r["has_defect"])
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and not gt:
            tn += 1
        else:
            fn += 1
        if gt:
            box = parsed.get("box") if isinstance(parsed, dict) else None
            if isinstance(box, list) and len(box) == 4 and r.get("gt_bbox"):
                v = iou([float(x) for x in box], [float(x) for x in r["gt_bbox"]])
            else:
                v = 0.0
            ious.append(v)
            hits += 1 if v >= 0.5 else 0
        if r.get("latency_s") is not None:
            latencies.append(r["latency_s"])
        total_cost += r.get("cost_usd") or 0.0

    n = len(rows)
    good = tn + fp
    defective = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
              if precision + recall else 0.0,
        "hallucination_rate": fp / good if good else 0.0,
        "miss_rate": fn / defective if defective else 0.0,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "iou_hit_rate": hits / len(ious) if ious else 0.0,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "total_cost_usd": total_cost,
        "cost_per_1000_usd": total_cost / n * 1000 if n else 0.0,
        "parse_failures": parse_fail,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def fmt_row(name: str, s: dict) -> str:
    return (f"| {name} | {s['n']} | {s['accuracy']:.3f} | "
            f"{s['hallucination_rate']:.3f} | {s['mean_iou']:.3f} | "
            f"{s['iou_hit_rate']:.3f} | {s['mean_latency_s']:.2f} | "
            f"${s['cost_per_1000_usd']:.2f} |")


HEADER = ("| run | n | accuracy | hallucination | mean IoU | IoU>=0.5 | "
          "latency (s) | cost / 1k |\n"
          "|---|---|---|---|---|---|---|---|\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="results/runs/*.jsonl")
    parser.add_argument("--out", default="results/comparison.md")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.run) if line.strip()]
    label = args.label or Path(args.run).stem
    overall = score(rows)

    lines = [f"# Comparison\n\n{HEADER}{fmt_row(label, overall)}\n"]
    lines.append("## Per category\n\n" + HEADER)
    for cat in sorted({r["category"] for r in rows}):
        lines.append(fmt_row(cat, score([r for r in rows if r["category"] == cat])))
    lines.append(f"\nOverall: TP={overall['tp']} FP={overall['fp']} "
                 f"TN={overall['tn']} FN={overall['fn']}, "
                 f"parse failures={overall['parse_failures']}, "
                 f"precision={overall['precision']:.3f}, "
                 f"recall={overall['recall']:.3f}, F1={overall['f1']:.3f}, "
                 f"miss rate={overall['miss_rate']:.3f}, "
                 f"total cost=${overall['total_cost_usd']:.4f}.\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
