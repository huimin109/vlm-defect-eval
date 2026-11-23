"""Few-shot VLM runner: normal reference images + one test image per call.

Same evaluator contract as src/run_vlm_batch.py (one JSONL row per test
image, same fields), but each API call attaches K normal train images of
the same category BEFORE the test image, so the model can compare against
what "normal" looks like. The prompt (prompts/defect_fewshot.txt) tells
the model that only the LAST image is to be judged and that box
coordinates refer to that image. Keeping the v1 JSON output format
isolates the few-shot effect from prompt-wording effects.

    export OPENAI_API_KEY="sk-..."   # never put the key in the code
    python -m src.run_vlm_fewshot --manifest results/manifest.jsonl \
        --data-root data --category cable --num-refs 3 \
        --out results/runs/gpt-4o-mini_fewshot_cable.jsonl --dry-run
    python -m src.run_vlm_fewshot --manifest results/manifest.jsonl \
        --data-root data --category cable --num-refs 3 \
        --out results/runs/gpt-4o-mini_fewshot_cable.jsonl

Reference images: the first K files (sorted) in
<data-root>/<category>/train/good/. If your data root nests the dataset
one level deeper (e.g. data/mvtec_ad/cable/...), pass that folder as
--data-root instead. The runner fails loudly if no train/good folder is
found, rather than silently running zero-shot.

Cost: each call carries K+1 images, so expect roughly (K+1)x the
single-image cost. Measured single-image cost was $0.00386 (2026-11-12);
cable-only (150 images, K=3) was estimated at ~$1.70. Recheck the OpenAI
pricing page before a big run. Token usage and cost are recorded per row
from the API response, so the real total is in the output file.

Resume-safe: rows already in --out (by image path) are skipped.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = ROOT / "prompts" / "defect_fewshot.txt"

INPUT_PRICE_PER_M = 0.15   # USD per 1M input tokens, gpt-4o-mini
OUTPUT_PRICE_PER_M = 0.60  # USD per 1M output tokens, gpt-4o-mini
MAX_SIDE = 1024
SINGLE_IMAGE_COST = 0.00386  # measured 2026-11-12, for estimates only


def encode_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def find_refs(data_root: Path, category: str, num_refs: int) -> list[Path]:
    """First `num_refs` normal train images for a category (sorted)."""
    candidates = [
        data_root / category / "train" / "good",
        data_root / "mvtec_ad" / category / "train" / "good",
    ]
    for folder in candidates:
        if folder.is_dir():
            refs = sorted(folder.glob("*.png"))[:num_refs]
            if len(refs) < num_refs:
                raise FileNotFoundError(
                    f"Only {len(refs)} reference images in {folder}, "
                    f"need {num_refs}.")
            return refs
    raise FileNotFoundError(
        f"No train/good folder for {category} under {data_root}. "
        f"Checked: {[str(c) for c in candidates]}. "
        "Pass the folder that directly contains <category>/train/good "
        "as --data-root.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/manifest.jsonl")
    parser.add_argument("--out",
                        default="results/runs/gpt-4o-mini_fewshot_cable.jsonl")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    parser.add_argument("--data-root", default="data",
                        help="Folder containing <category>/train/good")
    parser.add_argument("--category", default="cable",
                        help="Only run this category (cable-only default)")
    parser.add_argument("--num-refs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = whole (filtered) manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would run, make no API calls")
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text().strip()
    rows = [json.loads(line) for line in open(args.manifest) if line.strip()]
    rows = [r for r in rows if r["category"] == args.category]
    if args.limit:
        rows = rows[:args.limit]

    refs = find_refs(Path(args.data_root), args.category, args.num_refs)
    ref_b64 = [encode_image(str(p)) for p in refs]
    print(f"Category: {args.category} | test images: {len(rows)} | "
          f"refs per call: {args.num_refs}")
    for p in refs:
        print(f"  ref: {p}")

    if args.dry_run:
        est = len(rows) * SINGLE_IMAGE_COST * (args.num_refs + 1)
        print(f"{len(rows)} images would run. Rough estimate at "
              f"${SINGLE_IMAGE_COST}/image x {args.num_refs + 1} images "
              f"per call: ~${est:.2f} (estimate only; real cost is "
              "recorded per row).")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY first (do not put the key in the code).")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        done = {json.loads(l)["image"] for l in open(out) if l.strip()}
    print(f"Resuming: {len(done)} already done, {len(rows)} in manifest.")

    total_cost = 0.0
    with open(out, "a") as f:
        for i, row in enumerate(rows, 1):
            if row["image"] in done:
                continue
            content = [{"type": "text", "text": prompt}]
            for b64 in ref_b64:
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"}})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,"
                       f"{encode_image(row['image'])}"}})
            t0 = time.time()
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": content}],
            )
            latency = time.time() - t0
            raw = response.choices[0].message.content or ""
            usage = response.usage
            cost = 0.0
            if usage:
                cost = (usage.prompt_tokens * INPUT_PRICE_PER_M
                        + usage.completion_tokens * OUTPUT_PRICE_PER_M) / 1_000_000
            total_cost += cost
            record = {
                **row,
                "model": args.model,
                "prompt_version": Path(args.prompt_file).stem,
                "num_refs": args.num_refs,
                "ref_images": [str(p) for p in refs],
                "raw": raw,
                "parsed": parse_json(raw),
                "latency_s": round(latency, 2),
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "cost_usd": round(cost, 6),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"[{i}/{len(rows)}] {row['category']}/{row['defect_type']} "
                  f"{latency:.1f}s ${cost:.5f} "
                  f"parsed={'yes' if record['parsed'] else 'NO'}")
    print(f"Done. This run added ~${total_cost:.4f}. Results: {out}")
    print("Next: score with the same evaluator, e.g. "
          "python -m src.evaluate --help")


if __name__ == "__main__":
    main()
