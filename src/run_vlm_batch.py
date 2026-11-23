"""Batch VLM runner: MVTec AD test images -> one JSONL row per image.

Batch version of vlm_minimal_example.py (same provider, same pricing note):
asks GPT-4o-mini the fixed prompt in prompts/defect_v1.txt and records the
raw reply, the parsed JSON, latency, token usage and estimated cost.

    export OPENAI_API_KEY="sk-..."   # never put the key in the code
    python -m src.mvtec --root data/mvtec_ad --categories bottle carpet cable
    python -m src.run_vlm_batch --manifest results/manifest.jsonl \
        --out results/runs/gpt-4o-mini_v1.jsonl --limit 20

Pricing (measured 2026-11-12, recheck the OpenAI pricing page before a big run):
gpt-4o-mini input $0.15 / 1M tokens, output $0.60 / 1M tokens.
One 1024px image costs ~$0.00386 in practice, so the ~350-image starter
trio costs ~$1.35. Images are downscaled to max 1024px on the long side
before upload to keep that cost; bboxes stay in 0-1000 normalised
coordinates, so downscaling does not change them.

Resume-safe: rows already in --out (by image path) are skipped, so a
crashed run can be re-run with the same command.
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
DEFAULT_PROMPT = ROOT / "prompts" / "defect_v1.txt"

INPUT_PRICE_PER_M = 0.15   # USD per 1M input tokens, gpt-4o-mini
OUTPUT_PRICE_PER_M = 0.60  # USD per 1M output tokens, gpt-4o-mini
MAX_SIDE = 1024


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/manifest.jsonl")
    parser.add_argument("--out", default="results/runs/gpt-4o-mini_v1.jsonl")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    parser.add_argument("--limit", type=int, default=0, help="0 = whole manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would run, make no API calls")
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text().strip()
    rows = [json.loads(line) for line in open(args.manifest) if line.strip()]
    if args.limit:
        rows = rows[:args.limit]

    if args.dry_run:
        for row in rows:
            print(f"{row['category']:10s} {row['defect_type']:20s} {row['image']}")
        print(f"{len(rows)} images would run. Estimated cost at "
              f"$0.00386/image: ~${len(rows) * 0.00386:.2f}")
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
            t0 = time.time()
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,"
                               f"{encode_image(row['image'])}"}},
                ]}],
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


if __name__ == "__main__":
    main()
