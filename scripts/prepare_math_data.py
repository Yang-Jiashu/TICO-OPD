#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DATASETS = {
    "aime24": ("OpenRLHF/aime-2024", "train"),
    "aime25": ("MathArena/aime_2025", "train"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "dapo17k": ("BytedTsinghua-SIA/DAPO-Math-17k", "train"),
}

INPUT_KEYS = ["problem", "question", "prompt", "input", "query"]
ANSWER_KEYS = ["answer", "label", "target", "gold", "solution", "final_answer"]


def pick(row, keys):
    for key in keys:
        if key in row:
            return key
    raise KeyError(f"Cannot infer key from {list(row.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Download math datasets and convert to slime-friendly jsonl.")
    parser.add_argument("--name", choices=sorted(DATASETS), required=True)
    parser.add_argument("--out-dir", default="data/math")
    parser.add_argument("--hub", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--label-key", default="label")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc

    hub, split = DATASETS[args.name]
    hub = args.hub or hub
    split = args.split or split
    rows = list(load_dataset(hub, split=split))
    if not rows:
        raise SystemExit(f"No rows loaded from {hub}:{split}")

    input_key = pick(rows[0], INPUT_KEYS)
    answer_key = pick(rows[0], ANSWER_KEYS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.jsonl"
    with out_path.open("w") as f:
        for row in rows:
            item = {
                args.prompt_key: row[input_key],
                args.label_key: row[answer_key],
                "source": args.name,
                "source_dataset": hub,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(out_path)
    print(f"rows={len(rows)} input_key={input_key} answer_key={answer_key}")


if __name__ == "__main__":
    main()
