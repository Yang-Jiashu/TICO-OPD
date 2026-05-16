#!/usr/bin/env python3
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


DATASETS = {
    "aime24": ("OpenRLHF/aime-2024", "train", "Apache-2.0"),
    "aime25": ("MathArena/aime_2025", "train", "Dataset card"),
    "math500": ("HuggingFaceH4/MATH-500", "test", "MIT"),
    "dapo17k": ("BytedTsinghua-SIA/DAPO-Math-17k", "train", "Dataset card"),
}

INPUT_KEYS = ["problem", "question", "prompt", "input", "query"]
ANSWER_KEYS = ["answer", "label", "target", "gold", "solution", "final_answer"]


def pick(row, keys):
    for key in keys:
        if key in row:
            return key
    raise KeyError(f"Cannot infer key from {list(row.keys())}")


def normalize_prompt(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        contents = []
        for message in value:
            if isinstance(message, dict) and message.get("content"):
                contents.append(str(message["content"]))
        if contents:
            return "\n".join(contents)
    return str(value)


def normalize_answer(row, key):
    value = row[key]
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for nested_key in ("ground_truth", "answer", "label", "target"):
            if nested_key in value:
                return str(value[nested_key])
    return str(value)


def fetch_rows_from_datasets_server(dataset, split, batch_size=100, sleep=0.2, max_rows=0):
    rows = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": batch_size,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        last_error = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    payload = json.load(response)
                batch = [entry["row"] for entry in payload.get("rows", [])]
                break
            except Exception as exc:
                last_error = exc
                time.sleep(min(8, 2**attempt))
        else:
            raise RuntimeError(f"Failed to fetch {dataset}:{split} at offset={offset}: {last_error}")

        if not batch:
            return rows
        rows.extend(batch)
        print(f"fetched {len(rows)} rows from {dataset}:{split}", flush=True)
        if max_rows and len(rows) >= max_rows:
            return rows[:max_rows]
        if len(batch) < batch_size:
            return rows
        offset += len(batch)
        time.sleep(sleep)


def load_rows(dataset, split, prefer_http=False, max_rows=0):
    if not prefer_http:
        try:
            from datasets import load_dataset

            rows = list(load_dataset(dataset, split=split))
            return rows[:max_rows] if max_rows else rows
        except ImportError:
            pass
        except Exception as exc:
            print(f"datasets.load_dataset failed, falling back to HTTP rows API: {exc}")
    return fetch_rows_from_datasets_server(dataset, split, max_rows=max_rows)


def infer_keys(row):
    input_key = pick(row, INPUT_KEYS)
    try:
        answer_key = pick(row, ANSWER_KEYS)
    except KeyError:
        if "reward_model" in row:
            answer_key = "reward_model"
        else:
            raise
    return input_key, answer_key


def convert_rows(rows, name, hub, prompt_key, label_key):
    if not rows:
        raise SystemExit(f"No rows loaded for {name}")
    input_key, answer_key = infer_keys(rows[0])
    converted = []
    for idx, row in enumerate(rows):
        item = {
            prompt_key: normalize_prompt(row[input_key]),
            label_key: normalize_answer(row, answer_key),
            "source": name,
            "source_dataset": hub,
            "row_idx": idx,
        }
        if "subject" in row:
            item["subject"] = row["subject"]
        if "level" in row:
            item["level"] = row["level"]
        converted.append(item)
    return converted, input_key, answer_key


def write_jsonl(path, rows):
    with path.open("w") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download math datasets and convert to slime-friendly jsonl.")
    parser.add_argument("--name", choices=sorted(DATASETS), required=True)
    parser.add_argument("--out-dir", default="data/math")
    parser.add_argument("--hub", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--prefer-http", action="store_true", help="Use the Hugging Face datasets-server rows API.")
    parser.add_argument("--shard-size", type=int, default=0, help="Write shard files with at most this many rows.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap for smoke-test data preparation.")
    args = parser.parse_args()

    hub, split, license_name = DATASETS[args.name]
    hub = args.hub or hub
    split = args.split or split
    rows = load_rows(hub, split, prefer_http=args.prefer_http, max_rows=args.max_rows)
    converted, input_key, answer_key = convert_rows(rows, args.name, hub, args.prompt_key, args.label_key)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if args.shard_size and len(converted) > args.shard_size:
        for shard_idx, start in enumerate(range(0, len(converted), args.shard_size)):
            shard = converted[start : start + args.shard_size]
            out_path = out_dir / f"{args.name}-{shard_idx:05d}.jsonl"
            write_jsonl(out_path, shard)
            written.append(out_path)
    else:
        out_path = out_dir / f"{args.name}.jsonl"
        write_jsonl(out_path, converted)
        written.append(out_path)

    source_path = out_dir / "SOURCES.json"
    if source_path.exists():
        sources = json.loads(source_path.read_text())
    else:
        sources = {}
    sources[args.name] = {
        "hub": hub,
        "split": split,
        "license": license_name,
        "rows": len(converted),
        "input_key": input_key,
        "answer_key": answer_key,
        "files": [path.name for path in written],
    }
    if args.max_rows:
        sources[args.name]["max_rows"] = args.max_rows
    source_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n")

    for path in written:
        print(path)
    print(f"rows={len(converted)} input_key={input_key} answer_key={answer_key}")


if __name__ == "__main__":
    main()
