#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from math_answer import extract_answer, is_correct


DEFAULT_PROMPT = """Solve the problem. Reason carefully, and put the final answer in \\boxed{{}}.

Problem:
{problem}
"""

INPUT_KEYS = ["problem", "question", "prompt", "input", "query"]
ANSWER_KEYS = ["answer", "label", "target", "gold", "solution", "final_answer"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(args) -> list[dict]:
    if args.dataset_path:
        path = Path(args.dataset_path)
        if path.suffix == ".jsonl":
            return load_jsonl(path)
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            return data if isinstance(data, list) else data.get("data", [])
        if path.suffix == ".csv":
            with path.open() as f:
                return list(csv.DictReader(f))
        if path.suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise SystemExit("Reading parquet requires pandas: pip install pandas pyarrow") from exc
            return pd.read_parquet(path).to_dict("records")
        raise SystemExit(f"Unsupported dataset file: {path}")

    if not args.dataset_name:
        raise SystemExit("Pass --dataset-name or --dataset-path")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Loading Hugging Face datasets requires: pip install datasets") from exc

    ds = load_dataset(args.dataset_name, split=args.split)
    return list(ds)


def pick_key(row: dict, requested: str, candidates: list[str]) -> str:
    if requested != "auto":
        return requested
    for key in candidates:
        if key in row:
            return key
    raise KeyError(f"Cannot infer key from row keys={list(row.keys())}; pass an explicit key.")


def chat_completion(args, prompt: str) -> str:
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": 1,
    }
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.min_p is not None:
        payload["min_p"] = args.min_p

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    data = json.dumps(payload).encode()
    for attempt in range(args.retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                body = json.loads(resp.read().decode())
                return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if attempt + 1 == args.retries:
                raise
            time.sleep(1 + attempt * 2)
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate math datasets against an OpenAI-compatible endpoint.")
    parser.add_argument("--dataset-name", default=None, help="Hugging Face dataset name, e.g. OpenRLHF/aime-2024")
    parser.add_argument("--dataset-path", default=None, help="Local jsonl/json/csv/parquet dataset.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-key", default="auto")
    parser.add_argument("--answer-key", default="auto")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="Qwen3-4B")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    parser.add_argument("--out", default="eval_results/math_eval.jsonl")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    rows = load_rows(args)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows loaded.")

    input_key = pick_key(rows[0], args.input_key, INPUT_KEYS)
    answer_key = pick_key(rows[0], args.answer_key, ANSWER_KEYS)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct_any = 0
    with out_path.open("w") as f:
        for idx, row in enumerate(rows):
            problem = row[input_key]
            reference = row[answer_key]
            prompt = args.prompt_template.format(problem=problem)
            samples = []
            sample_correct = []
            for _ in range(args.n_samples):
                text = chat_completion(args, prompt)
                samples.append(text)
                sample_correct.append(is_correct(text, reference))
            total += 1
            correct_any += int(any(sample_correct))
            record = {
                "idx": idx,
                "problem": problem,
                "reference": reference,
                "predictions": samples,
                "extracted": [extract_answer(x) for x in samples],
                "correct": sample_correct,
                "pass_at_n": bool(any(sample_correct)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(rows)}] pass={record['pass_at_n']} acc={correct_any / total:.4f}")

    summary = {
        "dataset": args.dataset_name or args.dataset_path,
        "split": args.split,
        "model": args.model,
        "n_samples": args.n_samples,
        "total": total,
        "pass_at_n": correct_any / max(total, 1),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
