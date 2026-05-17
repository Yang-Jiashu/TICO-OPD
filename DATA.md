# Data Format

TICO-OPD scripts use a simple jsonl format.

## Training Data

For `slime` training, each line should contain at least:

```json
{"prompt": "Solve ...", "label": "42"}
```

The launcher uses:

```bash
--prompt-data data/math/dapo17k.jsonl
--input-key prompt
--label-key label
```

The repo already includes a ready-to-use math data bundle:

```text
data/math/dapo17k.jsonl
data/math/aime24.jsonl
data/math/aime25.jsonl
data/math/math500.jsonl
data/math/SOURCES.json
```

The default Qwen3 launcher points to these files when run from this repo.

To regenerate the files, use the conversion script:

```bash
python3 scripts/prepare_math_data.py --name dapo17k --out-dir data/math --prefer-http --max-rows 17000
```

It downloads the source dataset and writes:

```text
data/math/dapo17k.jsonl
```

with normalized keys:

```text
prompt
label
source
source_dataset
```

## Evaluation Data

The same conversion script can prepare AIME24, AIME25, and MATH500:

```bash
python3 scripts/prepare_math_data.py --name aime24 --out-dir data/math --prefer-http
python3 scripts/prepare_math_data.py --name aime25 --out-dir data/math --prefer-http
python3 scripts/prepare_math_data.py --name math500 --out-dir data/math --prefer-http
```

or use the checked-in files directly:

```text
data/math/aime24.jsonl
data/math/aime25.jsonl
data/math/math500.jsonl
```

## Provenance

Dataset sources, row counts, field mappings, and included files are recorded in:

```text
data/math/SOURCES.json
```

The checked-in `dapo17k.jsonl` is a 17,000-row repo-ready subset. The upstream Hugging Face split currently reports many more rows than the dataset name suggests, so the full split is not vendored into Git.

The standalone evaluator can also read Hugging Face datasets directly:

```bash
python3 eval/run_math_eval.py \
  --dataset-name OpenRLHF/aime-2024 \
  --split train \
  --base-url http://127.0.0.1:30000/v1 \
  --model Qwen3-4B
```

## Answer Extraction

`eval/math_answer.py` extracts final answers from:

```text
\boxed{...}
answer is ...
final answer is ...
last numeric expression
```

This is lightweight and intended for quick iteration. For leaderboard-grade reporting, run an official evaluator or a stricter math equivalence checker.
