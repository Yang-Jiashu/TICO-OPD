# Data Format

TICO-OPD scripts use a simple jsonl format.

## Training Data

For `slime` training, each line should contain at least:

```json
{"prompt": "Solve ...", "label": "42"}
```

The launcher uses:

```bash
--prompt-data /root/data/math/dapo17k.jsonl
--input-key prompt
--label-key label
```

Use the conversion script:

```bash
python3 scripts/prepare_math_data.py --name dapo17k --out-dir /root/data/math
```

It downloads the source dataset and writes:

```text
/root/data/math/dapo17k.jsonl
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
python3 scripts/prepare_math_data.py --name aime24 --out-dir /root/data/math
python3 scripts/prepare_math_data.py --name aime25 --out-dir /root/data/math
python3 scripts/prepare_math_data.py --name math500 --out-dir /root/data/math
```

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
