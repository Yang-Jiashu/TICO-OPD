# TICO-OPD

**Trajectory-Influence Compression for On-Policy Distillation**

TICO-OPD is an experimental full-framework fork of `slime` OPD that uses Future-KL-style token credit assignment to distill the tokens that steer reasoning, while applying compression pressure only to low-influence continuation tokens.

```text
Distill what changes the trajectory.
Compress what does not.
```

![TICO-OPD overview](assets/tico-opd-overview.png)

## Why TICO-OPD

Normal on-policy distillation is good at preserving teacher behavior, but it often inherits the teacher's verbosity:

```text
teacher long answer -> student learns long answer
```

TICO-OPD changes the training signal from sequence-level imitation to trajectory-aware token credit assignment:

```text
teacher rollout
  -> estimate which tokens influence future behavior
  -> distill high-influence tokens more strongly
  -> discourage low-influence continuation tokens
  -> encourage EOS after enough important content is covered
```

The goal is not blind length reduction. The goal is **behavior-preserving compression**:

```text
same useful reasoning behavior, fewer unnecessary tokens
```

## Core Idea

For each response token `y_t`, TICO-OPD estimates an importance score:

```text
I_t ~= Future-KL(t)
```

High `I_t` means changing that token is likely to change later model behavior. These tokens are protected and receive stronger distillation.

Low `I_t` means the token is likely to be filler, repeated explanation, formatting, or a low-value continuation. These tokens become compression candidates.

The training-side shaping is:

```text
A'_t = A_t
     - lambda * (1 - I_t) * compressible_t
     + eos_bonus_t
```

where `compressible_t` becomes active after a length budget or after cumulative importance coverage is high.

![Future-KL vs entropy](assets/future-kl-vs-entropy.png)

## What Is Implemented

This repo now contains the full `slime` training framework plus the TICO-OPD changes. It includes:

```text
train.py / train_async.py
Ray rollout and training orchestration
Megatron backend utilities
SGLang rollout utilities
model scripts, examples, docs, tests, and docker files
TICO-OPD algorithm changes
Qwen3 math training and evaluation scripts
```

The original upstream README is kept as:

```text
UPSTREAM_SLIME_README.md
UPSTREAM_SLIME_README_zh.md
```

### 1. Future-KL Policy Loss

Enable with:

```bash
--policy-loss-type future_kl
```

This reweights the policy objective with a discounted future trajectory signal.

Code:

```text
slime/utils/ppo_utils.py
  compute_future_token_importance(...)
  compute_future_kl_policy_loss(...)

slime/backends/megatron_utils/loss.py
  policy_loss_type == "future_kl"
```

### 2. Compression-Aware OPD

Enable with:

```bash
--use-compression-opd
```

This applies selective compression pressure after advantages are computed:

```text
high-importance token -> protect
low-importance token beyond budget -> penalize continuation
coverage high -> encourage stopping
coverage low -> avoid premature EOS
```

Code:

```text
slime/backends/megatron_utils/loss.py
  apply_compression_opd_to_advantages(...)

slime/rollout/on_policy_distillation.py
  reward_func(...)
  post_process_rewards(...)
```

### 3. OPD Teacher Logprob Plumbing

Teacher logprobs are extracted during OPD rollout post-processing and stored on each sample:

```text
sample.teacher_log_probs
sample.loss_mask
sample.metadata["compression_importance_mean"]
sample.metadata["compression_low_importance_tokens"]
```

Safety detail: if teacher logprobs do not cover the full response, missing positions are masked out instead of being treated as valid zero-logprob supervision.

Code:

```text
slime/rollout/on_policy_distillation.py
  _fit_response_log_probs(...)
```

### 4. CLI Arguments

All new arguments are wired through:

```text
slime/utils/arguments.py
```

Main switches:

```bash
--policy-loss-type future_kl
--use-compression-opd
--compression-length-budget-ratio
--compression-coverage-threshold
--compression-advantage-coef
--compression-eos-coef
--compression-reward-coef
```

## Quick Start

This path is for a first reproducible Qwen3 math run. It assumes you already have a CUDA environment that can run `slime`, `Megatron-LM`, Ray, and SGLang.

### 1. Clone and install

```bash
git clone https://github.com/Yang-Jiashu/TICO-OPD.git
cd TICO-OPD

pip install -r requirements.txt
pip install pyyaml
```

SwanLab is optional:

```bash
pip install swanlab
```

### 2. Check bundled data

The repo already includes training and evaluation jsonl files:

```bash
wc -l data/math/*.jsonl
```

Expected counts:

```text
aime24.jsonl       30
aime25.jsonl       30
math500.jsonl      500
dapo17k.jsonl      17000
```

Each row uses:

```json
{"prompt": "...", "label": "...", "source": "...", "source_dataset": "...", "row_idx": 0}
```

### 3. Edit one YAML config

Start from:

```text
configs/qwen3/tico_opd_4b_32b.yaml
```

At minimum, set the model and framework paths:

```yaml
paths:
  slime_dir: .
  megatron_dir: /root/Megatron-LM
  save_dir: /root/checkpoints/qwen3-4B-tico-opd

models:
  student_size: 4B
  teacher_size: 32B
  base_model: /root/Qwen3-4B
  teacher_model: /root/Qwen3-32B
  ref_load: /root/Qwen3-4B_torch_dist
```

Set GPU layout:

```yaml
resources:
  num_gpus: 8
  actor_num_gpus: 8
  rollout_num_gpus: 8
  student_tp: 1
  teacher_tp: 2
  teacher_cuda_visible_devices: "0,1"
```

If the teacher runs on another machine, use:

```yaml
teacher_server:
  use_external_teacher: true
  teacher_url: http://teacher-host:13141/generate
```

### 4. Dry-run the config

Before launching Ray/SGLang, check what the YAML expands to:

```bash
python3 scripts/run_qwen3_tico_opd_from_yaml.py \
  --config configs/qwen3/tico_opd_4b_32b.yaml \
  --dry-run
```

This prints all exported model, data, GPU, rollout, eval, training, tracking, and TICO hyperparameters.

### 5. Run training

```bash
python3 scripts/run_qwen3_tico_opd_from_yaml.py \
  --config configs/qwen3/tico_opd_4b_32b.yaml
```

The default YAML enables:

```text
OPD teacher logprob distillation
Future-KL trajectory credit assignment
compression-aware advantage shaping
AIME24/AIME25/MATH500 periodic eval
```

### 6. Run standalone eval

Launch a Qwen3 endpoint:

```bash
MODEL_SIZE=4B \
CUDA_VISIBLE_DEVICES=0 \
TP=1 \
PORT=30000 \
bash scripts/launch_qwen3_sglang.sh
```

Then evaluate:

```bash
MODEL=Qwen3-4B \
BASE_URL=http://127.0.0.1:30000/v1 \
N_SAMPLES=16 \
bash scripts/eval_qwen3_math.sh
```

### 7. Enable SwanLab

In YAML:

```yaml
tracking:
  use_swanlab: true
  swanlab_mode: cloud
  swanlab_project: TICO-OPD
  swanlab_workspace: your-workspace
  swanlab_experiment_name: qwen3-4B-teacher-32B
  swanlab_group: qwen3-tico-opd
  swanlab_tags: tico-opd,qwen3
```

SwanLab will receive the same unified metric stream as W&B/TensorBoard, including train, rollout, eval, perf, and TICO-specific metrics.

### Conservative Defaults

The default config starts conservatively:

```text
future_kl enabled
compression_opd enabled
compression_length_budget_ratio = 0.75
compression_advantage_coef = 0.02
compression_eos_coef = 0.01
compression_mask_low_importance_tokens = false
```

## Qwen3 Math Evaluation

TICO-OPD includes a small math evaluation harness for Qwen3-style reasoning models.

Covered datasets:

```text
AIME24  -> OpenRLHF/aime-2024
AIME25  -> MathArena/aime_2025
MATH500 -> HuggingFaceH4/MATH-500
DAPO17k -> BytedTsinghua-SIA/DAPO-Math-17k, for training prompts
```

The repo includes ready-to-use jsonl files:

```text
data/math/aime24.jsonl
data/math/aime25.jsonl
data/math/math500.jsonl
data/math/dapo17k.jsonl
```

Regenerate them when needed:

```bash
python3 scripts/prepare_math_data.py --name aime24 --out-dir data/math --prefer-http
python3 scripts/prepare_math_data.py --name aime25 --out-dir data/math --prefer-http
python3 scripts/prepare_math_data.py --name math500 --out-dir data/math --prefer-http
python3 scripts/prepare_math_data.py --name dapo17k --out-dir data/math --prefer-http --max-rows 17000
```

The prepared files use normalized keys:

```json
{"prompt": "...", "label": "...", "source": "dapo17k", "source_dataset": "..."}
```

See [DATA.md](DATA.md) for the exact expected format.

Launch a Qwen3 endpoint with sglang:

```bash
MODEL_SIZE=4B \
CUDA_VISIBLE_DEVICES=0 \
TP=1 \
PORT=30000 \
bash scripts/launch_qwen3_sglang.sh
```

Run AIME24, AIME25, and MATH500 evaluation:

```bash
MODEL=Qwen3-4B \
BASE_URL=http://127.0.0.1:30000/v1 \
N_SAMPLES=16 \
bash scripts/eval_qwen3_math.sh
```

The evaluator uses an OpenAI-compatible `/v1/chat/completions` endpoint and extracts final answers from `\boxed{...}` or common final-answer phrases.

Qwen3 thinking-mode defaults:

```text
temperature = 0.6
top_p       = 0.95
top_k       = 20
min_p       = 0.0
max_tokens  = 16384
```

## Qwen3 TICO-OPD Training Example

The general Qwen3 TICO-OPD training launcher is:

```bash
STUDENT_SIZE=4B \
TEACHER_SIZE=32B \
bash scripts/run_qwen3_tico_opd.sh
```

The compatibility wrapper below is equivalent to `STUDENT_SIZE=4B TEACHER_SIZE=32B`:

```bash
bash scripts/run_qwen3_4b_tico_opd.sh
```

Supported Qwen3 size names:

```text
0.6B
1.7B
4B
4B-Instruct-2507
8B
14B
32B
30B-A3B
235B-A22B
```

Example teacher/student pairs:

```bash
# cheap smoke run
STUDENT_SIZE=1.7B TEACHER_SIZE=8B bash scripts/run_qwen3_tico_opd.sh

# default dense distillation
STUDENT_SIZE=4B TEACHER_SIZE=32B bash scripts/run_qwen3_tico_opd.sh

# stronger MoE teacher
STUDENT_SIZE=8B TEACHER_SIZE=235B-A22B TEACHER_TP=8 bash scripts/run_qwen3_tico_opd.sh
```

For reproducible runs, keep the full experiment in one YAML file:

```bash
python3 scripts/run_qwen3_tico_opd_from_yaml.py \
  --config configs/qwen3/tico_opd_4b_32b.yaml
```

Preview what the YAML exports without launching Ray/SGLang:

```bash
python3 scripts/run_qwen3_tico_opd_from_yaml.py \
  --config configs/qwen3/tico_opd_4b_32b.yaml \
  --dry-run
```

The YAML groups model paths, teacher server settings, rollout inference hyperparameters, eval inference hyperparameters, optimizer settings, GPU layout, and TICO-OPD algorithm knobs.

Important environment variables:

```bash
SLIME_DIR=/path/to/TICO-OPD
MEGATRON_DIR=/root/Megatron-LM
STUDENT_SIZE=4B
TEACHER_SIZE=32B
BASE_MODEL=/root/Qwen3-${STUDENT_SIZE}
TEACHER_MODEL=/root/Qwen3-${TEACHER_SIZE}
TRAIN_DATA=$SLIME_DIR/data/math/dapo17k.jsonl
AIME24=$SLIME_DIR/data/math/aime24.jsonl
AIME25=$SLIME_DIR/data/math/aime25.jsonl
MATH500=$SLIME_DIR/data/math/math500.jsonl
SAVE_DIR=/root/checkpoints/qwen3-${STUDENT_SIZE}-tico-opd
NUM_GPUS=8
STUDENT_TP=auto
TEACHER_TP=auto
TEACHER_CUDA_VISIBLE_DEVICES=0,1
USE_EXTERNAL_TEACHER=false
```

Rollout and eval inference hyperparameters:

```bash
NUM_ROLLOUT=3000
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
ROLLOUT_MAX_RESPONSE_LEN=8192
ROLLOUT_TEMPERATURE=0.6
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1

EVAL_INTERVAL=20
N_SAMPLES_PER_EVAL_PROMPT=16
EVAL_MAX_RESPONSE_LEN=16384
EVAL_TEMPERATURE=0.6
EVAL_TOP_P=0.95
EVAL_TOP_K=20
```

Training hyperparameters:

```bash
SAVE_INTERVAL=20
GLOBAL_BATCH_SIZE=256
ADVANTAGE_ESTIMATOR=grpo
EPS_CLIP_C=3.0
OPTIMIZER=adam
LR=1e-6
LR_DECAY_STYLE=constant
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.98
SGLANG_MEM_FRACTION_STATIC=0.7
```

SwanLab tracking is optional. Install it only when you want to log experiments there:

```bash
pip install swanlab
```

Then enable it in YAML:

```yaml
tracking:
  use_swanlab: true
  swanlab_mode: cloud
  swanlab_project: TICO-OPD
  swanlab_workspace: your-workspace
  swanlab_experiment_name: qwen3-4B-teacher-32B
  swanlab_group: qwen3-tico-opd
  swanlab_tags: tico-opd,qwen3
```

or with environment variables:

```bash
USE_SWANLAB=true \
SWANLAB_PROJECT=TICO-OPD \
SWANLAB_WORKSPACE=your-workspace \
SWANLAB_API_KEY=... \
bash scripts/run_qwen3_tico_opd.sh
```

During training, SwanLab/W&B/TensorBoard can receive the same unified metric stream. Important tracked groups include:

```text
train/loss
train/pg_loss
train/entropy_loss
train/pg_clipfrac
train/ppo_kl
train/opd_reverse_kl
train/opd_reverse_kl_abs
train/student_logprob
train/teacher_logprob
train/future_kl_*
train/compression_importance_mean
train/compression_importance_max
train/compression_penalty_mean
train/compression_penalty_nonzero_ratio
train/compression_zone_ratio
train/compression_low_importance_ratio
train/compression_eos_bonus_mean
train/compression_eos_bonus_nonzero_ratio
train/compression_coverage_mean
rollout/response_len/*
rollout/tico/compression_importance_mean
rollout/tico/compression_low_importance_ratio
rollout/tico/compression_zone_ratio
rollout/tico/teacher_logprob_mean
rollout/tico/opd_valid_token_ratio
eval/aime24/*
eval/aime25/*
eval/math500/*
perf/*
```

TICO-OPD algorithm hyperparameters:

```bash
OPD_KL_COEF=1.0
POLICY_LOSS_TYPE=future_kl
FUTURE_KL_DECAY_RATE=32
FUTURE_KL_START=include_current
FUTURE_KL_WINDOW=-1
FUTURE_KL_AVERAGE=false
FUTURE_KL_CLIP_RATIO=0.2
FUTURE_KL_CLIP_HIGH_ONLY=true
FUTURE_KL_SAFETY_THRESHOLD=10.0

USE_COMPRESSION_OPD=true
COMPRESSION_LENGTH_BUDGET=-1
COMPRESSION_LENGTH_BUDGET_RATIO=0.75
COMPRESSION_ADVANTAGE_COEF=0.02
COMPRESSION_EOS_COEF=0.01
COMPRESSION_COVERAGE_THRESHOLD=0.90
COMPRESSION_MIN_RESPONSE_LEN=0
COMPRESSION_IMPORTANCE_DECAY_RATE=32
COMPRESSION_IMPORTANCE_START=include_current
COMPRESSION_IMPORTANCE_WINDOW=-1
COMPRESSION_IMPORTANCE_AVERAGE=false
COMPRESSION_IMPORTANCE_TEMPERATURE=1.0
COMPRESSION_REWARD_COEF=0.0
COMPRESSION_MASK_LOW_IMPORTANCE_TOKENS=false
COMPRESSION_LOW_IMPORTANCE_THRESHOLD=0.2
```

This script combines:

```text
DAPO17k training prompts
Qwen3 teacher logprob OPD
Future-KL policy loss
compression-aware OPD shaping
AIME24/AIME25/MATH500 eval hooks
```

Teacher and student do **not** need the same number of GPUs.

Recommended patterns:

```bash
# Same machine, explicit split: teacher on 0-1, slime/Ray sees the rest.
TEACHER_CUDA_VISIBLE_DEVICES=0,1 \
TEACHER_TP=2 \
STUDENT_SIZE=4B \
TEACHER_SIZE=32B \
bash scripts/run_qwen3_tico_opd.sh

# Separate teacher service, often best for large teachers.
USE_EXTERNAL_TEACHER=true \
TEACHER_URL=http://teacher-host:13141/generate \
STUDENT_SIZE=4B \
TEACHER_SIZE=235B-A22B \
bash scripts/run_qwen3_tico_opd.sh
```

The student-side GPU layout is controlled by:

```text
NUM_GPUS
ACTOR_NUM_GPUS
ROLLOUT_NUM_GPUS
STUDENT_TP
ROLLOUT_GPUS_PER_ENGINE
MAX_TOKENS_PER_GPU
```

The teacher-side GPU layout is controlled independently by:

```text
TEACHER_TP
TEACHER_CUDA_VISIBLE_DEVICES
TEACHER_MEM_FRACTION_STATIC
USE_EXTERNAL_TEACHER
TEACHER_URL
```

Then increase compression gradually:

```bash
--compression-length-budget-ratio 0.60 \
--compression-advantage-coef 0.05 \
--compression-eos-coef 0.02 \
--compression-reward-coef 0.02
```

Only enable hard low-importance masking after quality is stable:

```bash
--compression-mask-low-importance-tokens \
--compression-low-importance-threshold 0.2
```

## Important Parameters

`--future-kl-decay-rate`

Controls how much future positions contribute to each token's influence score. Larger values make the signal consider a longer future horizon.

`--future-kl-start`

Controls whether the current token contributes to its own future score. `include_current` is usually the first setting to try.

`--compression-length-budget-ratio`

Starts compression pressure after a fraction of the response length. `0.75` means the first 75% of the response is mostly protected from length pressure.

`--compression-coverage-threshold`

Encourages stopping after cumulative importance coverage is high, such as `0.90`.

`--compression-advantage-coef`

Strength of training-side low-importance continuation penalty.

`--compression-eos-coef`

Strength of coverage-aware EOS shaping.

`--compression-reward-coef`

Optional rollout-side scalar penalty for low-importance continuation tokens.

## Metrics to Watch

Track these together:

```text
train/policy_loss
train/future_kl_importance
train/compression_importance
train/compression_penalty
response_length
task reward / eval accuracy
```

Healthy early behavior:

```text
quality is stable
response length decreases slowly
compression_penalty is non-zero but not dominant
importance is not collapsed to all zeros or all ones
```

If quality drops, reduce:

```bash
--compression-advantage-coef
--compression-eos-coef
--compression-reward-coef
```

or relax:

```bash
--compression-length-budget-ratio
--compression-coverage-threshold
```

## Code Map

```text
TICO-OPD
├── README.md
├── DATA.md
├── IMPLEMENTATION.md
├── REFERENCES.md
├── train.py
├── train_async.py
├── docker/
├── docs/
├── scripts/
│   ├── run_qwen3_tico_opd.sh
│   ├── launch_qwen3_sglang.sh
│   └── models/
├── eval/
│   ├── run_math_eval.py
│   └── math_answer.py
├── configs/
│   └── qwen3/
├── assets/
│   ├── tico-opd-overview.png
│   └── future-kl-vs-entropy.png
├── slime/
│   ├── ray/
│   ├── rollout/
│   ├── backends/
│   └── utils/
├── slime_plugins/
├── examples/
└── tests/
```

## Implemented vs Optional Extensions

Implemented in this repo:

```text
Future-KL-style trajectory weighting
teacher-logprob OPD plumbing
compression-aware advantage shaping
coverage-aware EOS pressure
Qwen3 teacher/student launch scripts
AIME24/AIME25/MATH500 evaluation scripts
DAPO-Math-17k data preparation
```

Optional extension, not part of the core implementation:

```text
[y_i ... y_j] -> z
```

This is true span rewriting: replacing a multi-token span with a shorter behavior-equivalent phrase.

Why it is not in the core path:

1. It is a separate data-generation algorithm, not the OPD training objective itself.
2. It requires an additional rewrite model or teacher call, plus a judge/filter.
3. A behavior-equivalent rewrite should be accepted only after checking that future behavior changes little, which means extra Future-KL or proxy computation.
4. Mixing span rewriting into the first implementation makes ablations unclear: gains could come from better OPD credit assignment, from data rewriting, or from both.

So the current implementation intentionally focuses on training-time compression shaping:

```text
protect high-importance tokens
penalize low-importance continuation
encourage EOS after high coverage
```

If span rewriting is added later, the acceptance rule should be:

```text
accept rewrite z only if:
len(z) < len(y_i ... y_j)
Future-KL(original span, compressed span) < epsilon
```

In short: TICO-OPD already uses the importance signal to reduce output length during training; span rewriting is the next optional layer for data-side compression.

## References and Provenance

See [REFERENCES.md](REFERENCES.md) for papers, datasets, Qwen3 references, and upstream code provenance.

Short version:

- Future-KL credit assignment is from FIPO, not newly invented here.
- OPD/token-importance background is related to TIP and OPD literature.
- The implementation is a patch on top of `THUDM/slime`; files under `slime/` are modified slime files.
- Qwen3 launch defaults follow Qwen3 thinking-mode sampling recommendations.
- AIME24/AIME25/MATH500/DAPO17k are external datasets; this repo only provides download/conversion/evaluation scripts.

## Suggested Experiment Ladder

1. Run vanilla OPD as baseline.
2. Add `--policy-loss-type future_kl`.
3. Add `--use-compression-opd` with small coefficients.
4. Reduce `--compression-length-budget-ratio` slowly.
5. Only then test low-importance masking.

This makes it easier to separate gains from better trajectory credit assignment, compression shaping, and hard token masking.
