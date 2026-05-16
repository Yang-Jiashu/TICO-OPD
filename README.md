# TICO-OPD

**Trajectory-Influence Compression for On-Policy Distillation**

TICO-OPD is an experimental extension to `slime` OPD that uses Future-KL-style token credit assignment to distill the tokens that steer reasoning, while applying compression pressure only to low-influence continuation tokens.

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

This repo contains a patch-style implementation for `slime`.

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

Start conservative. This tests trajectory weighting and gentle compression without rollout-side masking:

```bash
--use-opd \
--opd-type sglang \
--opd-kl-coef 1.0 \
--custom-rm-path slime.rollout.on_policy_distillation.reward_func \
--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
--policy-loss-type future_kl \
--future-kl-decay-rate 32 \
--future-kl-start include_current \
--future-kl-window -1 \
--future-kl-clip-ratio 0.2 \
--future-kl-clip-high-only \
--future-kl-safety-threshold 10.0 \
--eps-clip-c 3.0 \
--use-compression-opd \
--compression-length-budget-ratio 0.75 \
--compression-advantage-coef 0.02 \
--compression-eos-coef 0.01 \
--compression-coverage-threshold 0.90 \
--compression-importance-decay-rate 32 \
--compression-reward-coef 0.0
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
├── IMPLEMENTATION.md
├── assets/
│   ├── tico-opd-overview.png
│   └── future-kl-vs-entropy.png
├── slime/
│   ├── utils/
│   │   ├── ppo_utils.py
│   │   └── arguments.py
│   ├── rollout/
│   │   └── on_policy_distillation.py
│   └── backends/megatron_utils/
│       ├── loss.py
│       ├── model.py
│       └── data.py
└── tests/
    └── utils/test_future_kl_policy_loss.py
```

## Algorithmic References

TICO-OPD builds on three ideas:

1. **Future-KL token credit assignment** from FIPO: Future-KL Influenced Policy Optimization introduces discounted future-KL as a way to assign denser token-level credit for reasoning trajectories.

2. **On-policy distillation** from OPD-style training: the student samples from its own policy, while a stronger teacher provides token-level supervision on those on-policy responses.

3. **Selective compression**: once token influence is available, low-influence spans can be penalized, masked, or later rewritten while high-influence tokens remain protected.

Useful links:

- FIPO paper page: https://huggingface.co/papers/2603.19835
- FIPO arXiv: https://arxiv.org/abs/2603.19835
- TIP / token importance in OPD: https://arxiv.org/abs/2604.14084

## What This Is Not Yet

This implementation does **not** yet perform true span rewriting:

```text
[y_i ... y_j] -> z
```

The current implementation is training-side shaping:

```text
protect high-importance tokens
penalize low-importance continuation
encourage EOS after high coverage
```

Span-level behavior-equivalent rewriting can be added as a later data-generation stage:

```text
accept rewrite z only if:
len(z) < len(y_i ... y_j)
Future-KL(original span, compressed span) < epsilon
```

## Suggested Experiment Ladder

1. Run vanilla OPD as baseline.
2. Add `--policy-loss-type future_kl`.
3. Add `--use-compression-opd` with small coefficients.
4. Reduce `--compression-length-budget-ratio` slowly.
5. Only then test low-importance masking.

This makes it easier to separate gains from better trajectory credit assignment, compression shaping, and hard token masking.

## Figure Generation

The README figures were generated with GPT Image 2 for documentation purposes:

```text
assets/tico-opd-overview.png
assets/future-kl-vs-entropy.png
```

They are explanatory diagrams, not empirical results.
