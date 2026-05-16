# TICO-OPD

**Trajectory-Influence Compression for On-Policy Distillation**

TICO-OPD is an experimental OPD extension for learning the teacher's reasoning behavior while reducing unnecessary output length.

The key idea is simple:

```text
Do not compress important reasoning tokens.
Compress only low-importance continuation tokens.
```

Instead of applying a global length penalty, TICO-OPD estimates which tokens influence the future reasoning trajectory, strengthens distillation on those tokens, and applies compression pressure only where the trajectory signal is weak.

## Why

Normal OPD can preserve teacher capability, but it often inherits the teacher's verbose style.

```text
Teacher long answer -> student learns long answer
```

TICO-OPD changes the credit assignment:

```text
Teacher long answer
  -> identify trajectory-critical tokens
  -> distill critical tokens more strongly
  -> penalize low-importance continuation tokens
  -> encourage stopping after coverage is high
```

The result should be a student that keeps important reasoning behavior while learning to stop earlier when the useful information is already covered.

## Core Intuition

TICO-OPD uses a token importance signal:

```text
I_t = normalized future trajectory influence of token t
```

High `I_t` means the token is likely to affect future reasoning. These tokens should be protected and distilled strongly.

Low `I_t` means the token is likely to be filler, repeated explanation, formatting, or low-value continuation. These tokens are compression candidates.

The compression-aware advantage shaping is:

```text
A'_t = A_t - lambda * (1 - I_t) * compressible_t
```

where:

```text
compressible_t = token is beyond length budget
              or token appears after enough importance coverage
```

This gives the model a very specific lesson:

```text
keep what changes the reasoning trajectory,
shorten what does not.
```

## What Is Implemented

TICO-OPD has two implemented pieces in this repo.

### 1. Future-KL Policy Loss

Enable:

```bash
--policy-loss-type future_kl
```

This adds a FIPO-style future-KL influence weight to the policy loss.

In slime notation:

```text
ppo_kl_t = old_log_prob_t - log_prob_t
negative_approx_kl_t = log_prob_t - old_log_prob_t = -ppo_kl_t
```

The future signal is accumulated over the response suffix:

```text
FutureKL_t = discounted_sum_{k >= t}(negative_approx_kl_k)
```

Then the token policy loss is weighted by the future influence signal.

Use this when you want OPD/RL updates to focus on tokens that change downstream reasoning behavior.

### 2. Compression-Aware OPD Shaping

Enable:

```bash
--use-compression-opd
```

This applies selective compression pressure after advantages are computed:

```text
high-importance tokens -> protected
low-importance continuation tokens -> penalized
last token / EOS region -> rewarded when coverage is high
truncated samples -> penalized
```

Training-time importance uses teacher/student divergence when teacher log-probs are available through OPD.

The OPD rollout post-process also includes a cheaper teacher-surprise proxy for:

```text
reward shaping
optional low-importance loss masking
```

## Quick Start

Start conservative. This turns on trajectory weighting and gentle compression without rollout-side masking:

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

This is the recommended first run because it mostly tests the training-side signal.

## Stronger Compression

After the first run is stable, increase compression gradually:

```bash
--compression-length-budget-ratio 0.60 \
--compression-advantage-coef 0.05 \
--compression-eos-coef 0.02 \
--compression-reward-coef 0.02
```

Only enable rollout-side low-importance masking after quality is stable:

```bash
--compression-mask-low-importance-tokens \
--compression-low-importance-threshold 0.2
```

This sets `loss_mask=0` for low-importance compressible tokens in the OPD rollout post-process path. In plain language: the student no longer directly distills those verbose parts.

## Main Arguments

### Trajectory-Influence Policy Loss

`--policy-loss-type future_kl`

Use future-KL influenced policy loss.

`--future-kl-decay-rate`

Controls how far future token drift influences the current token. Larger values make the signal more long-range.

`--future-kl-start`

`include_current` includes token `t` in its own future signal. `exclude_current` starts from the next token.

`--future-kl-window`

Maximum future window. `-1` means full suffix.

`--future-kl-clip-ratio`

Clips the influence weight. Start with `0.2`.

### Compression Shaping

`--use-compression-opd`

Enable compression-aware advantage shaping.

`--compression-length-budget`

Absolute response token budget. Use this when you know the desired target length.

`--compression-length-budget-ratio`

Relative response budget. `0.75` means compression pressure starts after roughly 75% of the sampled response.

`--compression-coverage-threshold`

Cumulative importance threshold. `0.90` means later low-importance tokens become compressible once 90% of token importance has been covered.

`--compression-advantage-coef`

Penalty strength for low-importance continuation tokens. Start small.

`--compression-eos-coef`

Final-token shaping coefficient. Completed samples can get a small bonus when coverage is high; truncated samples get a penalty.

`--compression-reward-coef`

Rollout reward penalty for low-importance compressible tokens. Keep this at `0.0` for the first run.

`--compression-mask-low-importance-tokens`

Optional stronger mode. Masks low-importance compressible tokens during OPD rollout post-processing.

## Compatibility

Recommended advantage estimators:

```bash
--advantage-estimator grpo
```

or:

```bash
--advantage-estimator ppo
```

Avoid combining:

```bash
--advantage-estimator gspo
--policy-loss-type future_kl
```

Reason: future-KL policy loss uses token-level ratios, while GSPO uses sequence-level KL.

## Metrics To Watch

Watch:

```text
train/compression_importance
train/compression_penalty
train/opd_reverse_kl
train/response_len/mean
train/reward
```

Healthy behavior usually looks like:

```text
response length decreases slowly
reward or accuracy stays flat or improves
opd_reverse_kl does not spike
compression_penalty is non-zero but not dominant
```

If quality drops, reduce:

```bash
--compression-advantage-coef
--compression-eos-coef
--compression-reward-coef
```

or relax compression:

```bash
--compression-length-budget-ratio
--compression-coverage-threshold
```

## What This Is Not

TICO-OPD does not currently rewrite the model output into a new shorter text sequence with an external compressor.

It implements the training machinery for compression:

```text
trajectory-critical token protection
low-importance continuation penalty
coverage-aware stopping pressure
optional low-importance loss masking
```

For explicit span rewriting:

```text
[y_i ... y_j] -> z
len(z) < len(y_i ... y_j)
Future-KL(original span, compressed span) < epsilon
```

add a custom rollout function that generates both long and short variants, then train with the same TICO-OPD knobs.

## Suggested Experiment Ladder

1. Run OPD baseline and record quality plus response length.
2. Add `--policy-loss-type future_kl` only.
3. Add `--use-compression-opd` with small coefficients.
4. Increase compression coefficient or reduce budget ratio.
5. Only then enable low-importance loss masking.

This ladder makes it easier to tell whether gains come from better trajectory credit assignment, compression shaping, or masking.
