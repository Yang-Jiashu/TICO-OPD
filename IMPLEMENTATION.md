# Implementation Map

This repository contains the TICO-OPD patch files for `slime`.

## Files

`slime/utils/ppo_utils.py`

Future trajectory importance utilities:

- `compute_future_token_importance`
- `compute_future_kl_policy_loss`
- token-importance normalization

`slime/backends/megatron_utils/loss.py`

Training-side integration:

- `--policy-loss-type future_kl`
- compression-aware advantage shaping
- logging for compression importance and penalty

`slime/utils/arguments.py`

CLI arguments and validation for Future-KL policy loss and compression-aware OPD.

`slime/rollout/on_policy_distillation.py`

OPD teacher logprob extraction and rollout-side compression metadata.

Important safety detail: if teacher logprobs do not cover the full response, the missing positions are masked out rather than treated as real zero logprob supervision.

`slime/backends/megatron_utils/model.py`

Batch key plumbing for compression tensors.

`slime/backends/megatron_utils/data.py`

Metric aggregation support for compression tensors.

`tests/utils/test_future_kl_policy_loss.py`

Unit tests for the future-KL loss behavior.

`docs/*/get_started/usage.md`

Short user-facing usage documentation in English and Chinese.

`examples/compression_opd/README.md`

Full algorithm README and recommended launch arguments.

## Current Scope

This version implements trajectory-influence weighted OPD and selective compression pressure.

It does not yet implement true span rewriting of the form:

```text
[y_i ... y_j] -> z
```

The current compression mechanism is training-time shaping:

```text
protect high-importance tokens
penalize low-importance continuation tokens
encourage stopping after coverage is high
```

Span-level behavior-equivalent rewriting can be added later as a data generation stage.
