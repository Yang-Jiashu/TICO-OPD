# References and Code Provenance

TICO-OPD is a research prototype built by combining ideas from Future-KL credit assignment, on-policy distillation, and selective compression. This file makes the dependency trail explicit.

## Papers and Algorithmic Ideas

### FIPO: Future-KL Influenced Policy Optimization

Future-KL token credit assignment is from FIPO.

- Paper page: https://huggingface.co/papers/2603.19835
- arXiv: https://arxiv.org/abs/2603.19835

What TICO-OPD uses:

```text
Estimate how much token t influences future model behavior.
Use this signal as dense token-level credit assignment.
```

What TICO-OPD changes:

```text
Move the Future-KL-style influence signal into OPD.
Use high-influence tokens for stronger teacher distillation.
Use low-influence continuation tokens as compression targets.
```

### TIP: Token Importance in On-Policy Distillation

TIP studies which tokens matter in OPD and analyzes entropy plus teacher-student divergence as token-importance signals.

- Paper page: https://huggingface.co/papers/2604.14084
- arXiv: https://arxiv.org/abs/2604.14084

What TICO-OPD uses:

```text
OPD should not treat all tokens as equally useful.
Entropy-only importance is incomplete.
Teacher-student disagreement is an important signal.
```

### On-Policy Distillation Background

OPD trains a student on its own rollouts while matching a teacher distribution at student-visited states.

- slime OPD docs: https://www.mintlify.com/THUDM/slime/advanced/on-policy-distillation
- Related OPD discussion in verl docs: https://verl.readthedocs.io/en/latest/algo/opd.html

## Upstream Code Provenance

This repository is not a full replacement for `slime`. It is a patch-style project built on top of the slime training stack.

Upstream framework:

- THUDM/slime: https://github.com/THUDM/slime

Files under `slime/` in this repo are modified copies or patch targets from the slime codebase:

```text
slime/utils/ppo_utils.py
slime/utils/arguments.py
slime/rollout/on_policy_distillation.py
slime/backends/megatron_utils/loss.py
slime/backends/megatron_utils/model.py
slime/backends/megatron_utils/data.py
```

Scripts were adapted from the structure of slime Qwen3 and OPD examples:

```text
examples/on_policy_distillation/run-qwen3-8B-opd.sh
scripts/run-qwen3-4B.sh
scripts/models/qwen3-*.sh
```

No FIPO or TIP code is copied here. Their papers are used as algorithmic references.

## Model References

Qwen3 model family:

- Qwen3 blog: https://qwenlm.github.io/blog/qwen3/
- Qwen3 GitHub: https://github.com/QwenLM/Qwen3
- Qwen3 Hugging Face organization: https://huggingface.co/Qwen
- Qwen3 technical report: https://arxiv.org/abs/2505.09388

Qwen3 thinking-mode sampling defaults used by the scripts:

```text
temperature = 0.6
top_p       = 0.95
top_k       = 20
min_p       = 0.0
```

These are exposed in `scripts/eval_qwen3_math.sh` and `eval/run_math_eval.py`.

## Dataset References

The repository includes scripts to download or evaluate against these external datasets. The datasets themselves are not redistributed here.

### AIME 2024

- OpenRLHF/aime-2024: https://huggingface.co/datasets/OpenRLHF/aime-2024
- HuggingFaceH4/aime_2024 alternative: https://huggingface.co/datasets/HuggingFaceH4/aime_2024

### AIME 2025

- MathArena organization: https://huggingface.co/MathArena
- MathArena/aime_2025: https://huggingface.co/datasets/MathArena/aime_2025
- test-time-compute/aime_2025 alternative: https://huggingface.co/datasets/test-time-compute/aime_2025
- MathArena paper: https://arxiv.org/abs/2605.00674

### MATH-500

- HuggingFaceH4/MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

### DAPO-Math-17k

- BytedTsinghua-SIA DAPO collection: https://huggingface.co/collections/BytedTsinghua-SIA/dapo
- BytedTsinghua-SIA/DAPO-Math-17k: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
- DAPO project page: https://dapo-sia.github.io/
