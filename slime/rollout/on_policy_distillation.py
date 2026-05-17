import aiohttp
import torch

from slime.utils.ppo_utils import compute_future_token_importance
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


def _fit_response_log_probs(log_probs: torch.Tensor, response_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if log_probs.numel() >= response_length:
        return log_probs[-response_length:], torch.ones(response_length, dtype=torch.float32)
    if response_length == 0:
        empty = log_probs[0:0]
        return empty, empty

    missing = response_length - log_probs.numel()
    fitted = torch.nn.functional.pad(log_probs, (missing, 0), value=0.0)
    valid_mask = torch.nn.functional.pad(torch.ones_like(log_probs, dtype=torch.float32), (missing, 0), value=0.0)
    return fitted, valid_mask


async def reward_func(args, sample, **kwargs):
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_fit_results = [
        _fit_response_log_probs(t_log_prob, response_length)
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]
    teacher_log_probs = [fit_result[0] for fit_result in teacher_fit_results]
    teacher_log_prob_masks = [fit_result[1] for fit_result in teacher_fit_results]

    for sample, t_log_probs, t_log_prob_mask in zip(samples, teacher_log_probs, teacher_log_prob_masks, strict=False):
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        sample.loss_mask = [
            int(mask) * int(valid)
            for mask, valid in zip(sample.loss_mask, t_log_prob_mask.tolist(), strict=False)
        ]
        sample.teacher_log_probs = t_log_probs

    compression_penalties = [0.0] * len(samples)
    if getattr(args, "use_compression_opd", False):
        for idx, (sample, t_log_probs) in enumerate(zip(samples, teacher_log_probs, strict=False)):
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            loss_mask = torch.tensor(sample.loss_mask, dtype=torch.float32)
            # A cheap rollout-time proxy: teacher surprise accumulated into the future.
            # The train-time path uses teacher/student divergence when available.
            token_signal = torch.clamp(-t_log_probs.float(), min=0.0)
            importance = compute_future_token_importance(
                token_signal,
                loss_mask,
                decay_rate=getattr(args, "compression_importance_decay_rate", 32.0),
                start=getattr(args, "compression_importance_start", "include_current"),
                window=getattr(args, "compression_importance_window", -1),
                average=getattr(args, "compression_importance_average", False),
                temperature=getattr(args, "compression_importance_temperature", 1.0),
            )

            if getattr(args, "compression_length_budget", -1) > 0:
                budget = min(args.compression_length_budget, sample.response_length)
            elif getattr(args, "compression_length_budget_ratio", 0.0) > 0:
                budget = max(
                    1,
                    min(sample.response_length, int(sample.response_length * args.compression_length_budget_ratio)),
                )
            else:
                budget = sample.response_length

            positions = torch.arange(sample.response_length)
            total_importance = torch.clamp_min((importance * loss_mask).sum(), 1e-6)
            coverage = torch.cumsum(importance * loss_mask, dim=0) / total_importance
            compressible = (positions >= budget) | (coverage >= getattr(args, "compression_coverage_threshold", 0.9))
            if getattr(args, "compression_min_response_len", 0) > 0:
                compressible = compressible & (positions >= args.compression_min_response_len)

            low_importance = importance <= getattr(args, "compression_low_importance_threshold", 0.2)
            low_importance_compressible = low_importance & compressible & loss_mask.bool()

            reward_coef = getattr(args, "compression_reward_coef", 0.0)
            if reward_coef != 0.0 and sample.response_length > 0:
                compression_penalties[idx] = reward_coef * low_importance_compressible.float().sum().item() / sample.response_length

            if getattr(args, "compression_mask_low_importance_tokens", False):
                sample.loss_mask = [
                    0 if drop else int(mask)
                    for mask, drop in zip(sample.loss_mask, low_importance_compressible.tolist(), strict=False)
                ]

            sample.metadata = sample.metadata or {}
            sample.metadata["compression_importance_mean"] = float((importance * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1.0))
            sample.metadata["compression_low_importance_tokens"] = int(low_importance_compressible.float().sum().item())
            sample.metadata["compression_low_importance_ratio"] = float(
                low_importance_compressible.float().sum() / torch.clamp_min(loss_mask.sum(), 1.0)
            )
            sample.metadata["compression_zone_ratio"] = float(
                (compressible & loss_mask.bool()).float().sum() / torch.clamp_min(loss_mask.sum(), 1.0)
            )
            sample.metadata["compression_budget"] = int(budget)
            sample.metadata["compression_coverage_final"] = float(coverage[-1]) if coverage.numel() else 0.0
            sample.metadata["teacher_logprob_mean"] = float((t_log_probs * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1.0))
            sample.metadata["opd_valid_token_ratio"] = float(loss_mask.sum() / max(sample.response_length, 1))

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [
        float((sample.metadata or {}).get("base_reward", 0.0)) - penalty
        for sample, penalty in zip(samples, compression_penalties, strict=False)
    ]

    return scalar_rewards, scalar_rewards
