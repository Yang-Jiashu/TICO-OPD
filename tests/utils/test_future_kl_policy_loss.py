import math

import torch

from slime.utils.ppo_utils import compute_future_kl_policy_loss, compute_policy_loss


def _future_loss(ppo_kl, advantages, **kwargs):
    params = {
        "future_kl_decay_rate": 32.0,
        "future_kl_start": "include_current",
        "future_kl_window": -1,
        "future_kl_average": False,
        "future_kl_clip_ratio": 0.2,
        "future_kl_clip_high_only": True,
        "future_kl_safety_threshold": 10.0,
    }
    params.update(kwargs)
    return compute_future_kl_policy_loss(
        local_ppo_kl=[ppo_kl],
        local_advantages=[advantages],
        loss_masks=[torch.ones_like(ppo_kl)],
        total_lengths=[ppo_kl.numel()],
        response_lengths=[ppo_kl.numel()],
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=3.0,
        **params,
    )


def test_future_kl_matches_vanilla_when_log_ratio_is_zero():
    ppo_kl = torch.zeros(4)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

    future_pg_loss, future_clipfrac, future_ppo_kl, metrics = _future_loss(ppo_kl, advantages)
    vanilla_pg_loss, vanilla_clipfrac = compute_policy_loss(ppo_kl, advantages, 0.2, 0.2)

    torch.testing.assert_close(future_pg_loss, vanilla_pg_loss)
    torch.testing.assert_close(future_clipfrac, vanilla_clipfrac)
    torch.testing.assert_close(future_ppo_kl, ppo_kl)
    torch.testing.assert_close(metrics["future_kl_influence_weight"], torch.ones_like(ppo_kl))


def test_future_kl_include_current_increases_positive_advantage_weight():
    ppo_kl = torch.tensor([-math.log(2.0), 0.0])
    advantages = torch.tensor([1.0, 1.0])

    future_pg_loss, _, _, metrics = _future_loss(ppo_kl, advantages)

    torch.testing.assert_close(metrics["future_kl_influence_weight"][0], torch.tensor(1.2))
    assert future_pg_loss[0] < -1.2


def test_future_kl_exclude_current_does_not_use_current_token_drift():
    ppo_kl = torch.tensor([-math.log(2.0), 0.0])
    advantages = torch.tensor([1.0, 1.0])

    _, _, _, metrics = _future_loss(ppo_kl, advantages, future_kl_start="exclude_current")

    torch.testing.assert_close(metrics["future_kl_influence_weight"], torch.ones_like(ppo_kl))
