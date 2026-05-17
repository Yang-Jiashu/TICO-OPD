import logging

import wandb

from . import wandb_utils
from . import swanlab_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
        swanlab_utils.init_swanlab_primary(args)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)
        swanlab_utils.init_swanlab_secondary(args)


def update_tracking_open_metrics(args, router_addr):
    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    if args.use_wandb:
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            logging.getLogger(__name__).exception("Failed to finish wandb run")
    swanlab_utils.finish_swanlab(args)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_swanlab:
        swanlab_utils.log_swanlab(args, metrics, step_key)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
