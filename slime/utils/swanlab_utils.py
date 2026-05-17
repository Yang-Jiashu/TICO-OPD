import logging
import os
from copy import deepcopy


logger = logging.getLogger(__name__)


def _import_swanlab():
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError("SwanLab tracking requires `pip install swanlab`.") from exc
    return swanlab


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)
    whitelist_env_vars = [
        "SLURM_JOB_ID",
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}
    return output


def _parse_tags(tags):
    if tags is None or tags == "":
        return None
    if isinstance(tags, list):
        return tags
    return [tag.strip() for tag in str(tags).split(",") if tag.strip()]


def init_swanlab_primary(args):
    if not args.use_swanlab:
        args.swanlab_run_id = None
        return

    swanlab = _import_swanlab()

    if args.swanlab_api_key is not None:
        login_kwargs = {"api_key": args.swanlab_api_key}
        if args.swanlab_host is not None:
            login_kwargs["host"] = args.swanlab_host
        swanlab.login(**login_kwargs)

    init_kwargs = {
        "project": args.swanlab_project,
        "workspace": args.swanlab_workspace,
        "experiment_name": args.swanlab_experiment_name or args.swanlab_group,
        "group": args.swanlab_group,
        "tags": _parse_tags(args.swanlab_tags),
        "config": _compute_config_for_logging(args),
        "logdir": args.swanlab_logdir,
        "mode": args.swanlab_mode,
        "id": args.swanlab_run_id,
        "resume": args.swanlab_resume,
        "public": args.swanlab_public,
    }
    if args.swanlab_mode in (None, "cloud"):
        init_kwargs["parallel"] = "shared"
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    run = swanlab.init(**init_kwargs)
    swanlab_run = getattr(swanlab, "run", None)
    args.swanlab_run_id = getattr(run, "id", None) or getattr(swanlab_run, "id", None)
    logger.info("SwanLab tracking initialized. run_id=%s", args.swanlab_run_id)


def init_swanlab_secondary(args):
    if not args.use_swanlab:
        return
    if args.swanlab_run_id is None:
        return

    swanlab = _import_swanlab()

    if args.swanlab_api_key is not None:
        login_kwargs = {"api_key": args.swanlab_api_key}
        if args.swanlab_host is not None:
            login_kwargs["host"] = args.swanlab_host
        swanlab.login(**login_kwargs)

    init_kwargs = {
        "project": args.swanlab_project,
        "workspace": args.swanlab_workspace,
        "experiment_name": args.swanlab_experiment_name or args.swanlab_group,
        "group": args.swanlab_group,
        "tags": _parse_tags(args.swanlab_tags),
        "config": args.__dict__,
        "logdir": args.swanlab_logdir,
        "mode": args.swanlab_mode,
        "id": args.swanlab_run_id,
        "resume": "allow",
        "public": args.swanlab_public,
        "reinit": True,
    }
    if args.swanlab_mode in (None, "cloud"):
        init_kwargs["parallel"] = "shared"
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    swanlab.init(**init_kwargs)


def finish_swanlab(args):
    if not args.use_swanlab:
        return
    try:
        swanlab = _import_swanlab()
        swanlab.finish()
    except Exception:
        logger.exception("Failed to finish SwanLab run")


def log_swanlab(args, metrics, step_key):
    if not args.use_swanlab:
        return
    swanlab = _import_swanlab()
    step = metrics.get(step_key)
    metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
    swanlab.log(metrics_except_step, step=step)
