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
        "SLIME_DIR",
        "MEGATRON_DIR",
        "STUDENT_SIZE",
        "TEACHER_SIZE",
        "STUDENT_MODEL_ID",
        "TEACHER_MODEL_ID",
        "BASE_MODEL",
        "TEACHER_MODEL",
        "REF_LOAD",
        "SAVE_DIR",
        "TRAIN_DATA",
        "AIME24",
        "AIME25",
        "MATH500",
        "NUM_GPUS",
        "ACTOR_NUM_GPUS",
        "ROLLOUT_NUM_GPUS",
        "STUDENT_TP",
        "TEACHER_TP",
        "ROLLOUT_GPUS_PER_ENGINE",
        "MAX_TOKENS_PER_GPU",
        "USE_EXTERNAL_TEACHER",
        "TEACHER_URL",
        "NUM_ROLLOUT",
        "ROLLOUT_BATCH_SIZE",
        "N_SAMPLES_PER_PROMPT",
        "ROLLOUT_MAX_RESPONSE_LEN",
        "ROLLOUT_TEMPERATURE",
        "ROLLOUT_TOP_P",
        "ROLLOUT_TOP_K",
        "EVAL_INTERVAL",
        "N_SAMPLES_PER_EVAL_PROMPT",
        "EVAL_MAX_RESPONSE_LEN",
        "EVAL_TEMPERATURE",
        "EVAL_TOP_P",
        "EVAL_TOP_K",
        "GLOBAL_BATCH_SIZE",
        "ADVANTAGE_ESTIMATOR",
        "LR",
        "LR_DECAY_STYLE",
        "WEIGHT_DECAY",
        "OPD_KL_COEF",
        "POLICY_LOSS_TYPE",
        "FUTURE_KL_DECAY_RATE",
        "FUTURE_KL_START",
        "FUTURE_KL_WINDOW",
        "FUTURE_KL_AVERAGE",
        "FUTURE_KL_CLIP_RATIO",
        "FUTURE_KL_CLIP_HIGH_ONLY",
        "FUTURE_KL_SAFETY_THRESHOLD",
        "USE_COMPRESSION_OPD",
        "COMPRESSION_LENGTH_BUDGET",
        "COMPRESSION_LENGTH_BUDGET_RATIO",
        "COMPRESSION_ADVANTAGE_COEF",
        "COMPRESSION_EOS_COEF",
        "COMPRESSION_COVERAGE_THRESHOLD",
        "COMPRESSION_MIN_RESPONSE_LEN",
        "COMPRESSION_IMPORTANCE_DECAY_RATE",
        "COMPRESSION_IMPORTANCE_START",
        "COMPRESSION_IMPORTANCE_WINDOW",
        "COMPRESSION_IMPORTANCE_AVERAGE",
        "COMPRESSION_IMPORTANCE_TEMPERATURE",
        "COMPRESSION_REWARD_COEF",
        "COMPRESSION_MASK_LOW_IMPORTANCE_TOKENS",
        "COMPRESSION_LOW_IMPORTANCE_THRESHOLD",
        "SWANLAB_PROJECT",
        "SWANLAB_WORKSPACE",
        "SWANLAB_EXPERIMENT_NAME",
        "SWANLAB_GROUP",
        "SWANLAB_TAGS",
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
