#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


ENV_MAP = {
    "paths": {
        "slime_dir": "SLIME_DIR",
        "megatron_dir": "MEGATRON_DIR",
        "save_dir": "SAVE_DIR",
    },
    "models": {
        "student_size": "STUDENT_SIZE",
        "teacher_size": "TEACHER_SIZE",
        "base_model": "BASE_MODEL",
        "teacher_model": "TEACHER_MODEL",
        "ref_load": "REF_LOAD",
    },
    "data": {
        "train_data": "TRAIN_DATA",
        "aime24": "AIME24",
        "aime25": "AIME25",
        "math500": "MATH500",
    },
    "resources": {
        "num_gpus": "NUM_GPUS",
        "actor_num_gpus": "ACTOR_NUM_GPUS",
        "rollout_num_gpus": "ROLLOUT_NUM_GPUS",
        "student_tp": "STUDENT_TP",
        "teacher_tp": "TEACHER_TP",
        "rollout_gpus_per_engine": "ROLLOUT_GPUS_PER_ENGINE",
        "max_tokens_per_gpu": "MAX_TOKENS_PER_GPU",
        "teacher_cuda_visible_devices": "TEACHER_CUDA_VISIBLE_DEVICES",
    },
    "teacher_server": {
        "use_external_teacher": "USE_EXTERNAL_TEACHER",
        "teacher_ip": "TEACHER_IP",
        "teacher_port": "TEACHER_PORT",
        "teacher_url": "TEACHER_URL",
        "teacher_mem_fraction_static": "TEACHER_MEM_FRACTION_STATIC",
        "teacher_chunked_prefill_size": "TEACHER_CHUNKED_PREFILL_SIZE",
    },
    "rollout": {
        "num_rollout": "NUM_ROLLOUT",
        "rollout_batch_size": "ROLLOUT_BATCH_SIZE",
        "n_samples_per_prompt": "N_SAMPLES_PER_PROMPT",
        "rollout_max_response_len": "ROLLOUT_MAX_RESPONSE_LEN",
        "rollout_temperature": "ROLLOUT_TEMPERATURE",
        "rollout_top_p": "ROLLOUT_TOP_P",
        "rollout_top_k": "ROLLOUT_TOP_K",
        "sglang_mem_fraction_static": "SGLANG_MEM_FRACTION_STATIC",
    },
    "eval": {
        "eval_interval": "EVAL_INTERVAL",
        "n_samples_per_eval_prompt": "N_SAMPLES_PER_EVAL_PROMPT",
        "eval_max_response_len": "EVAL_MAX_RESPONSE_LEN",
        "eval_temperature": "EVAL_TEMPERATURE",
        "eval_top_p": "EVAL_TOP_P",
        "eval_top_k": "EVAL_TOP_K",
    },
    "training": {
        "save_interval": "SAVE_INTERVAL",
        "global_batch_size": "GLOBAL_BATCH_SIZE",
        "advantage_estimator": "ADVANTAGE_ESTIMATOR",
        "eps_clip_c": "EPS_CLIP_C",
        "optimizer": "OPTIMIZER",
        "lr": "LR",
        "lr_decay_style": "LR_DECAY_STYLE",
        "weight_decay": "WEIGHT_DECAY",
        "adam_beta1": "ADAM_BETA1",
        "adam_beta2": "ADAM_BETA2",
    },
    "tracking": {
        "use_swanlab": "USE_SWANLAB",
        "swanlab_mode": "SWANLAB_MODE",
        "swanlab_project": "SWANLAB_PROJECT",
        "swanlab_workspace": "SWANLAB_WORKSPACE",
        "swanlab_experiment_name": "SWANLAB_EXPERIMENT_NAME",
        "swanlab_group": "SWANLAB_GROUP",
        "swanlab_tags": "SWANLAB_TAGS",
        "swanlab_logdir": "SWANLAB_LOGDIR",
        "swanlab_api_key": "SWANLAB_API_KEY",
        "swanlab_host": "SWANLAB_HOST",
        "swanlab_run_id": "SWANLAB_RUN_ID",
        "swanlab_resume": "SWANLAB_RESUME",
        "swanlab_public": "SWANLAB_PUBLIC",
    },
    "tico": {
        "opd_kl_coef": "OPD_KL_COEF",
        "policy_loss_type": "POLICY_LOSS_TYPE",
        "future_kl_decay_rate": "FUTURE_KL_DECAY_RATE",
        "future_kl_start": "FUTURE_KL_START",
        "future_kl_window": "FUTURE_KL_WINDOW",
        "future_kl_average": "FUTURE_KL_AVERAGE",
        "future_kl_clip_ratio": "FUTURE_KL_CLIP_RATIO",
        "future_kl_clip_high_only": "FUTURE_KL_CLIP_HIGH_ONLY",
        "future_kl_safety_threshold": "FUTURE_KL_SAFETY_THRESHOLD",
        "use_compression_opd": "USE_COMPRESSION_OPD",
        "compression_length_budget": "COMPRESSION_LENGTH_BUDGET",
        "compression_length_budget_ratio": "COMPRESSION_LENGTH_BUDGET_RATIO",
        "compression_advantage_coef": "COMPRESSION_ADVANTAGE_COEF",
        "compression_eos_coef": "COMPRESSION_EOS_COEF",
        "compression_coverage_threshold": "COMPRESSION_COVERAGE_THRESHOLD",
        "compression_min_response_len": "COMPRESSION_MIN_RESPONSE_LEN",
        "compression_importance_decay_rate": "COMPRESSION_IMPORTANCE_DECAY_RATE",
        "compression_importance_start": "COMPRESSION_IMPORTANCE_START",
        "compression_importance_window": "COMPRESSION_IMPORTANCE_WINDOW",
        "compression_importance_average": "COMPRESSION_IMPORTANCE_AVERAGE",
        "compression_importance_temperature": "COMPRESSION_IMPORTANCE_TEMPERATURE",
        "compression_reward_coef": "COMPRESSION_REWARD_COEF",
        "compression_mask_low_importance_tokens": "COMPRESSION_MASK_LOW_IMPORTANCE_TOKENS",
        "compression_low_importance_threshold": "COMPRESSION_LOW_IMPORTANCE_THRESHOLD",
    },
}


def env_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def resolve_path(value, repo_root):
    if value in (None, ""):
        return value
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3 TICO-OPD from a YAML config.")
    parser.add_argument("--config", default="configs/qwen3/tico_opd_4b_32b.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print exported environment and exit.")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Install PyYAML first: pip install pyyaml") from exc

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = yaml.safe_load(config_path.read_text()) or {}

    env = os.environ.copy()
    for section, key_map in ENV_MAP.items():
        values = config.get(section, {}) or {}
        for key, env_name in key_map.items():
            if key not in values or values[key] is None:
                continue
            value = values[key]
            if section in {"paths", "data"}:
                value = resolve_path(value, repo_root)
            env[env_name] = env_value(value)

    env.setdefault("SLIME_DIR", str(repo_root))

    if args.dry_run:
        for section, key_map in ENV_MAP.items():
            print(f"[{section}]")
            for env_name in key_map.values():
                if env_name in env:
                    print(f"{env_name}={env[env_name]}")
        return

    script = repo_root / "scripts" / "run_qwen3_tico_opd.sh"
    subprocess.run(["bash", str(script)], env=env, check=True)


if __name__ == "__main__":
    main()
