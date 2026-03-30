from __future__ import annotations

import argparse
import copy
import datetime as dt
from typing import Any

import optuna
import yaml

from src.ppo.train import run_training


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: YAML file path.

    Returns:
        Parsed config dictionary.
    """
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def timestamp_string() -> str:
    """Generate a filesystem-friendly timestamp.

    Returns:
        Timestamp string.
    """
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sample_hyperparameters(
    trial: optuna.Trial,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Sample PPO hyperparameters from config-defined ranges.

    Args:
        trial: Optuna trial.
        config: Search configuration.

    Returns:
        Training config with sampled PPO parameters.
    """
    sampled_config: dict[str, Any] = copy.deepcopy(config)
    search_space: dict[str, Any] = config["search_space"]

    actor_lr_cfg: dict[str, Any] = search_space["ppo"]["actor_lr"]
    critic_lr_cfg: dict[str, Any] = search_space["ppo"]["critic_lr"]
    gamma_cfg: dict[str, Any] = search_space["ppo"]["gamma"]
    lambd_cfg: dict[str, Any] = search_space["ppo"]["lambd"]
    k_epochs_cfg: dict[str, Any] = search_space["ppo"]["k_epochs"]
    eps_clip_cfg: dict[str, Any] = search_space["ppo"]["eps_clip"]
    entropy_cfg: dict[str, Any] = search_space["ppo"]["entropy_coef"]
    value_coef_cfg: dict[str, Any] = search_space["ppo"]["value_coef"]
    max_grad_norm_cfg: dict[str, Any] = search_space["ppo"]["max_grad_norm"]
    hidden_dim_cfg: dict[str, Any] = search_space["architecture"]["hidden_dim"]

    sampled_config["ppo"] = {
        "actor_lr": trial.suggest_float(
            "actor_lr",
            float(actor_lr_cfg["min"]),
            float(actor_lr_cfg["max"]),
            log=True,
        ),
        "critic_lr": trial.suggest_float(
            "critic_lr",
            float(critic_lr_cfg["min"]),
            float(critic_lr_cfg["max"]),
            log=True,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            float(gamma_cfg["min"]),
            float(gamma_cfg["max"]),
        ),
        "lambd": trial.suggest_float(
            "lambd",
            float(lambd_cfg["min"]),
            float(lambd_cfg["max"]),
        ),
        "k_epochs": trial.suggest_int(
            "k_epochs",
            int(k_epochs_cfg["min"]),
            int(k_epochs_cfg["max"]),
        ),
        "eps_clip": trial.suggest_float(
            "eps_clip",
            float(eps_clip_cfg["min"]),
            float(eps_clip_cfg["max"]),
        ),
        "entropy_coef": trial.suggest_float(
            "entropy_coef",
            float(entropy_cfg["min"]),
            float(entropy_cfg["max"]),
            log=True,
        ),
        "value_coef": trial.suggest_float(
            "value_coef",
            float(value_coef_cfg["min"]),
            float(value_coef_cfg["max"]),
        ),
        "max_grad_norm": trial.suggest_float(
            "max_grad_norm",
            float(max_grad_norm_cfg["min"]),
            float(max_grad_norm_cfg["max"]),
        ),
    }
    sampled_config["architecture"]["hidden_dim"] = trial.suggest_int(
        "hidden_dim",
        int(hidden_dim_cfg["min"]),
        int(hidden_dim_cfg["max"]),
        step=int(hidden_dim_cfg.get("step", 32)),
    )
    return sampled_config


def objective(
    trial: optuna.Trial,
    config: dict[str, Any],
    config_path: str,
    search_timestamp: str,
) -> float:
    """Execute one Optuna trial.

    Args:
        trial: Optuna trial.
        config: Base search configuration.
        config_path: Original config path.
        search_timestamp: Outer search timestamp.

    Returns:
        Final evaluation reward.
    """
    trial_config: dict[str, Any] = sample_hyperparameters(trial=trial, config=config)
    summary: dict[str, float] = run_training(
        config=trial_config,
        config_path=config_path,
        resume=None,
        new_run=True,
        trial=trial,
        search_timestamp=search_timestamp,
    )
    return float(summary["final_eval_reward"])


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    """Launch PPO hyperparameter search from the command line."""
    args: argparse.Namespace = parse_args()
    config: dict[str, Any] = load_config(config_path=args.config)
    search_timestamp: str = timestamp_string()
    study: optuna.Study = optuna.create_study(
        direction="maximize",
        study_name=str(config["experiment"]["name"]),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(
            trial=trial,
            config=config,
            config_path=args.config,
            search_timestamp=search_timestamp,
        ),
        n_trials=int(config["experiment"]["trials"]),
    )

    print("Best trial:")
    print(study.best_trial.number)
    print("Best value:")
    print(study.best_value)
    print("Best params:")
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
