from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import optuna
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

from jimena.src.sac.env import EnvSpec, make_eval_env, make_train_env, resolve_env_spec


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRIAL_STEPS   = 200_000   # steps por trial
N_TRIALS      = 30        # trials totales (~20-24h con GPU)
EVAL_EPISODES = 5         # episodios de evaluación al final de cada trial
STUDY_NAME    = "sac_walker"
STORAGE       = "sqlite:///optuna_sac.db"   # persiste resultados en disco
BEST_PARAMS_PATH = Path("optuna_best_params.yaml")


# ---------------------------------------------------------------------------
# Callback silencioso — solo actualiza tqdm
# ---------------------------------------------------------------------------

class TrialProgressCallback(BaseCallback):
    def __init__(self, total_steps: int) -> None:
        super().__init__(verbose=0)
        self._pbar = tqdm(
            total=int(total_steps),
            desc="  steps",
            unit="step",
            dynamic_ncols=True,
            leave=False,
        )

    def _on_step(self) -> bool:
        self._pbar.n = int(self.num_timesteps)
        self._pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        self._pbar.close()


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, env_spec: EnvSpec, device: str, seed: int) -> float:
    # --- sample hyperparameters ---
    lr              = trial.suggest_float("lr",              1e-5, 5e-4, log=True)
    tau             = trial.suggest_float("tau",             0.001, 0.02, log=True)
    gamma           = trial.suggest_float("gamma",           0.97, 0.999)
    batch_size      = trial.suggest_categorical("batch_size",      [128, 256, 512])
    buffer_size     = trial.suggest_categorical("buffer_size",     [50_000, 100_000, 200_000])
    learning_starts = trial.suggest_categorical("learning_starts", [5_000, 10_000, 25_000])
    gradient_steps  = trial.suggest_int("gradient_steps",   1, 4)
    hidden_dim      = trial.suggest_categorical("hidden_dim",      [128, 256, 512])

    trial_seed = seed + trial.number

    train_env = make_train_env(spec=env_spec, seed=trial_seed)

    model = SAC(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=1,
        gradient_steps=gradient_steps,
        learning_starts=learning_starts,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs={
            "net_arch": [hidden_dim, hidden_dim],
            "features_extractor_kwargs": {"features_dim": 256},
        },
        device=device,
        seed=trial_seed,
        verbose=0,
    )

    callback = TrialProgressCallback(total_steps=TRIAL_STEPS)

    model.learn(
        total_timesteps=TRIAL_STEPS,
        callback=callback,
        reset_num_timesteps=True,
    )

    # --- evaluate ---
    eval_env = make_eval_env(
        spec=env_spec,
        seed=trial_seed + 999,
        video_dir=str(Path("optuna_videos") / f"trial_{trial.number}"),
    )
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    eval_env.close()
    train_env.close()

    print(
        f"  trial {trial.number:>3}  "
        f"mean_reward={float(mean_reward):>8.2f}  "
        f"std={float(std_reward):>6.2f}  "
        f"lr={lr:.2e}  tau={tau:.4f}  gamma={gamma:.4f}  "
        f"batch={batch_size}  buf={buffer_size}  "
        f"starts={learning_starts}  grad_steps={gradient_steps}  hidden={hidden_dim}"
    )

    return float(mean_reward)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_best_params(trial: optuna.Trial) -> None:
    data = {
        "best_trial":  trial.number,
        "best_reward": round(float(trial.value), 3),
        "params": {k: v for k, v in trial.params.items()},
    }
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    cfg      = load_yaml(config_path)
    seed     = int(cfg.get("experiment", {}).get("seed", 0))
    device_s = str(cfg.get("experiment", {}).get("device", "cpu"))
    device   = device_s if not (device_s == "cuda" and not torch.cuda.is_available()) else "cpu"
    env_spec = resolve_env_spec(cfg)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Silenciar logs de optuna salvo warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,          # reanuda si se interrumpe
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )

    outer_pbar = tqdm(total=N_TRIALS, desc="Trials", unit="trial", dynamic_ncols=True)

    def wrapped_objective(trial: optuna.Trial) -> float:
        result = objective(trial=trial, env_spec=env_spec, device=device, seed=seed)
        outer_pbar.update(1)
        if study.best_trial.number == trial.number:
            save_best_params(study.best_trial)
        outer_pbar.set_postfix(best=f"{study.best_value:.2f}" if study.trials else "—")
        return result

    study.optimize(wrapped_objective, n_trials=N_TRIALS, gc_after_trial=True)
    outer_pbar.close()

    # --- resultados ---
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best trial : {best.number}")
    print(f"Best reward: {best.value:.3f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print(f"\nResults saved to: {STORAGE}")
    print("Visualize with:")
    print("  optuna-dashboard sqlite:///optuna_sac.db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sac.yaml")
    args = parser.parse_args()
    main(config_path=args.config)