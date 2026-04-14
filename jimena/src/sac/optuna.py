from __future__ import annotations

import argparse
import json
import os
import random
import time
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

TRIAL_STEPS = 2_000_000  # steps por trial
VIDEO_FREQ = 250_000  # grabar vídeo de evaluación cada N steps
N_TRIALS = 30  # trials totales (~20-24h con GPU)
EVAL_EPISODES = 5  # episodios de evaluación al final de cada trial
STUDY_NAME = "sac_walker_v4"
STORAGE = "sqlite:///optuna_sac_v4.db"  # persiste resultados en disco
BEST_PARAMS_PATH = Path("optuna_best_params_v4.yaml")


# ---------------------------------------------------------------------------
# Callback: progreso + vídeos periódicos de evaluación
# ---------------------------------------------------------------------------


class TrialProgressCallback(BaseCallback):
    def __init__(
        self,
        total_steps: int,
        env_spec: EnvSpec,
        trial_seed: int,
        video_base_dir: Path,
        video_freq: int,
        eval_episodes: int,
    ) -> None:
        super().__init__(verbose=0)
        self.env_spec = env_spec
        self.trial_seed = trial_seed
        self.video_base_dir = video_base_dir
        self.video_freq = int(video_freq)
        self.eval_episodes = int(eval_episodes)
        self._next_video_at = int(video_freq)
        self.eval_log: list[dict] = []
        self._pbar = tqdm(
            total=int(total_steps),
            desc="  steps",
            unit="step",
            dynamic_ncols=True,
            leave=False,
        )

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        self._pbar.n = step
        self._pbar.refresh()

        if step >= self._next_video_at:
            step_str = str(self._next_video_at).zfill(len(str(TRIAL_STEPS)))
            video_dir = self.video_base_dir / step_str
            eval_env = make_eval_env(
                spec=self.env_spec,
                seed=self.trial_seed + self._next_video_at,
                video_dir=str(video_dir),
            )
            mean_reward, std_reward = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
            )
            eval_env.close()
            self.eval_log.append({
                "step": self._next_video_at,
                "mean_reward": round(float(mean_reward), 3),
                "std_reward": round(float(std_reward), 3),
            })
            self._next_video_at += self.video_freq

        return True

    def _on_training_end(self) -> None:
        self._pbar.close()


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.Trial, env_spec: EnvSpec, device: str, seed: int, run_name: str
) -> float:
    # --- sample hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    buffer_size = trial.suggest_categorical(
        "buffer_size", [300_000, 500_000, 750_000, 1_000_000]
    )
    learning_starts = trial.suggest_categorical(
        "learning_starts", [50_000, 75_000, 100_000, 125_000]
    )

    trial_seed = seed + trial.number

    train_env = make_train_env(spec=env_spec, seed=trial_seed)
    batch_size = 128
    tau = 0.002
    gamma = 0.99
    gradient_steps = 1

    model = SAC(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=2,
        gradient_steps=gradient_steps,
        learning_starts=learning_starts,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs={
            "net_arch": [256, 256],
            "share_features_extractor": False,
            "normalize_images": False,
            "features_extractor_kwargs": {"features_dim": 512},
        },
        device=device,
        seed=trial_seed,
        verbose=0,
    )

    trial_zfill = len(str(N_TRIALS))
    trial_str = str(trial.number).zfill(trial_zfill)
    trial_video_dir = Path("optuna_videos") / run_name / f"trial_{trial_str}"
    run_dir = Path("optuna_videos") / run_name

    callback = TrialProgressCallback(
        total_steps=TRIAL_STEPS,
        env_spec=env_spec,
        trial_seed=trial_seed,
        video_base_dir=trial_video_dir,
        video_freq=VIDEO_FREQ,
        eval_episodes=EVAL_EPISODES,
    )

    model.learn(
        total_timesteps=TRIAL_STEPS,
        callback=callback,
        reset_num_timesteps=True,
    )

    # --- evaluate final (step TRIAL_STEPS) ---
    eval_env = make_eval_env(
        spec=env_spec,
        seed=trial_seed + 999,
        video_dir=str(trial_video_dir / str(TRIAL_STEPS).zfill(len(str(TRIAL_STEPS)))),
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

    # append final step to eval log and write JSON
    callback.eval_log.append({
        "step": TRIAL_STEPS,
        "mean_reward": round(float(mean_reward), 3),
        "std_reward": round(float(std_reward), 3),
    })
    results = {
        "trial": trial.number,
        "params": {
            "lr": lr,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "batch_size": batch_size,
            "tau": tau,
            "gamma": gamma,
            "gradient_steps": gradient_steps,
        },
        "eval": callback.eval_log,
    }
    results_path = run_dir / f"results_{trial_str}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"  trial {trial.number:>3}  "
        f"mean_reward={float(mean_reward):>8.2f}  "
        f"std={float(std_reward):>6.2f}  "
        f"lr={lr:.2e}  tau={tau:.4f}  gamma={gamma:.4f}  "
        f"batch={batch_size}  buf={buffer_size}  "
        f"starts={learning_starts}  grad_steps={gradient_steps}  hidden={512}"
    )

    return float(mean_reward)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def save_best_params(trial: optuna.Trial) -> None:
    data = {
        "best_trial": trial.number,
        "best_reward": round(float(trial.value), 3),
        "params": {k: v for k, v in trial.params.items()},
    }
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    device_s = str(cfg.get("experiment", {}).get("device", "cpu"))
    device = (
        device_s
        if not (device_s == "cuda" and not torch.cuda.is_available())
        else "cpu"
    )
    env_spec = resolve_env_spec(cfg)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    (Path("optuna_videos") / run_name).mkdir(parents=True, exist_ok=True)

    # Silenciar logs de optuna salvo warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,  # reanuda si se interrumpe
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )

    outer_pbar = tqdm(total=N_TRIALS, desc="Trials", unit="trial", dynamic_ncols=True)

    def wrapped_objective(trial: optuna.Trial) -> float:
        result = objective(
            trial=trial, env_spec=env_spec, device=device, seed=seed, run_name=run_name
        )
        outer_pbar.update(1)
        try:
            best = study.best_trial
            if best.number == trial.number:
                save_best_params(best)
            outer_pbar.set_postfix(best=f"{study.best_value:.2f}")
        except ValueError:
            pass  # primer trial aun no registrado como best
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
    print("  optuna-dashboard sqlite:///optuna_sac_v4.db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sac.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
