from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from jimena.src.sac.env import EnvSpec, make_eval_env, make_train_env, resolve_env_spec


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict[str, Any]) -> str:
    device = str(cfg.get("experiment", {}).get("device", "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def resolve_seed(cfg: dict[str, Any]) -> int:
    return int(cfg.get("experiment", {}).get("seed", 0))


class SaveEvalCallback(BaseCallback):
    def __init__(
        self,
        env_spec: EnvSpec,
        seed: int,
        ckpt_dir: Path,
        video_dir: Path,
        checkpoint_freq: int,
        eval_freq: int,
        eval_episodes: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.env_spec = env_spec
        self.seed = int(seed)
        self.ckpt_dir = ckpt_dir
        self.video_dir = video_dir
        self.checkpoint_freq = int(checkpoint_freq)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)

        if self.checkpoint_freq > 0 and step % self.checkpoint_freq == 0:
            save_path = self.ckpt_dir / f"step_{step}"
            self.model.save(str(save_path))
            if self.verbose:
                print(f"[checkpoint] saved: {save_path}.zip")

        if self.eval_freq > 0 and step % self.eval_freq == 0:
            eval_env = make_eval_env(
                spec=self.env_spec,
                seed=self.seed + step,
                video_dir=str(self.video_dir / f"step_{step}"),
            )
            mean_reward, std_reward = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
            )
            eval_env.close()
            if self.verbose:
                print(
                    f"[eval] step={step}  "
                    f"mean_reward={float(mean_reward):.2f}  "
                    f"std_reward={float(std_reward):.2f}"
                )

        return True


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)
    seed = resolve_seed(cfg)
    device = resolve_device(cfg)
    env_spec = resolve_env_spec(cfg)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir  = Path("runs")         / "sac" / run_name
    ckpt_dir = Path("checkpoints")  / "sac" / run_name
    video_dir = Path("videos")      / "sac" / run_name
    for d in (run_dir, ckpt_dir, video_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_env = make_train_env(spec=env_spec, seed=seed)

    sac_cfg   = cfg["sac"]
    train_cfg = cfg["train"]
    arch_cfg  = cfg.get("architecture", {})

    hidden_dim = int(arch_cfg.get("hidden_dim", 256))
    model = SAC(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=float(sac_cfg.get("lr", 1e-4)),
        buffer_size=int(sac_cfg.get("buffer_size", 100_000)),
        batch_size=int(sac_cfg.get("batch_size", 256)),
        tau=float(sac_cfg.get("tau", 0.005)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        train_freq=int(sac_cfg.get("train_freq", 1)),
        gradient_steps=int(train_cfg.get("gradient_steps", 1)),
        learning_starts=int(train_cfg.get("learning_starts", 25_000)),
        ent_coef=sac_cfg.get("ent_coef", "auto"),
        target_update_interval=int(sac_cfg.get("target_update_interval", 1)),
        policy_kwargs={
            "net_arch": [hidden_dim, hidden_dim],
            "features_extractor_kwargs": {"features_dim": 256},
        },
        tensorboard_log=str(run_dir),
        device=device,
        seed=seed,
        verbose=1,
    )

    callback = SaveEvalCallback(
        env_spec=env_spec,
        seed=seed,
        ckpt_dir=ckpt_dir,
        video_dir=video_dir,
        checkpoint_freq=int(train_cfg.get("checkpoint_freq", 200_000)),
        eval_freq=int(train_cfg.get("eval_freq", 100_000)),
        eval_episodes=int(train_cfg.get("eval_episodes", 3)),
        verbose=1,
    )

    model.learn(
        total_timesteps=int(train_cfg.get("total_steps", 1_500_000)),
        callback=callback,
        tb_log_name="sac",
        reset_num_timesteps=True,
    )

    final_path = ckpt_dir / "final"
    model.save(str(final_path))
    train_env.close()

    print("\nTraining complete")
    print(f"TensorBoard : tensorboard --logdir {run_dir}")
    print(f"Final model : {final_path}.zip")
    print(f"Checkpoints : {ckpt_dir}")
    print(f"Videos      : {video_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sac.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
