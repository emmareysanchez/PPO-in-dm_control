from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from tqdm import tqdm

from ppo_jimena.src.ppo.env import EnvSpec, make_eval_env, make_train_env


# ── YAML helpers (identical to original train.py) ────────────────────────────

def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict[str, Any]) -> torch.device:
    if "device" in cfg:
        name = str(cfg["device"])
    else:
        name = str(cfg.get("experiment", {}).get("device", "cpu"))
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def resolve_seed(cfg: dict[str, Any]) -> int:
    if "seed" in cfg:
        return int(cfg["seed"])
    return int(cfg.get("experiment", {}).get("seed", 0))


def resolve_env_spec(cfg: dict[str, Any]) -> EnvSpec:
    if "env" in cfg:
        e = cfg["env"]
        return EnvSpec(
            env_id=str(e["env_id"]),
            frame_stack=int(e["frame_stack"]),
            action_repeat=int(e["action_repeat"]),
            time_limit=int(e["time_limit"]),
            action_prototypes=e["action_prototypes"],
            obs_h=int(e.get("obs_h", 84)),
            obs_w=int(e.get("obs_w", 84)),
            grayscale=bool(e.get("grayscale", False)),
        )
    e = cfg["environment"]
    env_id = e.get("env_id")
    if env_id is None:
        domain = str(e.get("domain", "")).lower()
        task = str(e.get("task", "")).lower()
        if domain == "walker" and task == "walk":
            env_id = "Walker2d-v5"
        else:
            raise ValueError("YAML must define environment.env_id or a supported domain/task pair.")
    return EnvSpec(
        env_id=str(env_id),
        frame_stack=int(e["frame_stack"]),
        action_repeat=int(e["action_repeat"]),
        time_limit=int(e.get("time_limit", 1000)),
        action_prototypes=e["action_prototypes"],
        obs_h=int(e.get("obs_h", e.get("observation_height", 84))),
        obs_w=int(e.get("obs_w", e.get("observation_width", 84))),
        grayscale=bool(e.get("grayscale", False)),
    )


def resolve_train_params(cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = cfg.get("train", {})
    logging_cfg = cfg.get("logging", {})
    return {
        "total_steps":     int(train_cfg.get("total_steps", 1_000_000)),
        "checkpoint_freq": int(train_cfg.get("checkpoint_freq", logging_cfg.get("checkpoint_freq", logging_cfg.get("save_freq", 50_000)))),
        "video_freq":      int(train_cfg.get("video_freq",      logging_cfg.get("video_freq",      logging_cfg.get("save_freq", 50_000)))),
        "num_videos":      int(train_cfg.get("num_videos",      logging_cfg.get("num_videos", 4))),
        "eval_freq":       int(train_cfg.get("eval_freq",  10_000)),
        "eval_episodes":   int(train_cfg.get("eval_episodes", 5)),
    }


# ── Callbacks ─────────────────────────────────────────────────────────────────

class CheckpointCallback(BaseCallback):
    """Saves model every `save_freq` steps to checkpoints/ppo/<run_name>/step_<N>.zip"""

    def __init__(self, save_freq: int, ckpt_dir: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.ckpt_dir = ckpt_dir

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = self.ckpt_dir / f"step_{self.num_timesteps}"
            self.model.save(str(path))
            if self.verbose:
                print(f"  checkpoint saved → {path}.zip")
        return True


class VideoCallback(BaseCallback):
    """Records num_videos rollouts every video_freq steps to videos/ppo/<run_name>/step_<N>/"""

    def __init__(
        self,
        video_freq: int,
        num_videos: int,
        env_spec: EnvSpec,
        seed: int,
        video_dir: Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.video_freq = int(video_freq)
        self.num_videos = int(num_videos)
        self.env_spec = env_spec
        self.seed = int(seed)
        self.video_dir = video_dir

    def _on_step(self) -> bool:
        if self.n_calls % self.video_freq == 0:
            step_dir = self.video_dir / f"step_{self.num_timesteps}"
            step_dir.mkdir(parents=True, exist_ok=True)
            env = make_eval_env(spec=self.env_spec, seed=self.seed + self.num_timesteps, video_dir=str(step_dir))
            with torch.no_grad():
                for _ in range(self.num_videos):
                    obs, _ = env.reset()
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _r, terminated, truncated, _info = env.step(int(action))
                        done = bool(terminated or truncated)
            env.close()
            if self.verbose:
                print(f"  videos saved → {step_dir}")
        return True


class TensorBoardCallback(BaseCallback):
    """
    Replicates the exact same TensorBoard scalars as the original train.py:
      train/episode_reward, train/episode_length, train/episode_index
      train/policy_loss,    train/value_loss,      train/entropy
      eval/episode_reward,  eval/mean_episode_reward, eval/std_episode_reward
    """

    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int,
        env_spec: EnvSpec,
        seed: int,
        video_dir: Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.env_spec = env_spec
        self.seed = int(seed)
        self.video_dir = video_dir

        self._episode_idx = 0
        self._episode_reward = 0.0
        self._episode_length = 0

    def _on_step(self) -> bool:
        self._episode_reward += float(self.locals["rewards"][0])
        self._episode_length += 1

        done = bool(self.locals["dones"][0])
        if done:
            self._episode_idx += 1
            self.logger.record("train/episode_reward", self._episode_reward)
            self.logger.record("train/episode_length", self._episode_length)
            self.logger.record("train/episode_index",  self._episode_idx)
            self.logger.dump(self.num_timesteps)
            self._episode_reward = 0.0
            self._episode_length = 0

        # Losses and entropy come from SB3's internal logger — remap to train/* keys
        for sb3_key, our_key in (
            ("train/policy_gradient_loss", "train/policy_loss"),
            ("train/value_loss",           "train/value_loss"),
            ("train/entropy_loss",         "train/entropy"),
        ):
            val = self.model.logger.name_to_value.get(sb3_key)
            if val is not None:
                self.logger.record(our_key, val)

        # Periodic deterministic eval
        if self.n_calls % self.eval_freq == 0:
            eval_video_dir = str(self.video_dir / "eval" / f"step_{self.num_timesteps}")
            eval_returns = self._run_eval(eval_video_dir)
            for i, ep_ret in enumerate(eval_returns):
                self.logger.record("eval/episode_reward", ep_ret)
                self.logger.dump(self.num_timesteps + i)
            self.logger.record("eval/mean_episode_reward", float(np.mean(eval_returns)))
            self.logger.record("eval/std_episode_reward",  float(np.std(eval_returns)))
            self.logger.dump(self.num_timesteps)

        return True

    def _run_eval(self, video_dir: str) -> list[float]:
        env = make_eval_env(spec=self.env_spec, seed=self.seed + self.num_timesteps, video_dir=video_dir)
        returns: list[float] = []
        with torch.no_grad():
            for _ in range(self.eval_episodes):
                obs, _ = env.reset()
                done = False
                total = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, _ = env.step(int(action))
                    done = bool(terminated or truncated)
                    total += float(r)
                returns.append(total)
        env.close()
        return returns


class TqdmCallback(BaseCallback):
    """Progress bar identical in style to the original train.py."""

    def __init__(self, total_steps: int) -> None:
        super().__init__()
        self.pbar = tqdm(
            total=total_steps,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            colour="green",
        )
        self._recent: list[float] = []
        self._ep_reward = 0.0
        self._ep_idx = 0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        if done:
            self._ep_idx += 1
            self._recent.append(self._ep_reward)
            if len(self._recent) > 10:
                self._recent.pop(0)
            self._ep_reward = 0.0
        self.pbar.update(1)
        if done and self._recent:
            self.pbar.set_postfix(
                ep=self._ep_idx,
                ret=f"{float(np.mean(self._recent)):.1f}",
                best=f"{max(self._recent):.1f}",
            )
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str = "configs/ppo.yaml") -> None:
    cfg = load_yaml(config_path)

    device    = resolve_device(cfg)
    seed      = resolve_seed(cfg)
    env_spec  = resolve_env_spec(cfg)
    train_params = resolve_train_params(cfg)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name  = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir   = Path("runs")        / "ppo" / run_name
    video_dir = Path("videos")      / "ppo" / run_name
    ckpt_dir  = Path("checkpoints") / "ppo" / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_train_env(spec=env_spec, seed=seed)

    sb3_cfg = cfg.get("ppo", {})
    hidden_dim = int(cfg.get("architecture", {}).get("hidden_dim", 256))

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=float(sb3_cfg.get("actor_lr", 3e-4)),
        n_steps=int(train_params.get("rollout_size", sb3_cfg.get("rollout_size", 2048))),
        batch_size=int(sb3_cfg.get("minibatch_size", 512)),
        n_epochs=int(sb3_cfg.get("k_epochs", 10)),
        gamma=float(sb3_cfg.get("gamma", 0.99)),
        gae_lambda=float(sb3_cfg.get("lambd", 0.95)),
        clip_range=float(sb3_cfg.get("eps_clip", 0.2)),
        ent_coef=float(sb3_cfg.get("entropy_coef", 0.02)),
        vf_coef=float(sb3_cfg.get("value_coef", 0.5)),
        max_grad_norm=float(sb3_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs={"net_arch": [], "features_extractor_kwargs": {"features_dim": hidden_dim}},
        tensorboard_log=str(run_dir),
        device=device,
        seed=seed,
        verbose=0,
    )

    callbacks = [
        TensorBoardCallback(
            eval_freq=train_params["eval_freq"],
            eval_episodes=train_params["eval_episodes"],
            env_spec=env_spec,
            seed=seed,
            video_dir=video_dir,
        ),
        CheckpointCallback(
            save_freq=train_params["checkpoint_freq"],
            ckpt_dir=ckpt_dir,
        ),
        VideoCallback(
            video_freq=train_params["video_freq"],
            num_videos=train_params["num_videos"],
            env_spec=env_spec,
            seed=seed,
            video_dir=video_dir,
        ),
        TqdmCallback(total_steps=train_params["total_steps"]),
    ]

    model.learn(
        total_timesteps=train_params["total_steps"],
        callback=callbacks,
        tb_log_name="ppo",
        reset_num_timesteps=True,
    )

    model.save(str(ckpt_dir / "final"))
    train_env.close()
    print(f"\nTraining complete. Run: {run_name}")
    print(f"  TensorBoard : runs/ppo/{run_name}")
    print(f"  Checkpoints : checkpoints/ppo/{run_name}/")
    print(f"  Videos      : videos/ppo/{run_name}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo.yaml")
    args = parser.parse_args()
    main(config_path=str(args.config))