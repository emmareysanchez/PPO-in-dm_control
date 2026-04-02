from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from jimena.src.sac.env import EnvSpec, make_eval_env, make_train_env


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict[str, Any]) -> str:
    name = str(cfg.get("device") or cfg.get("experiment", {}).get("device", "cpu"))
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def resolve_seed(cfg: dict[str, Any]) -> int:
    return int(cfg.get("seed") or cfg.get("experiment", {}).get("seed", 0))


def resolve_env_spec(cfg: dict[str, Any]) -> EnvSpec:
    if "env" in cfg:
        e = cfg["env"]
        return EnvSpec(
            env_id=str(e["env_id"]),
            frame_stack=int(e["frame_stack"]),
            action_repeat=int(e.get("action_repeat", 1)),
            time_limit=int(e["time_limit"]),
            action_prototypes=e.get("action_prototypes"),
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
        action_repeat=int(e.get("action_repeat", 1)),
        time_limit=int(e.get("time_limit", 500)),
        action_prototypes=e.get("action_prototypes"),
        obs_h=int(e.get("obs_h", e.get("observation_height", 84))),
        obs_w=int(e.get("obs_w", e.get("observation_width", 84))),
        grayscale=bool(e.get("grayscale", False)),
    )


def resolve_train_params(cfg: dict[str, Any]) -> dict[str, Any]:
    t = cfg.get("train", {})
    l = cfg.get("logging", {})
    e = cfg.get("environment", cfg.get("env", {}))
    return {
        "total_steps": int(t.get("total_steps", 1_000_000)),
        # SAC uses gradient_steps per env step instead of a rollout buffer
        "gradient_steps": int(t.get("gradient_steps", 1)),
        "learning_starts": int(t.get("learning_starts", 10_000)),
        "checkpoint_freq": int(t.get("checkpoint_freq", l.get("save_freq", 50_000))),
        "video_freq": int(t.get("video_freq", l.get("save_freq", 50_000))),
        "num_videos": int(t.get("num_videos", 4)),
        "eval_freq": int(t.get("eval_freq", 10_000)),
        "eval_episodes": int(t.get("eval_episodes", 5)),
        # SAC is typically single-env; n_envs > 1 is allowed but unusual
        "n_envs": int(e.get("n_envs", 1)),
    }


class MasterCallback(BaseCallback):
    def __init__(
        self,
        writer: SummaryWriter,
        env_spec: EnvSpec,
        seed: int,
        video_dir: Path,
        ckpt_dir: Path,
        checkpoint_freq: int,
        video_freq: int,
        num_videos: int,
        eval_freq: int,
        eval_episodes: int,
        total_steps: int,
    ) -> None:
        super().__init__(verbose=0)
        self.writer = writer
        self.env_spec = env_spec
        self.seed = int(seed)
        self.video_dir = video_dir
        self.ckpt_dir = ckpt_dir
        self.checkpoint_freq = int(checkpoint_freq)
        self.video_freq = int(video_freq)
        self.num_videos = int(num_videos)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)

        self._ep_rewards: list[float] | None = None
        self._ep_lengths: list[int] | None = None
        self._ep_idx = 0
        self._recent: deque[float] = deque(maxlen=10)

        self._pbar = tqdm(
            total=int(total_steps),
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            colour="cyan",
        )

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals.get("infos", [])
        n_envs = len(rewards)

        if self._ep_rewards is None:
            self._ep_rewards = [0.0 for _ in range(n_envs)]
            self._ep_lengths = [0 for _ in range(n_envs)]

        assert self._ep_lengths is not None

        for i in range(n_envs):
            self._ep_rewards[i] += float(rewards[i])
            self._ep_lengths[i] += 1

            if bool(dones[i]):
                self._ep_idx += 1
                ep_reward = self._ep_rewards[i]
                ep_length = self._ep_lengths[i]

                episode_info = infos[i].get("episode") if i < len(infos) else None
                if isinstance(episode_info, dict):
                    ep_reward = float(episode_info.get("r", ep_reward))
                    ep_length = int(episode_info.get("l", ep_length))

                self._recent.append(ep_reward)
                self.writer.add_scalar("train/episode_reward", ep_reward, t)
                self.writer.add_scalar("train/episode_length", ep_length, t)
                self.writer.add_scalar("train/episode_index", self._ep_idx, t)

                self._ep_rewards[i] = 0.0
                self._ep_lengths[i] = 0

        if t % self.checkpoint_freq == 0:
            self.model.save(str(self.ckpt_dir / f"step_{t}"))

        if t % self.video_freq == 0:
            step_dir = self.video_dir / f"step_{t}"
            step_dir.mkdir(parents=True, exist_ok=True)
            self._record_videos(str(step_dir), seed_offset=t)

        if t % self.eval_freq == 0:
            eval_video_dir = str(self.video_dir / "eval" / f"step_{t}")
            returns = self._run_eval(eval_video_dir, seed_offset=t)
            for i, value in enumerate(returns):
                self.writer.add_scalar("eval/episode_reward", float(value), t + i)
            self.writer.add_scalar("eval/mean_episode_reward", float(np.mean(returns)), t)
            self.writer.add_scalar("eval/std_episode_reward", float(np.std(returns)), t)

        # SAC-specific losses logged by SB3
        if hasattr(self.model, "logger") and self.model.logger is not None:
            for key in [
                "train/actor_loss",
                "train/critic_loss",
                "train/ent_coef_loss",
                "train/ent_coef",
            ]:
                value = self.model.logger.name_to_value.get(key)
                if value is not None:
                    self.writer.add_scalar(key, float(value), t)

        self._pbar.n = min(t, self._pbar.total)
        self._pbar.refresh()

        if self._recent:
            self._pbar.set_postfix(
                ep=self._ep_idx,
                ret=f"{float(np.mean(self._recent)):.1f}",
                best=f"{max(self._recent):.1f}",
            )

        return True

    def _on_training_end(self) -> None:
        self._pbar.close()
        self.writer.flush()

    def _run_eval(self, video_dir: str, seed_offset: int) -> list[float]:
        env = make_eval_env(spec=self.env_spec, seed=self.seed + seed_offset, video_dir=video_dir)
        returns: list[float] = []
        with torch.no_grad():
            for _ in range(self.eval_episodes):
                obs, _ = env.reset()
                done = False
                total = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    total += float(reward)
                returns.append(total)
        env.close()
        return returns

    def _record_videos(self, video_dir: str, seed_offset: int) -> None:
        env = make_eval_env(spec=self.env_spec, seed=self.seed + seed_offset, video_dir=video_dir)
        with torch.no_grad():
            for _ in range(self.num_videos):
                obs, _ = env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
        env.close()


def main(config_path: str = "configs/sac.yaml") -> None:
    cfg = load_yaml(config_path)

    device = resolve_device(cfg)
    seed = resolve_seed(cfg)
    env_spec = resolve_env_spec(cfg)
    train_params = resolve_train_params(cfg)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / "sac" / run_name
    video_dir = Path("videos") / "sac" / run_name
    ckpt_dir = Path("checkpoints") / "sac" / run_name

    for directory in (run_dir, video_dir, ckpt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    n_envs = int(train_params["n_envs"])
    train_env = make_train_env(spec=env_spec, seed=seed, n_envs=n_envs)

    sac_cfg = cfg.get("sac", {})
    hidden_dim = int(cfg.get("architecture", {}).get("hidden_dim", 256))

    # SAC replay buffer size
    buffer_size = int(sac_cfg.get("buffer_size", 1_000_000))
    batch_size = int(sac_cfg.get("batch_size", 256))

    # Entropy temperature: "auto" lets SB3 learn it automatically
    ent_coef = sac_cfg.get("ent_coef", "auto")

    model = SAC(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=float(sac_cfg.get("lr", 3e-4)),
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=float(sac_cfg.get("tau", 0.005)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        train_freq=int(sac_cfg.get("train_freq", 1)),
        gradient_steps=int(train_params["gradient_steps"]),
        learning_starts=int(train_params["learning_starts"]),
        ent_coef=ent_coef,
        target_update_interval=int(sac_cfg.get("target_update_interval", 1)),
        policy_kwargs={
            "net_arch": [hidden_dim, hidden_dim],
            "features_extractor_kwargs": {"features_dim": hidden_dim},
        },
        tensorboard_log=None,
        device=device,
        seed=seed,
        verbose=1,
    )

    callback = MasterCallback(
        writer=writer,
        env_spec=env_spec,
        seed=seed,
        video_dir=video_dir,
        ckpt_dir=ckpt_dir,
        checkpoint_freq=train_params["checkpoint_freq"],
        video_freq=train_params["video_freq"],
        num_videos=train_params["num_videos"],
        eval_freq=train_params["eval_freq"],
        eval_episodes=train_params["eval_episodes"],
        total_steps=train_params["total_steps"],
    )

    obs = train_env.reset()
    print("Obs shape:", obs.shape)
    print("Obs min:", obs.min(), "Obs max:", obs.max())
    print("Action space:", train_env.action_space)

    sample_action = np.array([train_env.action_space.sample() for _ in range(train_env.num_envs)])
    obs, rewards, dones, infos = train_env.step(sample_action)
    print("Sample rewards:", rewards)
    train_env.reset()

    model.learn(
        total_timesteps=int(train_params["total_steps"]),
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name=".",
    )

    model.save(str(ckpt_dir / "final"))
    train_env.close()
    writer.close()

    print(f"\nTraining complete — run: {run_name}")
    print(f"  TensorBoard : tensorboard --logdir runs/sac/{run_name}")
    print(f"  Checkpoints : checkpoints/sac/{run_name}/")
    print(f"  Videos      : videos/sac/{run_name}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sac.yaml")
    args = parser.parse_args()
    main(config_path=str(args.config))