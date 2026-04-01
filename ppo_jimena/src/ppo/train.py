from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path
from typing import Any
import os
os.environ["MUJOCO_GL"] = "egl" 

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ppo_jimena.src.ppo.env import EnvSpec, make_eval_env, make_train_env

# ── YAML helpers ──────────────────────────────────────────────────────────────

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
            action_repeat=int(e["action_repeat"]),
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
        task   = str(e.get("task",   "")).lower()
        if domain == "walker" and task == "walk":
            env_id = "Walker2d-v5"
        else:
            raise ValueError("YAML must define environment.env_id or a supported domain/task pair.")
    return EnvSpec(
        env_id=str(env_id),
        frame_stack=int(e["frame_stack"]),
        action_repeat=int(e["action_repeat"]),
        time_limit=int(e.get("time_limit", 500)),
        action_prototypes=e.get("action_prototypes"),
        obs_h=int(e.get("obs_h", e.get("observation_height", 84))),
        obs_w=int(e.get("obs_w", e.get("observation_width",  84))),
        grayscale=bool(e.get("grayscale", False)),
    )


def resolve_train_params(cfg: dict[str, Any]) -> dict[str, Any]:
    t = cfg.get("train", {})
    l = cfg.get("logging", {})
    return {
        "total_steps":     int(t.get("total_steps",     2_000_000)),
        "rollout_size":    int(t.get("rollout_size",     2_048)),
        "checkpoint_freq": int(t.get("checkpoint_freq", l.get("save_freq", 50_000))),
        "video_freq":      int(t.get("video_freq",      l.get("save_freq", 50_000))),
        "num_videos":      int(t.get("num_videos",      4)),
        "eval_freq":       int(t.get("eval_freq",       10_000)),
        "eval_episodes":   int(t.get("eval_episodes",   5)),
    }


# ── Callbacks ─────────────────────────────────────────────────────────────────

class MasterCallback(BaseCallback):
    """
    Single callback that handles everything:
      - TensorBoard logging (episode reward, length, losses, entropy, eval)
      - Checkpoints
      - Video recording
      - tqdm progress bar

    Uses a dedicated SummaryWriter so logging is not subject to SB3's
    internal logger timing, which only flushes losses every n_steps.
    Losses are read from model.logger after each PPO update (when they exist).
    """

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
        device: str,
    ) -> None:
        super().__init__(verbose=0)
        self.writer          = writer
        self.env_spec        = env_spec
        self.seed            = seed
        self.video_dir       = video_dir
        self.ckpt_dir        = ckpt_dir
        self.checkpoint_freq = int(checkpoint_freq)
        self.video_freq      = int(video_freq)
        self.num_videos      = int(num_videos)
        self.eval_freq       = int(eval_freq)
        self.eval_episodes   = int(eval_episodes)
        self.device          = device

        self._ep_reward  = 0.0
        self._ep_length  = 0
        self._ep_idx     = 0
        self._recent: deque[float] = deque(maxlen=10)

        self._pbar = tqdm(
            total=total_steps,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            colour="green",
        )

    def _on_step(self) -> bool:
        t = self.num_timesteps

        reward = float(self.locals["rewards"][0])
        done   = bool(self.locals["dones"][0])

        self._ep_reward += reward
        self._ep_length += 1

        # ── Episode end ──
        if done:
            self._ep_idx += 1
            self._recent.append(self._ep_reward)
            self.writer.add_scalar("train/episode_reward", self._ep_reward, t)
            self.writer.add_scalar("train/episode_length", self._ep_length, t)
            self.writer.add_scalar("train/episode_index",  self._ep_idx,    t)
            self._ep_reward = 0.0
            self._ep_length = 0

        # ── Checkpoint ──
        if t % self.checkpoint_freq == 0:
            path = str(self.ckpt_dir / f"step_{t}")
            self.model.save(path)

        # ── Video ──
        if t % self.video_freq == 0:
            step_dir = self.video_dir / f"step_{t}"
            step_dir.mkdir(parents=True, exist_ok=True)
            self._record_videos(str(step_dir), seed_offset=t)

        # ── Eval ──
        if t % self.eval_freq == 0:
            eval_video_dir = str(self.video_dir / "eval" / f"step_{t}")
            returns = self._run_eval(eval_video_dir, seed_offset=t)
            for i, r in enumerate(returns):
                self.writer.add_scalar("eval/episode_reward", r, t + i)
            self.writer.add_scalar("eval/mean_episode_reward", float(np.mean(returns)), t)
            self.writer.add_scalar("eval/std_episode_reward",  float(np.std(returns)),  t)
        
        # ── PPO Losses ──  ← AÑADIR AQUÍ
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            for key in ['train/policy_gradient_loss', 'train/value_loss', 
                        'train/entropy_loss', 'train/approx_kl']:
                val = self.model.logger.name_to_value.get(key)
                if val is not None:
                    self.writer.add_scalar(key, val, t)

        # ── Progress bar ──
        self._pbar.update(1)
        if done and self._recent:
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
                    obs, r, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    total += float(r)
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
                    obs, _r, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
        env.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str = "configs/ppo.yaml") -> None:
    cfg = load_yaml(config_path)

    device       = resolve_device(cfg)
    seed         = resolve_seed(cfg)
    env_spec     = resolve_env_spec(cfg)
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

    for d in (run_dir, video_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    train_env = make_train_env(spec=env_spec, seed=seed)

    ppo_cfg    = cfg.get("ppo", {})
    hidden_dim = int(cfg.get("architecture", {}).get("hidden_dim", 256))

    n_steps    = train_params["rollout_size"]
    batch_size = int(ppo_cfg.get("minibatch_size", 512))
    # SB3 requires batch_size <= n_steps
    if batch_size > n_steps:
        batch_size = n_steps

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=float(ppo_cfg.get("actor_lr", 3e-5)),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(ppo_cfg.get("k_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("lambd", 0.95)),
        clip_range=float(ppo_cfg.get("eps_clip", 0.2)),
        ent_coef=float(ppo_cfg.get("entropy_coef", 0.02)),
        vf_coef=float(ppo_cfg.get("value_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs={
            "net_arch": [256, 256],
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
        device=device,
    )

    obs, _ = train_env.reset()
    print("Obs shape:", obs.shape)
    print("Obs min:", obs.min(), "Obs max:", obs.max())
    print("Action space:", train_env.action_space)

    # test recompensa con política aleatoria
    total_r = 0
    for _ in range(200):
        action = train_env.action_space.sample()
        obs, r, term, trunc, info = train_env.step(action)
        total_r += r
        if term or trunc:
            break
    print("Total reward random policy:", total_r)
    train_env.reset()

    model.learn(
        total_timesteps=train_params["total_steps"],
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name=".",
    )

    model.save(str(ckpt_dir / "final"))
    train_env.close()
    writer.close()

    print(f"\nTraining complete — run: {run_name}")
    print(f"  TensorBoard : tensorboard --logdir runs/ppo/{run_name}")
    print(f"  Checkpoints : checkpoints/ppo/{run_name}/")
    print(f"  Videos      : videos/ppo/{run_name}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo.yaml")
    args = parser.parse_args()
    main(config_path=str(args.config))