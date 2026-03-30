from __future__ import annotations


import argparse
import random
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from ppo_jimena.src.ppo.buffer import RolloutBuffer
from ppo_jimena.src.ppo.env import EnvSpec, make_eval_env, make_train_env
from ppo_jimena.src.ppo.ppo import PPOAgent

root_path = str(Path(__file__).parent.parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

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


def resolve_train_params(cfg: dict[str, Any]) -> dict[str, int]:
    train_cfg = cfg.get("train", {})
    logging_cfg = cfg.get("logging", {})

    return {
        "total_steps": int(train_cfg.get("total_steps", 1_000_000)),
        "rollout_size": int(train_cfg.get("rollout_size", cfg.get("ppo", {}).get("rollout_size", 1024))),
        "checkpoint_freq": int(train_cfg.get("checkpoint_freq", logging_cfg.get("checkpoint_freq", logging_cfg.get("save_freq", 50_000)))),
        "video_freq": int(train_cfg.get("video_freq", logging_cfg.get("video_freq", logging_cfg.get("save_freq", 50_000)))),
        "num_videos": int(train_cfg.get("num_videos", logging_cfg.get("num_videos", 4))),
    }


def record_videos(
    agent: PPOAgent,
    spec: EnvSpec,
    seed: int,
    video_dir: Path,
    device: torch.device,
    num_videos: int,
) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    env = make_eval_env(spec=spec, seed=seed, video_dir=str(video_dir))

    for _ in range(int(num_videos)):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            action = agent.act_deterministic(obs_t)
            obs, _r, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

    env.close()


def main(config_path: str = "configs/ppo.yaml") -> None:
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

    run_dir = Path("runs") / "ppo" / run_name
    video_dir = Path("videos") / "ppo" / run_name
    ckpt_dir = Path("checkpoints") / "ppo" / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    env = make_train_env(spec=env_spec, seed=seed)

    obs_shape = env.observation_space.shape
    n_actions = int(env.action_space.n)

    ppo_cfg = dict(cfg.get("ppo", {}))
    hidden_dim = int(cfg.get("architecture", {}).get("hidden_dim", 256))
    ppo_cfg["hidden_dim"] = hidden_dim

    agent = PPOAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        device=device,
        **ppo_cfg,
    )

    buffer = RolloutBuffer(
        size=train_params["rollout_size"],
        obs_shape=obs_shape,
        device=device,
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("lambd", 0.95)),
    )

    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    episode_reward = 0.0
    episode_length = 0
    episode_idx = 0

    total_steps = train_params["total_steps"]

    for global_step in range(1, total_steps + 1):
        action, log_prob, value = agent.act(obs_t)

        next_obs, reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)

        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
        buffer.add(obs=obs_t, action=action, reward=float(reward), done=done, log_prob=log_prob, value=value)

        obs_t = next_obs_t
        episode_reward += float(reward)
        episode_length += 1

        if buffer.ptr == buffer.size:
            with torch.no_grad():
                last_value = torch.tensor(0.0, dtype=torch.float32, device=device) if done else agent.critic(obs_t.unsqueeze(0)).squeeze(0)

            advantages, returns = buffer.compute_returns_advantages(last_value=last_value)
            loss_info = agent.update(buffer=buffer, advantages=advantages, returns=returns)
            buffer.reset()

            writer.add_scalar("train/policy_loss", loss_info["policy_loss"], global_step)
            writer.add_scalar("train/value_loss", loss_info["value_loss"], global_step)
            writer.add_scalar("train/entropy", loss_info["entropy"], global_step)

        if global_step % train_params["checkpoint_freq"] == 0:
            agent.save(ckpt_dir / f"step_{global_step}.pt")

        if global_step % train_params["video_freq"] == 0:
            record_videos(
                agent=agent,
                spec=env_spec,
                seed=seed + global_step,
                video_dir=video_dir / f"step_{global_step}",
                device=device,
                num_videos=train_params["num_videos"],
            )

        if done:
            episode_idx += 1
            writer.add_scalar("train/episode_reward", episode_reward, global_step)
            writer.add_scalar("train/episode_length", episode_length, global_step)
            writer.add_scalar("train/episode_index", episode_idx, global_step)

            obs, _ = env.reset()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            episode_reward = 0.0
            episode_length = 0

    agent.save(ckpt_dir / "final.pt")

    env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo.yaml")
    args = parser.parse_args()
    main(config_path=str(args.config))