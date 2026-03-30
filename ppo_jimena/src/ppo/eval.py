from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.ppo.env import EnvSpec, make_eval_env
from src.ppo.ppo import PPOAgent


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


def main(config_path: str, ckpt_path: str, episodes: int = 5) -> None:
    cfg = load_yaml(config_path)
    device = resolve_device(cfg)
    seed = resolve_seed(cfg)
    env_spec = resolve_env_spec(cfg)

    run_name = Path(ckpt_path).parent.name
    video_dir = Path("videos") / "ppo" / run_name / "manual_eval"

    env = make_eval_env(spec=env_spec, seed=seed + 999, video_dir=str(video_dir))

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
    agent.load(ckpt_path, map_location=device)

    returns: list[float] = []

    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done = False
        total = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            action = agent.act_deterministic(obs_t)
            obs, r, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total += float(r)

        returns.append(total)

    print(
        f"episodes={int(episodes)} "
        f"mean_return={float(np.mean(returns)):.3f} "
        f"std_return={float(np.std(returns)):.3f}"
    )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ppo.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    main(
        config_path=str(args.config),
        ckpt_path=str(args.ckpt),
        episodes=int(args.episodes),
    )