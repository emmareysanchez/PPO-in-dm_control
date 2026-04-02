from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO

from ppo_jimena.src.ppo.env import EnvSpec, make_eval_env


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def main(config_path: str, ckpt_path: str, episodes: int = 5) -> None:
    cfg      = load_yaml(config_path)
    seed     = resolve_seed(cfg)
    env_spec = resolve_env_spec(cfg)

    run_name  = Path(ckpt_path).parent.name
    video_dir = Path("videos") / "ppo" / run_name / "manual_eval"

    env = make_eval_env(spec=env_spec, seed=seed + 999, video_dir=str(video_dir))
    model = PPO.load(ckpt_path, env=env)

    returns: list[float] = []
    for ep in range(int(episodes)):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total += float(r)
        returns.append(total)
        print(f"  episode {ep + 1}/{episodes}  return={total:.3f}")

    print(
        f"\nepisodes={episodes}  "
        f"mean={float(np.mean(returns)):.3f}  "
        f"std={float(np.std(returns)):.3f}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="config/ppo.yaml")
    parser.add_argument("--ckpt",     type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    main(config_path=str(args.config), ckpt_path=str(args.ckpt), episodes=int(args.episodes))