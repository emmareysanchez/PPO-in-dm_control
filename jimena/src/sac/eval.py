from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import SAC

from env import make_eval_env, resolve_env_spec


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_seed(cfg: dict[str, Any]) -> int:
    return int(cfg.get("experiment", {}).get("seed", 0))


def main(config_path: str, ckpt_path: str, episodes: int) -> None:
    cfg      = load_yaml(config_path)
    seed     = resolve_seed(cfg)
    env_spec = resolve_env_spec(cfg)

    run_name  = Path(ckpt_path).stem
    video_dir = Path("videos") / "manual_eval" / run_name

    env   = make_eval_env(spec=env_spec, seed=seed + 999, video_dir=str(video_dir))
    model = SAC.load(ckpt_path, env=env)

    returns: list[float] = []
    for ep in range(int(episodes)):
        obs, _ = env.reset()
        done   = False
        total  = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done   = bool(terminated or truncated)
            total += float(reward)

        returns.append(total)
        print(f"episode {ep + 1}/{episodes}  return={total:.3f}")

    print(
        f"\nmean_return={float(np.mean(returns)):.3f}  "
        f"std_return={float(np.std(returns)):.3f}"
    )
    print(f"videos saved to: {video_dir}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="sac.yaml")
    parser.add_argument("--ckpt",     type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    main(config_path=args.config, ckpt_path=args.ckpt, episodes=args.episodes)
