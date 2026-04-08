from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import SAC

from jimena.src.sac.env import make_eval_env
from jimena.src.sac.utils import resolve_env_spec, resolve_seed


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str, ckpt_path: str, episodes: int = 5) -> None:
    cfg      = load_yaml(config_path)
    seed     = resolve_seed(cfg)
    env_spec = resolve_env_spec(cfg)

    run_name  = Path(ckpt_path).parent.name
    video_dir = Path("videos") / "sac" / run_name / "manual_eval"

    env = make_eval_env(spec=env_spec, seed=seed + 999, video_dir=str(video_dir))
    model = SAC.load(ckpt_path, env=env)

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
    parser.add_argument("--config",   type=str, default="config/sac.yaml")
    parser.add_argument("--ckpt",     type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    main(config_path=str(args.config), ckpt_path=str(args.ckpt), episodes=int(args.episodes))