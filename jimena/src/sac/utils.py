from __future__ import annotations

from typing import Any

from jimena.src.sac.env import EnvSpec


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