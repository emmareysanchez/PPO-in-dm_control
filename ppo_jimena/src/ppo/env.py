from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, TimeLimit

from ppo_jimena.src.ppo.reward import walker2d_reward


CROPS: tuple[float, float, float, float] = (0.15, 0.05, 0.05, 0.05)


# ── MuJoCo introspection helpers ─────────────────────────────────────────────
# All helpers use env.unwrapped so they always reach the native MuJoCo env.

def _mujoco_model_data(env: gym.Env) -> tuple[object, object] | None:
    base = env.unwrapped
    model = getattr(base, "model", None)
    data  = getattr(base, "data",  None)
    if model is None or data is None:
        return None
    return (model, data)


def _mujoco_body_id(model: object, name: str) -> int | None:
    try:
        return int(model.body(name).id)  # type: ignore[misc]
    except Exception:
        name2id = getattr(model, "name2id", None)
        if name2id is None:
            return None
        try:
            return int(name2id(name, "body"))
        except Exception:
            return None


def _body_xz(data: object, body_id: int) -> tuple[float, float] | None:
    try:
        p = data.xpos[body_id]
        return (float(p[0]), float(p[2]))
    except Exception:
        return None


def _effective_dt(env: gym.Env, action_repeat: int) -> float | None:
    md = _mujoco_model_data(env=env)
    if md is None:
        return None
    model, _ = md
    try:
        return float(model.opt.timestep) * int(action_repeat)
    except Exception:
        return None


def _geom_rot_z_axis(data: object, geom_id: int) -> np.ndarray | None:
    try:
        xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        return xmat[:, 2].copy()
    except Exception:
        return None


def _geom_center(data: object, geom_id: int) -> np.ndarray | None:
    try:
        return np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy()
    except Exception:
        return None


def _geom_size(model: object, geom_id: int) -> np.ndarray | None:
    try:
        return np.asarray(model.geom_size[geom_id], dtype=np.float64).copy()
    except Exception:
        return None


def _geom_type(model: object, geom_id: int) -> int | None:
    try:
        return int(model.geom_type[geom_id])
    except Exception:
        return None


def _geom_extreme_points(model: object, data: object, geom_id: int) -> list[np.ndarray]:
    c = _geom_center(data=data, geom_id=geom_id)
    if c is None:
        return []
    gsize = _geom_size(model=model, geom_id=geom_id)
    gtype = _geom_type(model=model, geom_id=geom_id)
    if gsize is None or gtype is None:
        return [c]

    try:
        import mujoco  # type: ignore
        mj_capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        mj_cyl     = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        mj_box     = int(mujoco.mjtGeom.mjGEOM_BOX)
        mj_sphere  = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        mj_plane   = int(mujoco.mjtGeom.mjGEOM_PLANE)
    except Exception:
        mj_capsule, mj_cyl, mj_box, mj_sphere, mj_plane = -1, -1, -1, -1, -1

    if gtype == mj_plane:
        return []
    if gtype in (mj_capsule, mj_cyl):
        axis_z = _geom_rot_z_axis(data=data, geom_id=geom_id)
        if axis_z is None:
            return [c]
        half_ext = float(gsize[1]) + (float(gsize[0]) if gtype == mj_capsule else 0.0)
        d = axis_z * half_ext
        return [c - d, c + d]
    if gtype == mj_box:
        try:
            xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        except Exception:
            return [c]
        pts: list[np.ndarray] = []
        for dx in (-gsize[0], gsize[0]):
            for dy in (-gsize[1], gsize[1]):
                for dz in (-gsize[2], gsize[2]):
                    pts.append(c + xmat @ np.array([dx, dy, dz], dtype=np.float64))
        return pts
    if gtype == mj_sphere:
        r = float(gsize[0])
        return [
            c + np.array([ r, 0., 0.], dtype=np.float64),
            c + np.array([-r, 0., 0.], dtype=np.float64),
            c + np.array([0.,  r, 0.], dtype=np.float64),
            c + np.array([0., -r, 0.], dtype=np.float64),
            c + np.array([0., 0.,  r], dtype=np.float64),
            c + np.array([0., 0., -r], dtype=np.float64),
        ]
    return [c]


def _argmax_point(points: list[np.ndarray], axis: int) -> np.ndarray | None:
    if not points:
        return None
    best = points[0]
    for p in points[1:]:
        if float(p[axis]) > float(best[axis]):
            best = p
    return best


def _argmin_point(points: list[np.ndarray], axis: int) -> np.ndarray | None:
    if not points:
        return None
    best = points[0]
    for p in points[1:]:
        if float(p[axis]) < float(best[axis]):
            best = p
    return best


def _geom_body_id(model: object, geom_id: int) -> int | None:
    try:
        return int(model.geom_bodyid[geom_id])
    except Exception:
        return None


def _body_geom_extreme_point(
    model: object, data: object, body_id: int, axis: int, maximize: bool,
) -> np.ndarray | None:
    points: list[np.ndarray] = []
    try:
        ngeom = int(model.ngeom)
    except Exception:
        ngeom = 0
    for geom_id in range(ngeom):
        if _geom_body_id(model=model, geom_id=geom_id) != body_id:
            continue
        points.extend(_geom_extreme_points(model=model, data=data, geom_id=geom_id))
    return (_argmax_point if maximize else _argmin_point)(points=points, axis=axis)


def _heel_xy(model: object, data: object, body_name: str) -> tuple[float, float] | None:
    body_id = _mujoco_body_id(model=model, name=body_name)
    if body_id is None:
        return None
    pt = _body_geom_extreme_point(model=model, data=data, body_id=body_id, axis=0, maximize=False)
    if pt is None:
        return None
    return (float(pt[0]), float(pt[2]))


def _head_xyz(model: object, data: object) -> np.ndarray | None:
    points: list[np.ndarray] = []
    try:
        ngeom = int(model.ngeom)
    except Exception:
        ngeom = 0
    for gid in range(ngeom):
        points.extend(_geom_extreme_points(model=model, data=data, geom_id=gid))
    return _argmax_point(points=points, axis=2)


_ALLOWED_DEFAULT_KEYS: tuple[str, ...] = (
    "x_position", "x_velocity", "reward_forward", "reward_ctrl", "reward_survive",
)

_BODY_KEY_MAP: dict[str, str] = {
    "torso":      "torso",
    "thigh":      "thigh_right",
    "leg":        "leg_right",
    "foot":       "foot_right",
    "thigh_left": "thigh_left",
    "leg_left":   "leg_left",
    "foot_left":  "foot_left",
}


def _build_info(
    raw_info: dict,
    env: gym.Env,
    action_repeat: int,
    terminated: bool,
    truncated: bool,
    prev_head_xy: tuple[float, float] | None,
    prev_body_xy: dict[str, tuple[float, float]],
) -> tuple[dict, tuple[float, float] | None, dict[str, tuple[float, float]]]:
    info: dict = {}

    for k in _ALLOWED_DEFAULT_KEYS:
        if k in raw_info:
            info[k] = raw_info[k]

    if "z_distance_from_origin" in raw_info:
        info["y_distance_from_origin"] = raw_info["z_distance_from_origin"]
    elif "y_distance_from_origin" in raw_info:
        info["y_distance_from_origin"] = raw_info["y_distance_from_origin"]

    info["terminated"] = bool(terminated)
    info["truncated"]  = bool(truncated)

    md = _mujoco_model_data(env=env)
    if md is None:
        return info, prev_head_xy, prev_body_xy

    model, data = md
    dt = _effective_dt(env=env, action_repeat=action_repeat)
    new_prev_body_xy = dict(prev_body_xy)

    for mujoco_name, out_name in _BODY_KEY_MAP.items():
        bid = _mujoco_body_id(model=model, name=mujoco_name)
        if bid is None:
            continue
        xz = _body_xz(data=data, body_id=bid)
        if xz is None:
            continue
        x, y = float(xz[0]), float(xz[1])
        info[f"{out_name}_x"] = x
        info[f"{out_name}_y"] = y
        if dt is not None:
            prev = prev_body_xy.get(out_name)
            info[f"{out_name}_dx"] = float((x - float(prev[0])) / dt) if prev else 0.0
            info[f"{out_name}_dy"] = float((y - float(prev[1])) / dt) if prev else 0.0
        else:
            info[f"{out_name}_dx"] = 0.0
            info[f"{out_name}_dy"] = 0.0
        new_prev_body_xy[out_name] = (x, y)
        if out_name == "torso":
            info["x_velocity"] = float(info.get("torso_dx", 0.0))

    for foot_name, heel_prefix in (("foot", "heel_right"), ("foot_left", "heel_left")):
        hxy = _heel_xy(model=model, data=data, body_name=foot_name)
        if hxy is not None:
            info[f"{heel_prefix}_x"] = float(hxy[0])
            info[f"{heel_prefix}_y"] = float(hxy[1])

    head_xyz = _head_xyz(model=model, data=data)
    if head_xyz is not None:
        head_x, head_y = float(head_xyz[0]), float(head_xyz[2])
        info["head_x"] = head_x
        info["head_y"] = head_y
        if dt is not None and prev_head_xy is not None:
            info["head_dx"] = float((head_x - float(prev_head_xy[0])) / dt)
            info["head_dy"] = float((head_y - float(prev_head_xy[1])) / dt)
        else:
            info["head_dx"] = 0.0
            info["head_dy"] = 0.0
        prev_head_xy = (head_x, head_y)

    return info, prev_head_xy, new_prev_body_xy


@dataclass
class EnvSpec:
    env_id: str
    frame_stack: int
    action_repeat: int
    time_limit: int
    obs_h: int
    obs_w: int
    grayscale: bool = False
    action_prototypes: list[list[float]] | None = None


# ── Wrappers ──────────────────────────────────────────────────────────────────

class ActionRepeatWrapper(gym.Wrapper):
    """Repeats each continuous action `repeat` times, accumulating reward."""

    def __init__(self, env: gym.Env, repeat: int) -> None:
        super().__init__(env=env)
        self.repeat = int(repeat)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        obs = None
        for _ in range(self.repeat):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_reward += float(r)
            if terminated or truncated:
                break
        assert obs is not None
        return obs, float(total_reward), bool(terminated), bool(truncated), info


RewardFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, bool, bool, dict, float, gym.Env],
    float,
]



class PixelObservationWrapper(gym.Wrapper):
    """
    Replaces state-vector obs with rendered pixels.

    Uses self.unwrapped.render() so the MuJoCo renderer is always reached
    regardless of wrapper depth.  _build_info receives `self` so it can
    call env.unwrapped internally for MuJoCo model/data access.
    """

    def __init__(
        self,
        env: gym.Env,
        height: int = 84,
        width: int = 84,
        grayscale: bool = False,
        action_repeat: int = 1,
        reward_fn: RewardFn | None = None,
    ) -> None:
        super().__init__(env=env)
        self.height        = int(height)
        self.width         = int(width)
        self.grayscale     = bool(grayscale)
        self.action_repeat = int(action_repeat)
        self._reward_fn     = reward_fn

        c_out = 1 if self.grayscale else 3
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(c_out, self.height, self.width),
            dtype=np.uint8,
        )
        self._prev_head_xy: tuple[float, float] | None = None
        self._prev_body_xy: dict[str, tuple[float, float]] = {}

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        top_f, bottom_f, left_f, right_f = CROPS
        h, w = int(frame.shape[0]), int(frame.shape[1])
        y0 = max(0, int(round(h * top_f)))
        y1 = min(h, h - max(0, int(round(h * bottom_f))))
        x0 = max(0, int(round(w * left_f)))
        x1 = min(w, w - max(0, int(round(w * right_f))))
        return frame[y0:y1, x0:x1, :]

    def _get_obs(self) -> np.ndarray:
        frame = self.unwrapped.render()  # always hits the MuJoCo renderer
        frame = self._crop_frame(frame)
        frame = cv2.resize(src=frame, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
            return gray[None, :, :].astype(np.uint8)
        return np.transpose(frame, (2, 0, 1)).astype(np.uint8)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        _obs, info = self.env.reset(**kwargs)
        self._prev_head_xy = None
        self._prev_body_xy = {}
        return self._get_obs(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        _obs, reward, terminated, truncated, raw_info = self.env.step(action)
        raw_info = dict(raw_info)
        raw_info["terminated"] = bool(terminated)
        raw_info["truncated"]  = bool(truncated)

        prev_obs = self._get_obs()
        info, self._prev_head_xy, self._prev_body_xy = _build_info(
            raw_info=raw_info,
            env=self,                        # self.unwrapped used inside
            action_repeat=self.action_repeat,
            terminated=bool(terminated),
            truncated=bool(truncated),
            prev_head_xy=self._prev_head_xy,
            prev_body_xy=self._prev_body_xy,
        )
        next_obs = self._get_obs()
        if self._reward_fn is not None:
            reward = float(self._reward_fn(
                prev_obs,
                np.asarray(action, dtype=np.float32),
                next_obs,
                bool(terminated),
                bool(truncated),
                info,
                float(reward),
                self.env,
            ))
        return next_obs, float(reward), bool(terminated), bool(truncated), info


class FrameStack(gym.Wrapper):
    """Stacks k most recent frames along the channel axis."""

    def __init__(self, env: gym.Env, k: int) -> None:
        super().__init__(env=env)
        self.k = int(k)
        self.frames: list[np.ndarray] | None = None
        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(int(c) * self.k, int(h), int(w)),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_obs(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert self.frames is not None
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self) -> np.ndarray:
        assert self.frames is not None
        return np.concatenate(self.frames, axis=0)


# ── Stack builder ─────────────────────────────────────────────────────────────

def _build_env_stack(spec: EnvSpec, seed: int, reward_fn: RewardFn | None) -> gym.Env:
    """
    Wrapper order:
      gym.make
        → ActionRepeatWrapper    (continuous actions repeated in the sim)
        → PixelObservationWrapper
        → FrameStack
        → TimeLimit              (counts agent steps, not sim steps)
    """
    env = gym.make(id=spec.env_id, render_mode="rgb_array")

    if spec.action_repeat > 1:
        env = ActionRepeatWrapper(env=env, repeat=spec.action_repeat)


    env = PixelObservationWrapper(
        env=env,
        height=int(spec.obs_h),
        width=int(spec.obs_w),
        grayscale=bool(spec.grayscale),
        action_repeat=int(spec.action_repeat),
        reward_fn=reward_fn,
    )

    if spec.frame_stack > 1:
        env = FrameStack(env=env, k=spec.frame_stack)

    env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))

    env.reset(seed=int(seed))
    return env


def make_train_env(spec: EnvSpec, seed: int) -> gym.Env:
    return _build_env_stack(spec=spec, seed=seed, reward_fn=walker2d_reward)


def make_eval_env(spec: EnvSpec, seed: int, video_dir: str) -> gym.Env:
    env = _build_env_stack(spec=spec, seed=seed, reward_fn=walker2d_reward)
    env = RecordVideo(
        env=env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix="eval",
        disable_logger=True,
    )
    return env