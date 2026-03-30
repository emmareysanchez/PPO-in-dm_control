from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, TimeLimit

from src.ppo.reward import walker2d_reward


CROPS: tuple[float, float, float, float] = (0.25, 0.05, 0.1, 0.1)


def _mujoco_model_data(env: gym.Env) -> tuple[object, object] | None:
    base: gym.Env = env.unwrapped
    model: object | None = getattr(base, "model", None)
    data: object | None = getattr(base, "data", None)
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


def _effective_dt(env: gym.Env, info: dict) -> float | None:
    md = _mujoco_model_data(env=env)
    if md is None:
        return None
    model, _ = md
    try:
        base_dt = float(model.opt.timestep)
    except Exception:
        return None
    repeat = int(info.get("action_repeat", 1))
    return float(base_dt * repeat)


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

        mj_capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)  # type: ignore[attr-defined]
        mj_cyl = int(mujoco.mjtGeom.mjGEOM_CYLINDER)  # type: ignore[attr-defined]
        mj_box = int(mujoco.mjtGeom.mjGEOM_BOX)  # type: ignore[attr-defined]
        mj_sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)  # type: ignore[attr-defined]
        mj_plane = int(mujoco.mjtGeom.mjGEOM_PLANE)  # type: ignore[attr-defined]
    except Exception:
        mj_capsule, mj_cyl, mj_box, mj_sphere, mj_plane = -1, -1, -1, -1, -1

    if gtype == mj_plane:
        return []

    if gtype == mj_capsule or gtype == mj_cyl:
        axis_z = _geom_rot_z_axis(data=data, geom_id=geom_id)
        if axis_z is None:
            return [c]

        radius = float(gsize[0])
        half_len = float(gsize[1])
        half_extent = float(half_len + (radius if gtype == mj_capsule else 0.0))
        d = axis_z * half_extent
        return [c - d, c + d]

    if gtype == mj_box:
        try:
            xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        except Exception:
            return [c]

        sx = float(gsize[0])
        sy = float(gsize[1])
        sz = float(gsize[2])

        pts: list[np.ndarray] = []
        for dx in (-sx, sx):
            for dy in (-sy, sy):
                for dz in (-sz, sz):
                    local = np.array([dx, dy, dz], dtype=np.float64)
                    pts.append(c + xmat @ local)
        return pts

    if gtype == mj_sphere:
        r = float(gsize[0])
        return [
            c + np.array([r, 0.0, 0.0], dtype=np.float64),
            c + np.array([-r, 0.0, 0.0], dtype=np.float64),
            c + np.array([0.0, r, 0.0], dtype=np.float64),
            c + np.array([0.0, -r, 0.0], dtype=np.float64),
            c + np.array([0.0, 0.0, r], dtype=np.float64),
            c + np.array([0.0, 0.0, -r], dtype=np.float64),
        ]

    return [c]


def _argmax_point(points: list[np.ndarray], axis: int) -> np.ndarray | None:
    if not points:
        return None
    best = points[0]
    best_v = float(best[axis])
    for p in points[1:]:
        v = float(p[axis])
        if v > best_v:
            best = p
            best_v = v
    return best


def _argmin_point(points: list[np.ndarray], axis: int) -> np.ndarray | None:
    if not points:
        return None
    best = points[0]
    best_v = float(best[axis])
    for p in points[1:]:
        v = float(p[axis])
        if v < best_v:
            best = p
            best_v = v
    return best


def _geom_body_id(model: object, geom_id: int) -> int | None:
    try:
        return int(model.geom_bodyid[geom_id])
    except Exception:
        return None


def _body_geom_extreme_point(
    model: object,
    data: object,
    body_id: int,
    axis: int,
    maximize: bool,
) -> np.ndarray | None:
    points: list[np.ndarray] = []

    try:
        ngeom = int(model.ngeom)
    except Exception:
        ngeom = 0

    for geom_id in range(ngeom):
        geom_body_id = _geom_body_id(model=model, geom_id=geom_id)
        if geom_body_id != body_id:
            continue
        points.extend(_geom_extreme_points(model=model, data=data, geom_id=geom_id))

    if maximize:
        return _argmax_point(points=points, axis=axis)
    return _argmin_point(points=points, axis=axis)


def _heel_xy(model: object, data: object, body_name: str) -> tuple[float, float] | None:
    body_id = _mujoco_body_id(model=model, name=body_name)
    if body_id is None:
        return None

    heel_xyz = _body_geom_extreme_point(
        model=model,
        data=data,
        body_id=body_id,
        axis=0,
        maximize=False,
    )
    if heel_xyz is None:
        return None

    return (float(heel_xyz[0]), float(heel_xyz[2]))


def _head_xyz(model: object, data: object) -> np.ndarray | None:
    head_candidates: list[np.ndarray] = []
    try:
        ngeom = int(model.ngeom)
    except Exception:
        ngeom = 0

    for gid in range(ngeom):
        head_candidates.extend(_geom_extreme_points(model=model, data=data, geom_id=gid))
    return _argmax_point(points=head_candidates, axis=2)


_ALLOWED_DEFAULT_KEYS: tuple[str, ...] = (
    "x_position",
    "x_velocity",
    "reward_forward",
    "reward_ctrl",
    "reward_survive",
)

_BODY_KEY_MAP: dict[str, str] = {
    "torso": "torso",
    "thigh": "thigh_right",
    "leg": "leg_right",
    "foot": "foot_right",
    "thigh_left": "thigh_left",
    "leg_left": "leg_left",
    "foot_left": "foot_left",
}


def _build_info(
    raw_info: dict,
    env: gym.Env,
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
    info["truncated"] = bool(truncated)

    md = _mujoco_model_data(env=env)
    if md is None:
        return info, prev_head_xy, prev_body_xy

    model, data = md
    dt = _effective_dt(env=env, info=raw_info)

    new_prev_body_xy = dict(prev_body_xy)

    for mujoco_name, out_name in _BODY_KEY_MAP.items():
        bid = _mujoco_body_id(model=model, name=mujoco_name)
        if bid is None:
            continue

        xz = _body_xz(data=data, body_id=bid)
        if xz is None:
            continue

        x = float(xz[0])
        y = float(xz[1])

        info[f"{out_name}_x"] = x
        info[f"{out_name}_y"] = y

        if dt is not None:
            prev_xy = prev_body_xy.get(out_name)
            if prev_xy is not None:
                info[f"{out_name}_dx"] = float((x - float(prev_xy[0])) / dt)
                info[f"{out_name}_dy"] = float((y - float(prev_xy[1])) / dt)
            else:
                info[f"{out_name}_dx"] = 0.0
                info[f"{out_name}_dy"] = 0.0
        else:
            info[f"{out_name}_dx"] = 0.0
            info[f"{out_name}_dy"] = 0.0

        new_prev_body_xy[out_name] = (x, y)

        if out_name == "torso":
            info["x_velocity"] = float(info.get("torso_dx", 0.0))

    for foot_body_name, heel_prefix in (
        ("foot", "heel_right"),
        ("foot_left", "heel_left"),
    ):
        heel_xy = _heel_xy(
            model=model,
            data=data,
            body_name=foot_body_name,
        )
        if heel_xy is None:
            continue

        info[f"{heel_prefix}_x"] = float(heel_xy[0])
        info[f"{heel_prefix}_y"] = float(heel_xy[1])

    head_xyz = _head_xyz(model=model, data=data)
    if head_xyz is not None:
        head_x = float(head_xyz[0])
        head_y = float(head_xyz[2])

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
    action_prototypes: list[list[float]]
    obs_h: int
    obs_w: int
    grayscale: bool = False


class PixelObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        height: int = 84,
        width: int = 84,
        grayscale: bool = False,
    ) -> None:
        super().__init__(env=env)
        self.height = int(height)
        self.width = int(width)
        self.grayscale = bool(grayscale)

        c_out = 1 if self.grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c_out, self.height, self.width),
            dtype=np.uint8,
        )

        self._prev_head_xy: tuple[float, float] | None = None
        self._prev_body_xy: dict[str, tuple[float, float]] = {}

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        top_f, bottom_f, left_f, right_f = CROPS

        h = int(frame.shape[0])
        w = int(frame.shape[1])

        top_px = int(round(float(h) * float(top_f)))
        bottom_px = int(round(float(h) * float(bottom_f)))
        left_px = int(round(float(w) * float(left_f)))
        right_px = int(round(float(w) * float(right_f)))

        y0 = max(0, top_px)
        y1 = min(h, h - max(0, bottom_px))
        x0 = max(0, left_px)
        x1 = min(w, w - max(0, right_px))

        return frame[y0:y1, x0:x1, :]

    def _get_obs(self) -> np.ndarray:
        frame = self.env.render()
        frame = self._crop_frame(frame=frame)
        frame = cv2.resize(
            src=frame,
            dsize=(self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

        if self.grayscale:
            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
            return gray[None, :, :].astype(np.uint8)

        return np.transpose(frame, (2, 0, 1)).astype(np.uint8)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        _obs, info = self.env.reset(**kwargs)
        self._prev_head_xy = None
        self._prev_body_xy = {}
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        _obs, reward, terminated, truncated, raw_info = self.env.step(action=action)

        raw_info = dict(raw_info)
        raw_info["terminated"] = bool(terminated)
        raw_info["truncated"] = bool(truncated)

        info, self._prev_head_xy, self._prev_body_xy = _build_info(
            raw_info=raw_info,
            env=self.env,
            terminated=bool(terminated),
            truncated=bool(truncated),
            prev_head_xy=self._prev_head_xy,
            prev_body_xy=self._prev_body_xy,
        )

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int) -> None:
        super().__init__(env=env)
        self.k = int(k)
        self.frames: list[np.ndarray] | None = None

        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(int(c) * self.k, int(h), int(w)),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action=action)
        assert self.frames is not None
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self) -> np.ndarray:
        assert self.frames is not None
        return np.concatenate(self.frames, axis=0)


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat: int) -> None:
        super().__init__(env=env)
        self.repeat = int(repeat)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        obs: np.ndarray | None = None

        for _ in range(self.repeat):
            obs, r, term, trunc, info = self.env.step(action=action)
            total_reward += float(r)
            terminated = terminated or bool(term)
            truncated = truncated or bool(trunc)
            if terminated or truncated:
                break

        assert obs is not None
        info = dict(info)
        info["action_repeat"] = int(self.repeat)
        return obs, float(total_reward), bool(terminated), bool(truncated), info


RewardFn = Callable[
    [np.ndarray, int, np.ndarray, bool, bool, dict, float, gym.Env],
    float,
]


class DiscreteActionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        prototypes: np.ndarray,
        reward_fn: RewardFn | None = None,
    ) -> None:
        super().__init__(env=env)

        self.prototypes = prototypes.astype(np.float32)
        self.action_space = spaces.Discrete(n=len(self.prototypes))

        self._reward_fn = reward_fn
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before step().")

        cont_action = self.prototypes[int(action)]

        next_obs, env_reward, terminated, truncated, info = self.env.step(action=cont_action)

        if self._reward_fn is None:
            self._last_obs = next_obs
            return next_obs, float(env_reward), bool(terminated), bool(truncated), info

        new_reward = float(
            self._reward_fn(
                self._last_obs,
                int(action),
                next_obs,
                bool(terminated),
                bool(truncated),
                info,
                float(env_reward),
                self.env,
            )
        )

        self._last_obs = next_obs
        return next_obs, float(new_reward), bool(terminated), bool(truncated), info


def make_env(spec: EnvSpec, seed: int) -> gym.Env:
    env = gym.make(id=spec.env_id, render_mode="rgb_array")
    env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))
    env.reset(seed=int(seed))

    if spec.action_repeat > 1:
        env = ActionRepeat(env=env, repeat=spec.action_repeat)

    env = PixelObservationWrapper(
        env=env,
        height=int(spec.obs_h),
        width=int(spec.obs_w),
        grayscale=bool(spec.grayscale),
    )

    prototypes = np.array(spec.action_prototypes, dtype=np.float32)
    cont_dim = int(env.action_space.shape[0])

    if prototypes.shape[1] != cont_dim:
        raise ValueError(
            "action_prototypes dim mismatch: "
            f"got {prototypes.shape[1]} but env action dim is {cont_dim}"
        )

    env = DiscreteActionWrapper(
        env=env,
        prototypes=prototypes,
        reward_fn=walker2d_reward,
    )

    if spec.frame_stack > 1:
        env = FrameStack(env=env, k=spec.frame_stack)

    return env


def make_train_env(spec: EnvSpec, seed: int) -> gym.Env:
    return make_env(spec=spec, seed=seed)


def make_eval_env(spec: EnvSpec, seed: int, video_dir: str) -> gym.Env:
    env = gym.make(id=spec.env_id, render_mode="rgb_array")
    env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))
    env.reset(seed=int(seed))

    env = RecordVideo(
        env=env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix="eval",
        disable_logger=True,
    )

    if spec.action_repeat > 1:
        env = ActionRepeat(env=env, repeat=spec.action_repeat)

    env = PixelObservationWrapper(
        env=env,
        height=int(spec.obs_h),
        width=int(spec.obs_w),
        grayscale=bool(spec.grayscale),
    )

    prototypes = np.array(spec.action_prototypes, dtype=np.float32)
    cont_dim = int(env.action_space.shape[0])

    if prototypes.shape[1] != cont_dim:
        raise ValueError(
            "action_prototypes dim mismatch: "
            f"got {prototypes.shape[1]} but env action dim is {cont_dim}"
        )

    env = DiscreteActionWrapper(
        env=env,
        prototypes=prototypes,
        reward_fn=walker2d_reward,
    )

    if spec.frame_stack > 1:
        env = FrameStack(env=env, k=spec.frame_stack)

    return env