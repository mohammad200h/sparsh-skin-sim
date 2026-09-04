"""Grasp-pattern and cyringe policies for data collection."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterator

import mujoco as mj
import numpy as np

from util.fk_taxel_util import LEAP_JOINT_ORDER
from util.ik_common import (
    CYRINGE_MF_RF_DIP,
    CYRINGE_MF_RF_MCP,
    CYRINGE_MF_RF_PIP,
    DEFAULT_DELTA_IK_Z,
    DEFAULT_HANDLE_SITE,
    DEFAULT_IK_REACH_TOL,
    load_ik_backend,
)
from util.motion_util import (
    DEFAULT_CLOSE_DURATION,
    GRASP_PATTERNS,
    GraspProfile,
    default_grasp_profile,
    grasp_target,
    resolve_actuator_ids,
)
from util.trajectory_generation import DEFAULT_TRAJ_STROKE, DEFAULT_TRAJ_WAYPOINTS

_PROFILE_FIELDS = frozenset(
    field.name for field in fields(GraspProfile) if field.name != "pattern"
)


def _profile_from_params(motion_type: str, params: dict) -> GraspProfile:
    if motion_type not in GRASP_PATTERNS:
        raise ValueError(
            f"Unknown grasp pattern '{motion_type}'. Expected one of {GRASP_PATTERNS}"
        )

    defaults = default_grasp_profile(motion_type)
    profile_kwargs = {
        name: getattr(defaults, name) for name in _PROFILE_FIELDS
    }
    for name in _PROFILE_FIELDS:
        if name in params:
            profile_kwargs[name] = params[name]
    return GraspProfile(pattern=motion_type, **profile_kwargs)


class Policy:
    """Open-loop grasp policy driven by ``util.motion_util.grasp_target``."""

    def __init__(self, motion_type: str, params: dict) -> None:
        if "model" not in params:
            raise ValueError("params must include 'model' (mujoco.MjModel)")

        self._model: mj.MjModel = params["model"]
        self._profile = _profile_from_params(motion_type, params)
        self._timestep = float(params.get("timestep", self._model.opt.timestep))
        self._time_offset = float(params.get("time_offset", 0.0))
        self._actuator_ids = resolve_actuator_ids(self._model, LEAP_JOINT_ORDER)
        self._step = 0

    @property
    def profile(self) -> GraspProfile:
        return self._profile

    def reset(self) -> None:
        """Restart the motion from ``time_offset``."""
        self._step = 0

    def act(self, obs: dict) -> np.ndarray:
        """Return the next joint-position target in ``LEAP_JOINT_ORDER``."""
        del obs  # open-loop for now

        time_seconds = self._time_offset + self._step * self._timestep
        ctrl = grasp_target(self._model, time_seconds, self._profile)
        self._step += 1
        return ctrl[self._actuator_ids].astype(np.float64, copy=False)


class VectorPolicy:
    """Independent open-loop policies for each vector-env slot."""

    def __init__(self, motion_type: str, params: dict, num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self._policies = [Policy(motion_type, params) for _ in range(num_envs)]

    @property
    def profile(self) -> GraspProfile:
        return self._policies[0].profile

    def reset(self, env_id: int | None = None) -> None:
        if env_id is None:
            for policy in self._policies:
                policy.reset()
            return
        self._policies[env_id].reset()

    def act(self, obs: dict | None = None) -> np.ndarray:
        del obs
        return np.stack([policy.act({}) for policy in self._policies])


@dataclass(frozen=True)
class CyringeMotionParams:
    """Tunables for ``cyringe_two_phase_motion`` (matches ``env_cyringe_demo``)."""

    ik_backend: str = "pyroki"
    close_duration: float = DEFAULT_CLOSE_DURATION
    mcp_target: float = CYRINGE_MF_RF_MCP
    pip_target: float = CYRINGE_MF_RF_PIP
    dip_target: float = CYRINGE_MF_RF_DIP
    handle_site: str = DEFAULT_HANDLE_SITE
    reach_tol: float = DEFAULT_IK_REACH_TOL
    delta_ik_z: float = DEFAULT_DELTA_IK_Z
    use_trajectory: bool = True
    traj_stroke: float | None = None
    traj_waypoints: int = DEFAULT_TRAJ_WAYPOINTS

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> CyringeMotionParams:
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


class CyringePolicy:
    """MF/RF close + thumb IK policy matching ``demos/env_cyringe_demo.py``."""

    motion_name = "cyringe_two_phase"

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        actuator_ids: np.ndarray,
        *,
        ee_site: str | None,
        goal_mocap: str | None,
        traj_mocap_names: list[str] | None,
        params: CyringeMotionParams | dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
        self._ee_site = ee_site
        self._goal_mocap = goal_mocap
        self._traj_mocap_names = traj_mocap_names
        self._params = (
            params
            if isinstance(params, CyringeMotionParams)
            else CyringeMotionParams.from_dict(params)
        )
        self._ik_mod = load_ik_backend(self._params.ik_backend)
        self._motion: Iterator[dict[str, Any]] | None = None
        self._last_info: dict[str, Any] | None = None

    @property
    def params(self) -> CyringeMotionParams:
        return self._params

    @property
    def profile(self) -> CyringeMotionParams:
        """Alias so collectors can read ``policy.profile`` like grasp policies."""
        return self._params

    @property
    def last_info(self) -> dict[str, Any] | None:
        return self._last_info

    def _motion_kwargs(self) -> dict[str, Any]:
        p = self._params
        kwargs: dict[str, Any] = {
            "close_duration": p.close_duration,
            "mcp_target": p.mcp_target,
            "pip_target": p.pip_target,
            "dip_target": p.dip_target,
            "handle_site": p.handle_site,
            "ee_site": self._ee_site,
            "reach_tol": p.reach_tol,
            "delta_ik_z": p.delta_ik_z,
            "goal_mocap": self._goal_mocap,
        }
        if p.use_trajectory:
            kwargs.update(
                traj_stroke=p.traj_stroke,
                traj_waypoints=int(p.traj_waypoints),
                traj_mocap_names=self._traj_mocap_names,
            )
        return kwargs

    def reset(self) -> None:
        """Restart two-phase motion from the current ``data`` state."""
        self._motion = self._ik_mod.cyringe_two_phase_motion(
            self._model, self._data, **self._motion_kwargs()
        )
        self._last_info = None

    def act(self, obs: dict | None = None) -> np.ndarray:
        """Advance motion (writes ``data.ctrl``) and return Leap joint targets."""
        del obs
        if self._motion is None:
            self.reset()
        assert self._motion is not None
        self._last_info = dict(next(self._motion))
        return np.asarray(
            self._data.ctrl[self._actuator_ids], dtype=np.float64
        ).copy()


class VectorCyringePolicy:
    """Independent cyringe policies for each vector-env slot."""

    motion_name = "cyringe_two_phase"

    def __init__(self, policies: list[CyringePolicy]) -> None:
        if not policies:
            raise ValueError("policies must be non-empty")
        self._policies = policies

    @property
    def params(self) -> CyringeMotionParams:
        return self._policies[0].params

    @property
    def profile(self) -> CyringeMotionParams:
        return self._policies[0].profile

    def reset(self, env_id: int | None = None) -> None:
        if env_id is None:
            for policy in self._policies:
                policy.reset()
            return
        self._policies[env_id].reset()

    def act(self, obs: dict | None = None) -> np.ndarray:
        del obs
        return np.stack([policy.act({}) for policy in self._policies])


@dataclass(frozen=True)
class RandomMotionParams:
    """Tunables for uniform random joint targets held for several steps."""

    action_hold: int = 40

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RandomMotionParams:
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "action_hold" in kwargs:
            hold = int(kwargs["action_hold"])
            if hold < 1:
                raise ValueError("policy.action_hold must be at least 1")
            kwargs["action_hold"] = hold
        return cls(**kwargs)


class RandomPolicy:
    """Open-loop random joint targets, reused for ``action_hold`` env steps.

    Matches ``demos/env_cyringe_domain_randomiztion_random_policy.py``: each
    sample is drawn uniformly in ``[low, high]`` (actuator ctrlrange) and held
    so the hand moves smoothly instead of jumping every physics step.
    """

    motion_name = "random"

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        *,
        params: RandomMotionParams | dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self._low = np.asarray(low, dtype=np.float64).reshape(-1)
        self._high = np.asarray(high, dtype=np.float64).reshape(-1)
        if self._low.shape != self._high.shape:
            raise ValueError("low and high must have the same shape")
        if np.any(self._low > self._high):
            raise ValueError("low must be <= high componentwise")
        self._params = (
            params
            if isinstance(params, RandomMotionParams)
            else RandomMotionParams.from_dict(params)
        )
        self._seed = seed
        self._n_resets = 0
        self._rng = np.random.default_rng(seed)
        self._hold_left = 0
        self._action: np.ndarray | None = None

    @property
    def params(self) -> RandomMotionParams:
        return self._params

    @property
    def profile(self) -> RandomMotionParams:
        """Alias so collectors can read ``policy.profile`` like grasp policies."""
        return self._params

    def reset(self) -> None:
        """Clear the held action and reseed from ``seed + reset_index``."""
        if self._seed is not None:
            self._rng = np.random.default_rng(self._seed + self._n_resets)
        self._n_resets += 1
        self._hold_left = 0
        self._action = None

    def act(self, obs: dict | None = None) -> np.ndarray:
        """Return the current (possibly held) joint-position target."""
        del obs
        if self._hold_left <= 0:
            self._action = self._rng.uniform(self._low, self._high)
            self._hold_left = int(self._params.action_hold)
        assert self._action is not None
        self._hold_left -= 1
        return np.asarray(self._action, dtype=np.float64).copy()


class VectorRandomPolicy:
    """Independent random policies for each vector-env slot."""

    motion_name = "random"

    def __init__(self, policies: list[RandomPolicy]) -> None:
        if not policies:
            raise ValueError("policies must be non-empty")
        self._policies = policies

    @property
    def params(self) -> RandomMotionParams:
        return self._policies[0].params

    @property
    def profile(self) -> RandomMotionParams:
        return self._policies[0].profile

    def reset(self, env_id: int | None = None) -> None:
        if env_id is None:
            for policy in self._policies:
                policy.reset()
            return
        self._policies[env_id].reset()

    def act(self, obs: dict | None = None) -> np.ndarray:
        del obs
        return np.stack([policy.act({}) for policy in self._policies])