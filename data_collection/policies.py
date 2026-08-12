"""Grasp-pattern policies for data collection."""

from __future__ import annotations

from dataclasses import fields

import mujoco as mj
import numpy as np

from util.fk_taxel_util import LEAP_JOINT_ORDER
from util.motion_util import (
    GRASP_PATTERNS,
    GraspProfile,
    default_grasp_profile,
    grasp_target,
    resolve_actuator_ids,
)

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
