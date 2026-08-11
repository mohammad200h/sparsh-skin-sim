"""Hand motion helpers: action generators that write into ``data.ctrl``."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Iterable

import mujoco as mj
import numpy as np

# Index / middle / ring flexion only — no rot or thumb joints.
FINGER_CLOSE_JOINTS: tuple[str, ...] = (
    "if_mcp",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_pip",
    "rf_dip",
)

# Comfortable closed pose within Leap MCP / PIP / DIP ctrl ranges.
DEFAULT_CLOSE_DURATION = 1.0
DEFAULT_MCP_TARGET = 1.6
DEFAULT_PIP_TARGET = 1.5
DEFAULT_DIP_TARGET = 0.9


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _actuator_id(model: mj.MjModel, joint_name: str) -> int:
    act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_act")
    if act_id < 0:
        raise ValueError(f"Actuator '{joint_name}_act' not found")
    return int(act_id)


def resolve_actuator_ids(
    model: mj.MjModel, joint_names: Iterable[str]
) -> np.ndarray:
    return np.asarray(
        [_actuator_id(model, name) for name in joint_names], dtype=np.int32
    )


def finger_close_generator(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    duration: float = DEFAULT_CLOSE_DURATION,
    mcp_target: float = DEFAULT_MCP_TARGET,
    pip_target: float = DEFAULT_PIP_TARGET,
    dip_target: float = DEFAULT_DIP_TARGET,
    joint_names: Iterable[str] = FINGER_CLOSE_JOINTS,
    hold: bool = True,
) -> Iterator[None]:
    """Yield forever; each ``next()`` writes a finger-close command to ``data.ctrl``.

    Only MCP / PIP / DIP of IF, MF, and RF are driven. Thumb and rot are left
    untouched so whatever is already in ``data.ctrl`` for those actuators stays.

    Call once per physics step, immediately before ``mj.mj_step``::

        motion = finger_close_generator(model, data)
        ...
        next(motion)
        mj.mj_step(model, data)

    Parameters
    ----------
    duration
        Seconds to ramp from the start pose to the closed targets (smoothstep).
    mcp_target, pip_target, dip_target
        Absolute position targets for ``*_mcp`` / ``*_pip`` / ``*_dip`` actuators.
    hold
        If True, stay at the closed pose after ``duration``. If False, freeze
        targets at the last commanded values once the ramp ends (same effect
        here since targets are absolute).
    """
    names = tuple(joint_names)
    act_ids = resolve_actuator_ids(model, names)
    lo = model.actuator_ctrlrange[act_ids, 0].copy()
    hi = model.actuator_ctrlrange[act_ids, 1].copy()

    targets = np.empty(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        if name.endswith("_mcp"):
            targets[i] = mcp_target
        elif name.endswith("_pip"):
            targets[i] = pip_target
        elif name.endswith("_dip"):
            targets[i] = dip_target
        else:
            raise ValueError(
                f"finger_close_generator only supports mcp/pip/dip joints, got '{name}'"
            )
    targets = np.clip(targets, lo, hi)

    # Snapshot open pose at generator creation (usually zeros / current ctrl).
    start = data.ctrl[act_ids].copy()
    t0 = float(data.time)
    duration = max(float(duration), 1e-6)

    while True:
        elapsed = float(data.time) - t0
        if elapsed < duration:
            alpha = _smoothstep(elapsed / duration)
            data.ctrl[act_ids] = start + alpha * (targets - start)
        elif hold:
            data.ctrl[act_ids] = targets
        # else: leave ctrl as last commanded values
        yield
