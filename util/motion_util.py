"""Hand motion helpers: action generators that write into ``data.ctrl``."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Iterable, Protocol

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


def _finger_close_closed_targets(
    names: tuple[str, ...],
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    mcp_target: float,
    pip_target: float,
    dip_target: float,
) -> np.ndarray:
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
                f"finger_close only supports mcp/pip/dip joints, got '{name}'"
            )
    return np.clip(targets, lo, hi)


def _finger_close_ramped_values(
    start: np.ndarray,
    targets: np.ndarray,
    elapsed: float,
    duration: float,
    *,
    hold: bool,
) -> np.ndarray:
    duration = max(float(duration), 1e-6)
    if elapsed < duration:
        alpha = _smoothstep(elapsed / duration)
        return start + alpha * (targets - start)
    if hold:
        return targets
    return targets


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

    targets = _finger_close_closed_targets(
        names,
        lo,
        hi,
        mcp_target=mcp_target,
        pip_target=pip_target,
        dip_target=dip_target,
    )

    # Snapshot open pose at generator creation (usually zeros / current ctrl).
    start = data.ctrl[act_ids].copy()
    t0 = float(data.time)
    duration = max(float(duration), 1e-6)

    while True:
        elapsed = float(data.time) - t0
        if elapsed < duration or hold:
            data.ctrl[act_ids] = _finger_close_ramped_values(
                start, targets, elapsed, duration, hold=hold
            )
        yield


# ---------------------------------------------------------------------------
# Grasp-pattern generators (actuator position targets over time)
# ---------------------------------------------------------------------------

GRASP_PATTERNS = (
    "hold",
    "pulse",
    "squeeze",
    "regrasp",
    "tap",
    "shear",
    "finger_close",
)

# Match ``LeapFlexEnv`` reset for the undriven thumb axial joint.
FINGER_CLOSE_TH_AXL = 1.6

CLOSE_START = 0.0
CLOSE_END = 1.5
PREGRIP_FRACTION = 0.40
GRIP_FRACTION = 0.85
THUMB_GRIP_FRACTION = 0.35
THUMB_DELAY = 0.5
PULSE_HZ = 1.0
PULSE_AMPLITUDE = 0.15
SHEAR_AMPLITUDE = 0.15
SQUEEZE_STEPS = 4
SQUEEZE_STEP_SECONDS = 0.9
REGRASP_HZ = 0.35
TAP_HZ = 1.5
TAP_DEPTH = 0.5
SHEAR_HZ = 0.7


class ActuatorView(Protocol):
    name: str


class ActuatorModel(Protocol):
    nu: int
    actuator_ctrlrange: np.ndarray

    def actuator(self, actuator_id: int) -> ActuatorView:
        ...


@dataclass(frozen=True)
class GraspProfile:
    """Parameters controlling one grasp pattern."""

    pattern: str
    grip_fraction: float
    thumb_grip_fraction: float
    pregrip_fraction: float
    thumb_delay: float
    pulse_hz: float
    pulse_amplitude: float
    shear_amplitude: float
    squeeze_steps: int
    close_duration: float = DEFAULT_CLOSE_DURATION
    mcp_target: float = DEFAULT_MCP_TARGET
    pip_target: float = DEFAULT_PIP_TARGET
    dip_target: float = DEFAULT_DIP_TARGET
    hold_after_close: bool = True


def default_grasp_profile(pattern: str) -> GraspProfile:
    """Return the profile used by ``grasp_touch_test.py``."""
    profile = GraspProfile(
        pattern=pattern,
        grip_fraction=GRIP_FRACTION,
        thumb_grip_fraction=THUMB_GRIP_FRACTION,
        pregrip_fraction=PREGRIP_FRACTION,
        thumb_delay=THUMB_DELAY,
        pulse_hz=PULSE_HZ,
        pulse_amplitude=PULSE_AMPLITUDE,
        shear_amplitude=SHEAR_AMPLITUDE,
        squeeze_steps=SQUEEZE_STEPS,
        close_duration=DEFAULT_CLOSE_DURATION,
        mcp_target=DEFAULT_MCP_TARGET,
        pip_target=DEFAULT_PIP_TARGET,
        dip_target=DEFAULT_DIP_TARGET,
        hold_after_close=True,
    )
    validate_profile(profile)
    return profile


def validate_profile(profile: GraspProfile) -> None:
    """Validate a profile before generating actuator targets."""
    if profile.pattern not in GRASP_PATTERNS:
        raise ValueError(
            f"Unknown grasp pattern '{profile.pattern}'. "
            f"Expected one of {GRASP_PATTERNS}"
        )
    for name, value in (
        ("grip_fraction", profile.grip_fraction),
        ("thumb_grip_fraction", profile.thumb_grip_fraction),
        ("pregrip_fraction", profile.pregrip_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if profile.thumb_delay < 0.0:
        raise ValueError("thumb_delay must be non-negative")
    if profile.pulse_hz < 0.0:
        raise ValueError("pulse_hz must be non-negative")
    if profile.pulse_amplitude < 0.0:
        raise ValueError("pulse_amplitude must be non-negative")
    if profile.shear_amplitude < 0.0:
        raise ValueError("shear_amplitude must be non-negative")
    if profile.squeeze_steps < 1:
        raise ValueError("squeeze_steps must be at least 1")
    if profile.close_duration <= 0.0:
        raise ValueError("close_duration must be positive")


def pregrip_targets(
    model: ActuatorModel, pregrip_fraction: float
) -> np.ndarray:
    """Return the initial curled-hand actuator targets."""
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    return low + pregrip_fraction * (high - low)


def thumb_mask(model: ActuatorModel) -> np.ndarray:
    """Identify thumb actuators from the LeapXELA actuator names."""
    return np.asarray(
        [
            model.actuator(actuator_id).name.startswith("th_")
            for actuator_id in range(model.nu)
        ],
        dtype=bool,
    )


def lateral_mask(model: ActuatorModel) -> np.ndarray:
    """Identify abduction and axial-roll actuators used by ``shear``."""
    names = [
        model.actuator(actuator_id).name
        for actuator_id in range(model.nu)
    ]
    return np.asarray(
        ["_rot_" in name or "_axl_" in name for name in names],
        dtype=bool,
    )


def finger_close_grasp_target(
    model: mj.MjModel,
    time_seconds: float,
    profile: GraspProfile,
    *,
    joint_names: Iterable[str] = FINGER_CLOSE_JOINTS,
    start_ctrl: np.ndarray | None = None,
) -> np.ndarray:
    """Return a full actuator target vector for the ``finger_close`` pattern."""
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]

    if start_ctrl is None:
        targets = np.zeros(model.nu, dtype=np.float64)
        th_axl_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "th_axl_act")
        if th_axl_id >= 0:
            targets[th_axl_id] = np.clip(
                FINGER_CLOSE_TH_AXL, low[th_axl_id], high[th_axl_id]
            )
    else:
        targets = np.asarray(start_ctrl, dtype=np.float64).copy()

    names = tuple(joint_names)
    act_ids = resolve_actuator_ids(model, names)
    lo = model.actuator_ctrlrange[act_ids, 0]
    hi = model.actuator_ctrlrange[act_ids, 1]
    closed = _finger_close_closed_targets(
        names,
        lo,
        hi,
        mcp_target=profile.mcp_target,
        pip_target=profile.pip_target,
        dip_target=profile.dip_target,
    )
    start = targets[act_ids].copy()
    targets[act_ids] = _finger_close_ramped_values(
        start,
        closed,
        time_seconds,
        profile.close_duration,
        hold=profile.hold_after_close,
    )
    return np.clip(targets, low, high)


def grasp_target(
    model: ActuatorModel, time_seconds: float, profile: GraspProfile
) -> np.ndarray:
    """Return one actuator target vector for the requested simulation time."""
    validate_profile(profile)

    if profile.pattern == "finger_close":
        if not isinstance(model, mj.MjModel):
            raise TypeError("finger_close requires a mujoco.MjModel")
        return finger_close_grasp_target(model, time_seconds, profile)

    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    span = high - low
    is_thumb = thumb_mask(model)
    grip = low + np.where(
        is_thumb, profile.thumb_grip_fraction, profile.grip_fraction
    ) * span
    pregrip = pregrip_targets(model, profile.pregrip_fraction)

    close_span = CLOSE_END - CLOSE_START
    finger_phase = np.clip(
        (time_seconds - CLOSE_START) / close_span, 0.0, 1.0
    )
    thumb_phase = np.clip(
        (time_seconds - CLOSE_START - profile.thumb_delay) / close_span,
        0.0,
        1.0,
    )
    phase = np.where(is_thumb, thumb_phase, finger_phase)
    smooth_phase = phase * phase * (3.0 - 2.0 * phase)
    targets = pregrip + smooth_phase * (grip - pregrip)

    if time_seconds < CLOSE_END + profile.thumb_delay:
        return np.clip(targets, low, high)

    elapsed = time_seconds - CLOSE_END - profile.thumb_delay
    if profile.pattern == "hold":
        modulation = 0.0
    elif profile.pattern == "pulse":
        modulation = 0.5 * profile.pulse_amplitude * (
            1.0 - np.cos(2.0 * np.pi * profile.pulse_hz * elapsed)
        )
    elif profile.pattern == "squeeze":
        cycle = profile.squeeze_steps * 2
        index = int(elapsed / SQUEEZE_STEP_SECONDS) % cycle
        level = index if index < profile.squeeze_steps else cycle - index - 1
        modulation = profile.pulse_amplitude * level / max(
            profile.squeeze_steps - 1, 1
        )
    elif profile.pattern == "regrasp":
        release = 0.5 * (
            1.0 - np.cos(2.0 * np.pi * REGRASP_HZ * elapsed)
        )
        return np.clip(
            targets + release * (pregrip - targets),
            low,
            high,
        )
    elif profile.pattern == "tap":
        release = 0.5 * (1.0 - np.cos(2.0 * np.pi * TAP_HZ * elapsed))
        return np.clip(
            targets + TAP_DEPTH * release * (pregrip - targets),
            low,
            high,
        )
    elif profile.pattern == "shear":
        lateral = profile.shear_amplitude * np.sin(
            2.0 * np.pi * SHEAR_HZ * elapsed
        )
        return np.clip(
            targets + np.where(lateral_mask(model), lateral * span, 0.0),
            low,
            high,
        )
    else:
        raise ValueError(f"Unknown grasp pattern '{profile.pattern}'")

    return np.clip(
        targets + np.where(is_thumb, 0.0, modulation * span),
        low,
        high,
    )


def generate_grasp_pattern(
    model: ActuatorModel, times: np.ndarray, profile: GraspProfile
) -> np.ndarray:
    """Generate an ``(len(times), model.nu)`` actuator-target trajectory."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError(f"times must be one-dimensional, got shape {times.shape}")
    if not np.all(np.isfinite(times)):
        raise ValueError("times contains NaN or infinite values")
    if np.any(times < 0.0):
        raise ValueError("times must be non-negative")

    return np.asarray(
        [grasp_target(model, float(time), profile) for time in times],
        dtype=np.float64,
    )


def sample_times(duration: float, sample_hz: float) -> np.ndarray:
    """Return uniformly sampled times including zero."""
    if duration < 0.0:
        raise ValueError("duration must be non-negative")
    if sample_hz <= 0.0:
        raise ValueError("sample_hz must be positive")
    sample_count = int(np.floor(duration * sample_hz)) + 1
    return np.arange(sample_count, dtype=np.float64) / sample_hz
