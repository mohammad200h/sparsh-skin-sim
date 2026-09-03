"""Hand-pose and cyringe-pose domain randomization for ``LeapFlexCyringeEnv``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from util.fk_taxel_util import LEAP_JOINT_ORDER

N_HAND_DOFS = len(LEAP_JOINT_ORDER)
DEFAULT_OFFSET_LOW = (-0.02, -0.02, -0.01)
DEFAULT_OFFSET_HIGH = (0.02, 0.02, 0.01)
DEFAULT_EULER_DELTA_LOW = (0.0, 0.0, -0.4)
DEFAULT_EULER_DELTA_HIGH = (0.0, 0.0, 0.4)


def _is_xyz_triplet(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _as_xyz(raw: Any, key: str) -> tuple[float, float, float]:
    if not _is_xyz_triplet(raw):
        raise ValueError(f"domain_randomization.{key} must be [x, y, z]")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _as_bool(raw: Any, key: str, default: bool) -> bool:
    if raw is None:
        return default
    if not isinstance(raw, bool):
        raise ValueError(f"domain_randomization.{key} must be a bool")
    return raw


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    section = raw.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"domain_randomization.{key} must be a mapping")
    return section


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = out.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


def _validate_range(
    low: tuple[float, ...],
    high: tuple[float, ...],
    key: str,
) -> None:
    if len(low) != len(high):
        raise ValueError(
            f"domain_randomization.{key}_low and {key}_high must have the same length"
        )
    if any(lo > hi for lo, hi in zip(low, high)):
        raise ValueError(
            f"domain_randomization.{key}_low must be <= {key}_high componentwise"
        )


@dataclass(frozen=True)
class CyringeEpisodeSample:
    """Randomized hand and cyringe pose for one reset."""

    hand_qpos: tuple[float, ...] | None
    cyringe_offset: tuple[float, float, float]
    cyringe_euler_delta: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "hand_qpos": None if self.hand_qpos is None else list(self.hand_qpos),
            "cyringe_offset": list(self.cyringe_offset),
            "cyringe_euler_delta": list(self.cyringe_euler_delta),
        }


@dataclass(frozen=True)
class CyringeDomainRandomizationConfig:
    """Ranges sampled on every ``reset()`` when the env has no sticky housing.

    Hand joints are drawn uniformly in the actuator ``ctrlrange`` (or an optional
    override). Cyringe pose is the compiled spawn plus a uniform XYZ offset and
    XYZ Euler delta (radians), matching ``delta_x/y/z`` and ``delta_rot_z``.
    """

    randomize_hand: bool = True
    randomize_cyringe: bool = True
    offset_low: tuple[float, float, float] = DEFAULT_OFFSET_LOW
    offset_high: tuple[float, float, float] = DEFAULT_OFFSET_HIGH
    euler_delta_low: tuple[float, float, float] = DEFAULT_EULER_DELTA_LOW
    euler_delta_high: tuple[float, float, float] = DEFAULT_EULER_DELTA_HIGH

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any] | None
    ) -> CyringeDomainRandomizationConfig | None:
        if raw is None or not bool(raw.get("enabled", False)):
            return None

        hand = _section(raw, "hand")
        cyringe = _section(raw, "cyringe")
        offset_low = _as_xyz(
            cyringe.get("offset_low", list(DEFAULT_OFFSET_LOW)),
            "cyringe.offset_low",
        )
        offset_high = _as_xyz(
            cyringe.get("offset_high", list(DEFAULT_OFFSET_HIGH)),
            "cyringe.offset_high",
        )
        euler_delta_low = _as_xyz(
            cyringe.get("euler_delta_low", list(DEFAULT_EULER_DELTA_LOW)),
            "cyringe.euler_delta_low",
        )
        euler_delta_high = _as_xyz(
            cyringe.get("euler_delta_high", list(DEFAULT_EULER_DELTA_HIGH)),
            "cyringe.euler_delta_high",
        )
        _validate_range(offset_low, offset_high, "cyringe.offset")
        _validate_range(euler_delta_low, euler_delta_high, "cyringe.euler_delta")

        return cls(
            randomize_hand=_as_bool(hand.get("enabled"), "hand.enabled", True),
            randomize_cyringe=_as_bool(
                cyringe.get("enabled"), "cyringe.enabled", True
            ),
            offset_low=offset_low,
            offset_high=offset_high,
            euler_delta_low=euler_delta_low,
            euler_delta_high=euler_delta_high,
        )

    @classmethod
    def resolve(
        cls,
        value: CyringeDomainRandomizationConfig | dict[str, Any] | None,
        raw_config: dict[str, Any] | None = None,
    ) -> CyringeDomainRandomizationConfig | None:
        json_cfg = dict(raw_config) if isinstance(raw_config, dict) else {}
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            raw = _deep_merge(json_cfg, value)
            raw.setdefault("enabled", True)
            return cls.from_dict(raw)
        if value is not None:
            raise TypeError(
                "domain_randomization must be a "
                "CyringeDomainRandomizationConfig, dict, or None"
            )
        return cls.from_dict(json_cfg or None)

    @classmethod
    def from_env_config(
        cls,
        raw: dict[str, Any] | None,
        *,
        randomize_hand: bool | None = None,
        randomize_cyringe: bool | None = None,
    ) -> CyringeDomainRandomizationConfig | None:
        """Load ranges from ``env_config.json``; optional flags only toggle what runs.

        Demo flags turn DR on even if JSON has ``enabled: false``. Offset and
        euler ranges always come from the JSON ``cyringe`` section.
        """
        flags_passed = randomize_hand is not None or randomize_cyringe is not None
        payload = dict(raw) if isinstance(raw, dict) else {}
        if flags_passed:
            payload["enabled"] = True
        parsed = cls.from_dict(payload or None)
        if parsed is None or not flags_passed:
            return parsed
        return replace(
            parsed,
            randomize_hand=bool(randomize_hand),
            randomize_cyringe=bool(randomize_cyringe),
        )

    def sample(
        self,
        rng: np.random.Generator,
        *,
        hand_lo: np.ndarray | None = None,
        hand_hi: np.ndarray | None = None,
    ) -> CyringeEpisodeSample:
        """Draw one episode's hand qpos and cyringe pose deltas."""
        hand_qpos: tuple[float, ...] | None = None
        if self.randomize_hand:
            if hand_lo is None or hand_hi is None:
                raise ValueError(
                    "hand_lo and hand_hi are required when randomize_hand is True"
                )
            lo = np.asarray(hand_lo, dtype=np.float64).reshape(-1)
            hi = np.asarray(hand_hi, dtype=np.float64).reshape(-1)
            if lo.size != N_HAND_DOFS or hi.size != N_HAND_DOFS:
                raise ValueError(
                    f"hand_lo/hand_hi must have length {N_HAND_DOFS}, "
                    f"got {lo.size} and {hi.size}"
                )
            hand_qpos = tuple(float(v) for v in rng.uniform(lo, hi))

        offset = (0.0, 0.0, 0.0)
        euler_delta = (0.0, 0.0, 0.0)
        if self.randomize_cyringe:
            offset = tuple(
                float(v)
                for v in rng.uniform(self.offset_low, self.offset_high)
            )
            euler_delta = tuple(
                float(v)
                for v in rng.uniform(self.euler_delta_low, self.euler_delta_high)
            )
        return CyringeEpisodeSample(
            hand_qpos=hand_qpos,
            cyringe_offset=offset,
            cyringe_euler_delta=euler_delta,
        )
