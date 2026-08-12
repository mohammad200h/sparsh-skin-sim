"""Tetris spawn domain randomization for ``LeapFlexEnv``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from util.objects_util import TETRIS_SHAPE_NAMES


def _is_xyz_triplet(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _as_str_list(raw: Any, key: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"domain_randomization.{key} must be a non-empty list")
    return tuple(str(item) for item in raw)


def _as_float_list(raw: Any, key: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"domain_randomization.{key} must be a non-empty list")
    return tuple(float(item) for item in raw)


def _as_flex_list(raw: Any, key: str) -> tuple[str | None, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"domain_randomization.{key} must be a non-empty list")
    return tuple(None if item is None else str(item) for item in raw)


def _as_offset_list(raw: Any) -> tuple[tuple[float, float, float], ...]:
    if _is_xyz_triplet(raw):
        return (tuple(float(v) for v in raw),)
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "domain_randomization.offset must be [x, y, z] or a list of offsets"
        )
    if _is_xyz_triplet(raw[0]):
        return tuple(tuple(float(v) for v in item) for item in raw)
    raise ValueError(
        "domain_randomization.offset must be [x, y, z] or a list of [x, y, z] triplets"
    )


@dataclass(frozen=True)
class TetrisSpawn:
    """Resolved tetris spawn parameters for one episode / env slot."""

    shape: str
    scale: float
    offset: tuple[float, float, float]
    flex: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "scale": self.scale,
            "offset": list(self.offset),
            "flex": self.flex,
        }


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Pools of tetris spawn values sampled on reset or per vector-env slot."""

    shapes: tuple[str, ...]
    scales: tuple[float, ...]
    offsets: tuple[tuple[float, float, float], ...]
    flexes: tuple[str | None, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> DomainRandomizationConfig | None:
        if raw is None or not bool(raw.get("enabled", False)):
            return None

        shapes = _as_str_list(raw.get("shape", ["T"]), "shape")
        scales = _as_float_list(raw.get("scale", [1.0]), "scale")
        offsets = _as_offset_list(raw.get("offset", [0.0, 0.0, 0.0]))
        flexes = _as_flex_list(raw.get("flex", [None]), "flex")

        for shape in shapes:
            if shape.upper() not in {name.upper() for name in TETRIS_SHAPE_NAMES}:
                raise ValueError(
                    f"domain_randomization.shape contains unknown shape {shape!r}; "
                    f"choose from {TETRIS_SHAPE_NAMES}"
                )

        return cls(shapes=shapes, scales=scales, offsets=offsets, flexes=flexes)

    def _pick(self, values: tuple[Any, ...], index: int) -> Any:
        if len(values) == 1:
            return values[0]
        return values[index % len(values)]

    def for_slot(self, slot: int) -> TetrisSpawn:
        """Deterministic assignment for one vector-env slot."""
        return TetrisSpawn(
            shape=self._pick(self.shapes, slot),
            scale=float(self._pick(self.scales, slot)),
            offset=tuple(float(v) for v in self._pick(self.offsets, slot)),
            flex=self._pick(self.flexes, slot),
        )

    def sample(self, rng: np.random.Generator) -> TetrisSpawn:
        """Random assignment from each pool."""
        return TetrisSpawn(
            shape=str(rng.choice(self.shapes)),
            scale=float(rng.choice(self.scales)),
            offset=tuple(float(v) for v in rng.choice(self.offsets)),
            flex=rng.choice(self.flexes),
        )
