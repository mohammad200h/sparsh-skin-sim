"""Precomputed thumb-push trajectories along the cyringe handle axis."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco as mj
import numpy as np

from util.ik_common import mocap_id

DEFAULT_TRAJ_WAYPOINTS = 15  # was 8; 2× length at the same spacing
DEFAULT_TRAJ_STROKE = 0.08  # was 0.04 m along handle +Z
DEFAULT_TRAJ_MOCAP_PREFIX = "traj_wp"
DEFAULT_TRAJ_MOCAP_RADIUS = 0.004


def handle_frame_from_id(
    data: mj.MjData, handle_site_id: int
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(data.site_xpos[handle_site_id], dtype=np.float64).copy()
    axis = np.asarray(
        data.site_xmat[handle_site_id].reshape(3, 3)[:, 2], dtype=np.float64
    ).copy()
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-12:
        raise ValueError("Handle site local +Z is degenerate")
    return origin, axis / nrm


def compute_handle_push_trajectory(
    data: mj.MjData,
    *,
    handle_site_id: int,
    stroke: float = DEFAULT_TRAJ_STROKE,
    n_waypoints: int = DEFAULT_TRAJ_WAYPOINTS,
    start_offset: float = 0.0,
) -> np.ndarray:
    """Precompute world-frame waypoints along handle local +Z.

    Waypoints run from ``origin + start_offset * z`` to
    ``origin + (start_offset + stroke) * z`` (inclusive).
    """
    n = int(n_waypoints)
    if n < 2:
        raise ValueError("n_waypoints must be >= 2")
    origin, axis = handle_frame_from_id(data, handle_site_id)
    s0 = float(start_offset)
    s1 = s0 + float(stroke)
    alphas = np.linspace(s0, s1, n, dtype=np.float64)
    return origin[None, :] + alphas[:, None] * axis[None, :]


def add_trajectory_mocaps(
    spec: mj.MjSpec,
    n_waypoints: int = DEFAULT_TRAJ_WAYPOINTS,
    *,
    prefix: str = DEFAULT_TRAJ_MOCAP_PREFIX,
    radius: float = DEFAULT_TRAJ_MOCAP_RADIUS,
) -> list[str]:
    """Add ``n_waypoints`` non-colliding mocap spheres (blue→red) for viz."""
    n = int(n_waypoints)
    if n < 1:
        raise ValueError("n_waypoints must be >= 1")
    names: list[str] = []
    for i in range(n):
        name = f"{prefix}_{i}"
        existing = spec.body(name) if hasattr(spec, "body") else None
        if existing is not None:
            try:
                spec.delete(existing)
            except Exception:
                pass
        t = i / max(n - 1, 1)
        rgba = (0.15 + 0.75 * t, 0.35, 0.95 - 0.75 * t, 0.85)
        body = spec.worldbody.add_body(name=name, mocap=True)
        body.add_geom(
            name=f"{name}_geom",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[float(radius), 0.0, 0.0],
            rgba=[float(c) for c in rgba],
            contype=0,
            conaffinity=0,
            group=0,
        )
        names.append(name)
    return names


def place_trajectory_mocaps(
    model: mj.MjModel,
    data: mj.MjData,
    waypoints: np.ndarray,
    mocap_names: Sequence[str],
) -> None:
    """Write waypoint positions into the matching mocap bodies."""
    pts = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
    if len(mocap_names) != pts.shape[0]:
        raise ValueError(
            f"Expected {pts.shape[0]} mocap names, got {len(mocap_names)}"
        )
    for name, p in zip(mocap_names, pts):
        data.mocap_pos[mocap_id(model, name)] = p


class TrajectoryTracker:
    """Advance through fixed world-frame waypoints as the EE reaches each one."""

    def __init__(self, waypoints: np.ndarray) -> None:
        self.waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
        if self.waypoints.shape[0] < 1:
            raise ValueError("waypoints must be non-empty")
        self.idx = 0

    @property
    def current(self) -> np.ndarray:
        return self.waypoints[self.idx]

    @property
    def done(self) -> bool:
        return self.idx >= self.waypoints.shape[0] - 1

    def advance_if_reached(self, dist: float, reach_tol: float) -> bool:
        """Advance index when close enough. Returns True if index changed."""
        if dist > float(reach_tol) or self.done:
            return False
        self.idx += 1
        return True
