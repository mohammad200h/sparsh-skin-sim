"""Shared constants and helpers for cyringe thumb IK backends."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import mujoco as mj
import numpy as np

from util.motion_util import (
    DEFAULT_CLOSE_DURATION,
    finger_close_generator,
    resolve_actuator_ids,
)

# Cyringe barrel grip: middle + ring only (matches the on-pad grasp).
MF_RF_CLOSE_JOINTS: tuple[str, ...] = (
    "mf_mcp",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_pip",
    "rf_dip",
)

THUMB_JOINTS: tuple[str, ...] = ("th_cmc", "th_axl", "th_mcp", "th_ipl")

# Values from the sticky cyringe grasp (viewer joint panel).
CYRINGE_MF_RF_MCP = 0.91
CYRINGE_MF_RF_PIP = 1.89
CYRINGE_MF_RF_DIP = 0.592
DEFAULT_HANDLE_SITE = "cyringe/handle"
DEFAULT_THUMB_FLEX = "flex_th_tip"
DEFAULT_THUMB_EE_SITE = "flex_th_tip_ee"
DEFAULT_IK_GAIN = 1.0
DEFAULT_IK_DAMPING = 1e-3
DEFAULT_IK_MAX_DQ = 0.02
DEFAULT_IK_REACH_TOL = 0.008
DEFAULT_DELTA_IK_Z = 0.0
DEFAULT_IK_GOAL_MOCAP = "ik_goal"

IK_BACKENDS = ("dls", "mink", "pyroki")

ThumbIkGenerator = Callable[..., Iterator[dict[str, float | int | bool | np.ndarray]]]

# MuJoCo body that carries each thumb joint (child body of the hinge).
THUMB_JOINT_BODY = {
    "th_cmc": "th_mp",
    "th_axl": "th_bs",
    "th_mcp": "th_px",
    "th_ipl": "th_ds",
}


def quat_wxyz_to_rpy(wxyz: np.ndarray) -> tuple[float, float, float]:
    """Convert MuJoCo wxyz quaternion to URDF RPY."""
    w, x, y, z = [float(v) for v in wxyz]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def body_rel_pose(model: mj.MjModel, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(model.body_pos[body_id], dtype=np.float64)
    quat = np.asarray(model.body_quat[body_id], dtype=np.float64)
    return pos, quat


def thumb_base_body_id(model: mj.MjModel) -> int:
    th_mp = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "th_mp")
    if th_mp < 0:
        raise ValueError("Body 'th_mp' not found")
    return int(model.body_parentid[th_mp])


def world_to_body(
    data: mj.MjData, body_id: int, p_world: np.ndarray
) -> np.ndarray:
    R = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    o = np.asarray(data.xpos[body_id], dtype=np.float64)
    return R.T @ (np.asarray(p_world, dtype=np.float64) - o)


def read_joint_qpos(
    model: mj.MjModel, data: mj.MjData, joint_names: tuple[str, ...]
) -> np.ndarray:
    q = np.empty(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        q[i] = float(data.qpos[int(model.jnt_qposadr[jid])])
    return q


def ee_pose_in_body(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    ee_site: str,
    parent_body: str = "th_ds",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, quat_wxyz) of ``ee_site`` expressed in ``parent_body``."""
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, ee_site)
    if ee_id < 0:
        raise ValueError(f"EE site '{ee_site}' not found")
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, parent_body)
    if bid < 0:
        raise ValueError(f"Body '{parent_body}' not found")
    R_p = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
    p_p = np.asarray(data.xpos[bid], dtype=np.float64)
    p_ee = np.asarray(data.site_xpos[ee_id], dtype=np.float64)
    R_ee = np.asarray(data.site_xmat[ee_id], dtype=np.float64).reshape(3, 3)
    p_local = R_p.T @ (p_ee - p_p)
    R_local = R_p.T @ R_ee
    # Rotation matrix → wxyz via MuJoCo.
    quat = np.empty(4, dtype=np.float64)
    mj.mju_mat2Quat(quat, R_local.reshape(9))
    return p_local, quat


def thumb_dof_and_actuator_ids(
    model: mj.MjModel, joint_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Map thumb hinge joints to ``qvel`` columns and position actuators."""
    dof_ids: list[int] = []
    for name in joint_names:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found")
        dof_ids.append(int(model.jnt_dofadr[jid]))
    act_ids = resolve_actuator_ids(model, joint_names)
    return (
        np.asarray(dof_ids, dtype=np.int32),
        act_ids,
    )


def add_ik_goal_mocap(
    spec: mj.MjSpec,
    *,
    name: str = DEFAULT_IK_GOAL_MOCAP,
    radius: float = 0.008,
    rgba: tuple[float, float, float, float] = (0.95, 0.85, 0.1, 0.9),
) -> str:
    """Add a non-colliding mocap sphere used to visualise the IK goal."""
    existing = spec.body(name) if hasattr(spec, "body") else None
    if existing is not None:
        try:
            spec.delete(existing)
        except Exception:
            pass

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
    return name


def mocap_id(model: mj.MjModel, body_name: str) -> int:
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Mocap body '{body_name}' not found")
    mid = int(model.body_mocapid[bid])
    if mid < 0:
        raise ValueError(f"Body '{body_name}' is not a mocap body")
    return mid


def ik_goal_position(
    data: mj.MjData,
    goal_site_id: int,
    delta_ik_z: float,
) -> np.ndarray:
    """World-frame IK goal: ``site_xpos + delta_ik_z * site_local_z``."""
    site_pos = np.asarray(data.site_xpos[goal_site_id], dtype=np.float64)
    site_z = np.asarray(
        data.site_xmat[goal_site_id].reshape(3, 3)[:, 2], dtype=np.float64
    )
    return site_pos + float(delta_ik_z) * site_z


def resolve_step_target(
    data: mj.MjData,
    *,
    goal_site_id: int,
    delta_ik_z: float,
    trajectory: object | None,
    goal_mocap_id: int = -1,
) -> np.ndarray:
    """Live handle+δz goal, or the current precomputed trajectory waypoint."""
    if trajectory is not None:
        target = np.asarray(trajectory.current, dtype=np.float64).copy()
    else:
        target = ik_goal_position(data, goal_site_id, delta_ik_z)
    if goal_mocap_id >= 0:
        data.mocap_pos[goal_mocap_id] = target
    return target


def maybe_advance_trajectory(
    trajectory: object | None,
    *,
    dist: float,
    reach_tol: float,
    hold_when_reached: bool,
) -> tuple[bool, np.ndarray | None]:
    """Advance trajectory waypoint or decide to hold.

    Returns ``(should_hold, new_target_or_None)``.
    """
    if trajectory is None:
        if dist <= float(reach_tol) and hold_when_reached:
            return True, None
        return False, None
    if dist > float(reach_tol):
        return False, None
    if trajectory.done:
        return bool(hold_when_reached), None
    trajectory.advance_if_reached(dist, reach_tol)
    return False, np.asarray(trajectory.current, dtype=np.float64).copy()


def resolve_ee_position(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    ee_site_id: int,
    flex_name: str,
    ee_vertex: int,
    vert_adr: int,
) -> np.ndarray:
    if ee_site_id >= 0:
        return np.asarray(data.site_xpos[ee_site_id], dtype=np.float64)
    return np.asarray(data.flexvert_xpos[vert_adr + ee_vertex], dtype=np.float64)


def cyringe_two_phase_motion(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    ik_generator: ThumbIkGenerator,
    close_duration: float = DEFAULT_CLOSE_DURATION,
    mcp_target: float = CYRINGE_MF_RF_MCP,
    pip_target: float = CYRINGE_MF_RF_PIP,
    dip_target: float = CYRINGE_MF_RF_DIP,
    handle_site: str = DEFAULT_HANDLE_SITE,
    thumb_flex: str = DEFAULT_THUMB_FLEX,
    ee_site: str | None = DEFAULT_THUMB_EE_SITE,
    ik_gain: float = DEFAULT_IK_GAIN,
    ik_damping: float = DEFAULT_IK_DAMPING,
    ik_max_dq: float = DEFAULT_IK_MAX_DQ,
    reach_tol: float = DEFAULT_IK_REACH_TOL,
    delta_ik_z: float = DEFAULT_DELTA_IK_Z,
    goal_mocap: str | None = DEFAULT_IK_GOAL_MOCAP,
    trajectory=None,
    traj_stroke: float | None = None,
    traj_waypoints: int | None = None,
    traj_mocap_names: list[str] | None = None,
) -> Iterator[dict[str, float | int | bool | str | np.ndarray]]:
    """Two-phase cyringe grasp: MF/RF close, then thumb IK via ``ik_generator``.

    If ``trajectory`` is None but ``traj_mocap_names`` is set, a push trajectory is
    precomputed once when thumb IK starts (after finger close) and visualised.
    """
    from util.trajectory_generation import (
        DEFAULT_TRAJ_STROKE,
        DEFAULT_TRAJ_WAYPOINTS,
        TrajectoryTracker,
        compute_handle_push_trajectory,
        place_trajectory_mocaps,
    )

    close = finger_close_generator(
        model,
        data,
        duration=close_duration,
        mcp_target=mcp_target,
        pip_target=pip_target,
        dip_target=dip_target,
        joint_names=MF_RF_CLOSE_JOINTS,
        hold=True,
    )
    tracker = trajectory
    ik = None
    t0 = float(data.time)
    duration = max(float(close_duration), 1e-6)
    nan3 = np.full(3, np.nan, dtype=np.float64)
    goal_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, handle_site)

    while True:
        next(close)
        elapsed = float(data.time) - t0
        if elapsed < duration:
            if goal_mocap and goal_site_id >= 0:
                mid = mocap_id(model, goal_mocap)
                if tracker is not None:
                    data.mocap_pos[mid] = tracker.current
                else:
                    data.mocap_pos[mid] = ik_goal_position(
                        data, goal_site_id, delta_ik_z
                    )
            yield {
                "phase": "finger_close",
                "elapsed": elapsed,
                "reached": False,
                "dist": float("nan"),
                "err_xyz": nan3.copy(),
                "ee_vertex": -1,
            }
            continue

        if ik is None:
            if tracker is None and traj_mocap_names is not None and goal_site_id >= 0:
                stroke = float(
                    traj_stroke
                    if traj_stroke is not None
                    else (
                        delta_ik_z
                        if abs(float(delta_ik_z)) > 1e-9
                        else DEFAULT_TRAJ_STROKE
                    )
                )
                n_wp = int(
                    traj_waypoints
                    if traj_waypoints is not None
                    else len(traj_mocap_names)
                )
                waypoints = compute_handle_push_trajectory(
                    data,
                    handle_site_id=goal_site_id,
                    stroke=stroke,
                    n_waypoints=n_wp,
                )
                place_trajectory_mocaps(
                    model, data, waypoints, traj_mocap_names[:n_wp]
                )
                tracker = TrajectoryTracker(waypoints)
            ik = ik_generator(
                model,
                data,
                target_site=handle_site,
                flex_name=thumb_flex,
                ee_site=ee_site,
                gain=ik_gain,
                damping=ik_damping,
                max_dq=ik_max_dq,
                reach_tol=reach_tol,
                delta_ik_z=delta_ik_z,
                goal_mocap=goal_mocap,
                hold_when_reached=True,
                trajectory=tracker,
            )

        info = next(ik)
        err = np.asarray(info.get("err_xyz", nan3), dtype=np.float64).reshape(3)
        yield {
            "phase": "thumb_ik",
            "elapsed": elapsed,
            "reached": bool(info["reached"]),
            "dist": float(info["dist"]),
            "err_xyz": err,
            "ee_vertex": int(info["ee_vertex"]),
            "traj_idx": int(getattr(tracker, "idx", -1)) if tracker is not None else -1,
        }


def load_ik_backend(name: str):
    """Return the util module for ``name`` in ``IK_BACKENDS``."""
    key = str(name).lower().strip()
    if key == "dls":
        from util import ik_dls_util as mod
    elif key == "mink":
        from util import ik_mink_util as mod
    elif key == "pyroki":
        from util import ik_pyroki_util as mod
    else:
        raise ValueError(f"Unknown IK backend '{name}'. Choose from {IK_BACKENDS}")
    return mod
