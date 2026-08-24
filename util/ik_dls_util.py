"""Thumb / cyringe damped least-squares (DLS) IK helpers that write into ``data.ctrl``."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Iterable

import mujoco as mj
import numpy as np

from util.ik_common import (
    CYRINGE_MF_RF_DIP,
    CYRINGE_MF_RF_MCP,
    CYRINGE_MF_RF_PIP,
    DEFAULT_DELTA_IK_Z,
    DEFAULT_HANDLE_SITE,
    DEFAULT_IK_DAMPING,
    DEFAULT_IK_GAIN,
    DEFAULT_IK_GOAL_MOCAP,
    DEFAULT_IK_MAX_DQ,
    DEFAULT_IK_REACH_TOL,
    DEFAULT_THUMB_EE_SITE,
    DEFAULT_THUMB_FLEX,
    THUMB_JOINTS,
    add_ik_goal_mocap,
    cyringe_two_phase_motion as _cyringe_two_phase_motion,
    maybe_advance_trajectory,
    mocap_id,
    resolve_ee_position,
    resolve_step_target,
    thumb_dof_and_actuator_ids,
)
from util.motion_util import DEFAULT_CLOSE_DURATION as _DEFAULT_CLOSE_DURATION

# Re-export shared symbols used by the demo.
__all__ = [
    "CYRINGE_MF_RF_DIP",
    "CYRINGE_MF_RF_MCP",
    "CYRINGE_MF_RF_PIP",
    "DEFAULT_CLOSE_DURATION",
    "DEFAULT_DELTA_IK_Z",
    "DEFAULT_HANDLE_SITE",
    "DEFAULT_IK_DAMPING",
    "DEFAULT_IK_GAIN",
    "DEFAULT_IK_GOAL_MOCAP",
    "DEFAULT_IK_MAX_DQ",
    "DEFAULT_IK_REACH_TOL",
    "DEFAULT_THUMB_EE_SITE",
    "DEFAULT_THUMB_FLEX",
    "add_ik_goal_mocap",
    "cyringe_two_phase_motion",
    "thumb_flex_center_ik_generator",
    "thumb_flex_surface_ik_generator",
]

DEFAULT_CLOSE_DURATION = _DEFAULT_CLOSE_DURATION


def thumb_flex_center_ik_generator(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    target_site: str = DEFAULT_HANDLE_SITE,
    flex_name: str = DEFAULT_THUMB_FLEX,
    ee_site: str | None = DEFAULT_THUMB_EE_SITE,
    joint_names: Iterable[str] = THUMB_JOINTS,
    gain: float = DEFAULT_IK_GAIN,
    damping: float = DEFAULT_IK_DAMPING,
    max_dq: float = DEFAULT_IK_MAX_DQ,
    reach_tol: float = DEFAULT_IK_REACH_TOL,
    delta_ik_z: float = DEFAULT_DELTA_IK_Z,
    goal_mocap: str | None = DEFAULT_IK_GOAL_MOCAP,
    hold_when_reached: bool = True,
    trajectory=None,
) -> Iterator[dict[str, float | int | bool | np.ndarray]]:
    """Yield forever; each ``next()`` drives the thumb toward the IK goal via DLS."""
    from util.flex_util import flex_id, flex_surface_center, flex_vertex_body_ids

    names = tuple(joint_names)
    dof_ids, act_ids = thumb_dof_and_actuator_ids(model, names)
    lo = model.actuator_ctrlrange[act_ids, 0].copy()
    hi = model.actuator_ctrlrange[act_ids, 1].copy()

    goal_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, target_site)
    if goal_site_id < 0:
        raise ValueError(f"Site '{target_site}' not found")

    ee_site_id = -1
    if ee_site:
        ee_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, ee_site)

    fid = flex_id(model, flex_name)
    vert_adr = int(model.flex_vertadr[fid])
    vert_num = int(model.flex_vertnum[fid])
    if vert_num <= 0:
        raise ValueError(f"Flex '{flex_name}' has no vertices")
    body_ids = flex_vertex_body_ids(model, flex_name)

    ee_vertex, _center = flex_surface_center(model, data, flex_name)
    ee_body = int(body_ids[ee_vertex])
    use_ee_site = ee_site_id >= 0

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    reached = False

    mid = mocap_id(model, goal_mocap) if goal_mocap else -1

    while True:
        target = resolve_step_target(
            data,
            goal_site_id=goal_site_id,
            delta_ik_z=delta_ik_z,
            trajectory=trajectory,
            goal_mocap_id=mid,
        )

        ee_pos = resolve_ee_position(
            model,
            data,
            ee_site_id=ee_site_id if use_ee_site else -1,
            flex_name=flex_name,
            ee_vertex=ee_vertex,
            vert_adr=vert_adr,
        )
        err = target - ee_pos
        dist = float(np.linalg.norm(err))

        should_hold, new_target = maybe_advance_trajectory(
            trajectory,
            dist=dist,
            reach_tol=reach_tol,
            hold_when_reached=hold_when_reached,
        )
        if new_target is not None:
            target = new_target
            if mid >= 0:
                data.mocap_pos[mid] = target
            err = target - ee_pos
            dist = float(np.linalg.norm(err))
        if should_hold:
            reached = True
            yield {
                "reached": True,
                "dist": dist,
                "err_xyz": err.copy(),
                "ee_vertex": ee_vertex,
                "goal": target.copy(),
            }
            continue

        if use_ee_site:
            mj.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        else:
            mj.mj_jacBody(model, data, jacp, jacr, ee_body)
        J = jacp[:, dof_ids]
        JJT = J @ J.T + float(damping) * eye3
        dq = J.T @ np.linalg.solve(JJT, float(gain) * err)
        max_step = float(max_dq)
        nrm = float(np.linalg.norm(dq))
        if nrm > max_step > 0.0:
            dq *= max_step / nrm

        data.ctrl[act_ids] = np.clip(data.ctrl[act_ids] + dq, lo, hi)
        yield {
            "reached": reached and dist <= float(reach_tol),
            "dist": dist,
            "err_xyz": err.copy(),
            "ee_vertex": ee_vertex,
            "goal": target.copy(),
        }


thumb_flex_surface_ik_generator = thumb_flex_center_ik_generator


def cyringe_two_phase_motion(
    model: mj.MjModel,
    data: mj.MjData,
    **kwargs,
) -> Iterator[dict[str, float | int | bool | str | np.ndarray]]:
    return _cyringe_two_phase_motion(
        model, data, ik_generator=thumb_flex_center_ik_generator, **kwargs
    )
