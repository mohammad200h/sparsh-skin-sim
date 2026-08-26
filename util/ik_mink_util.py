"""Thumb / cyringe IK via Mink (MuJoCo differential IK / QP).

See https://kevinzakka.github.io/mink/

The full Leap+flex scene has O(1000) DOFs, which makes a dense QP impractical.
This backend builds a small rigid thumb chain MuJoCo model, runs Mink IK on that
chain, and maps the resulting joint increments onto the main scene actuators.
"""

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
    THUMB_JOINT_BODY,
    THUMB_JOINTS,
    add_ik_goal_mocap,
    body_rel_pose,
    cyringe_two_phase_motion as _cyringe_two_phase_motion,
    ee_pose_in_body,
    maybe_advance_trajectory,
    mocap_id,
    read_joint_qpos,
    resolve_ee_position,
    resolve_step_target,
    thumb_base_body_id,
    thumb_dof_and_actuator_ids,
    world_to_body,
)
from util.motion_util import DEFAULT_CLOSE_DURATION

__all__ = [
    "CYRINGE_MF_RF_DIP",
    "CYRINGE_MF_RF_MCP",
    "CYRINGE_MF_RF_PIP",
    "DEFAULT_CLOSE_DURATION",
    "DEFAULT_DELTA_IK_Z",
    "DEFAULT_HANDLE_SITE",
    "DEFAULT_IK_GOAL_MOCAP",
    "DEFAULT_IK_REACH_TOL",
    "DEFAULT_THUMB_EE_SITE",
    "DEFAULT_THUMB_FLEX",
    "add_ik_goal_mocap",
    "cyringe_two_phase_motion",
    "thumb_flex_center_ik_generator",
]

_EE_SITE = "ee"


def build_thumb_mjcf(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    ee_site: str,
    joint_names: tuple[str, ...] = THUMB_JOINTS,
) -> str:
    """Minimal MJCF serial chain matching Leap thumb kinematics."""
    ee_pos, ee_quat = ee_pose_in_body(model, data, ee_site=ee_site, parent_body="th_ds")

    lines = [
        '<mujoco model="leap_thumb_ik">',
        '  <compiler angle="radian"/>',
        '  <option timestep="0.002"/>',
        "  <worldbody>",
        '    <body name="thumb_base">',
    ]
    indent = "      "
    for jname in joint_names:
        body_name = THUMB_JOINT_BODY[jname]
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        if bid < 0 or jid < 0:
            raise ValueError(f"Missing thumb frame for '{jname}'")
        pos, quat = body_rel_pose(model, bid)
        axis = np.asarray(model.jnt_axis[jid], dtype=np.float64)
        lo, hi = [float(v) for v in model.jnt_range[jid]]
        lines.append(
            f'{indent}<body name="{body_name}" '
            f'pos="{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}" '
            f'quat="{quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}">'
        )
        lines.append(
            f'{indent}  <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>'
        )
        lines.append(
            f'{indent}  <joint name="{jname}" type="hinge" '
            f'axis="{axis[0]:.8f} {axis[1]:.8f} {axis[2]:.8f}" '
            f'range="{lo:.8f} {hi:.8f}"/>'
        )
        indent += "  "

    lines.append(
        f'{indent}<site name="{_EE_SITE}" '
        f'pos="{ee_pos[0]:.8f} {ee_pos[1]:.8f} {ee_pos[2]:.8f}" '
        f'quat="{ee_quat[0]:.8f} {ee_quat[1]:.8f} {ee_quat[2]:.8f} {ee_quat[3]:.8f}" '
        f'size="0.005"/>'
    )
    for _ in joint_names:
        indent = indent[:-2]
        lines.append(f"{indent}</body>")
    lines.append("    </body>")
    lines.append("  </worldbody>")
    lines.append("</mujoco>")
    return "\n".join(lines) + "\n"


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
    qp_solver: str = "daqp",
    trajectory=None,
) -> Iterator[dict[str, float | int | bool | np.ndarray]]:
    """Drive the thumb EE toward the handle with Mink on a reduced thumb model."""
    try:
        import mink
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "IK backend 'mink' requires the mink package (pip install mink)."
        ) from exc

    from util.flex_util import flex_id, flex_surface_center

    if not ee_site:
        raise ValueError("mink backend requires ee_site")

    names = tuple(joint_names)
    _dof_ids, act_ids = thumb_dof_and_actuator_ids(model, names)
    lo = model.actuator_ctrlrange[act_ids, 0].copy()
    hi = model.actuator_ctrlrange[act_ids, 1].copy()

    goal_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, target_site)
    if goal_site_id < 0:
        raise ValueError(f"Site '{target_site}' not found")
    ee_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, ee_site)
    if ee_site_id < 0:
        raise ValueError(f"EE site '{ee_site}' not found")

    fid = flex_id(model, flex_name)
    vert_adr = int(model.flex_vertadr[fid])
    ee_vertex, _ = flex_surface_center(model, data, flex_name)
    base_body_id = thumb_base_body_id(model)

    thumb_model = mj.MjModel.from_xml_string(
        build_thumb_mjcf(model, data, ee_site=ee_site, joint_names=names)
    )
    thumb_data = mj.MjData(thumb_model)
    configuration = mink.Configuration(thumb_model)
    task = mink.FrameTask(
        frame_name=_EE_SITE,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,
        gain=float(np.clip(gain, 1e-3, 1.0)),
    )
    posture = mink.PostureTask(thumb_model, cost=1e-3)
    limits = [mink.ConfigurationLimit(thumb_model)]
    dt = float(model.opt.timestep)
    mid = mocap_id(model, goal_mocap) if goal_mocap else -1
    reached = False

    # qpos layout of reduced model: one hinge per thumb joint (nq == n_joints).
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
            ee_site_id=ee_site_id,
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

        q_thumb = read_joint_qpos(model, data, names)
        thumb_data.qpos[:] = q_thumb
        mj.mj_forward(thumb_model, thumb_data)
        configuration.update(thumb_data.qpos)
        posture.set_target_from_configuration(configuration)

        target_base = world_to_body(data, base_body_id, target)
        task.set_target(mink.SE3.from_translation(target_base))
        vel = mink.solve_ik(
            configuration,
            [task, posture],
            dt,
            qp_solver,
            damping=max(float(damping), 1e-8),
            limits=limits,
        )
        dq = np.asarray(vel, dtype=np.float64) * dt
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


def cyringe_two_phase_motion(
    model: mj.MjModel,
    data: mj.MjData,
    **kwargs,
) -> Iterator[dict[str, float | int | bool | str | np.ndarray]]:
    return _cyringe_two_phase_motion(
        model, data, ik_generator=thumb_flex_center_ik_generator, **kwargs
    )
