"""Thumb / cyringe IK via PyRoki (JAX LM kinematic optimization).

See https://pyroki-toolkit.github.io/

PyRoki expects a URDF robot. This backend builds a minimal 4-DoF thumb chain
URDF from the Leap MuJoCo bodies (``th_mp`` … ``th_ds`` + EE site) and solves
pose IK each step, mapping the solution onto thumb position actuators.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
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
    quat_wxyz_to_rpy,
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

_EE_LINK = "thumb_ee"


def build_thumb_urdf_xml(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    ee_site: str,
    joint_names: tuple[str, ...] = THUMB_JOINTS,
) -> str:
    """Author a serial-chain URDF matching Leap thumb kinematics."""
    ee_pos, ee_quat = ee_pose_in_body(model, data, ee_site=ee_site, parent_body="th_ds")

    lines = [
        '<?xml version="1.0"?>',
        '<robot name="leap_thumb">',
        '  <link name="thumb_base"/>',
    ]
    parent_link = "thumb_base"
    for jname in joint_names:
        body_name = THUMB_JOINT_BODY[jname]
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found")
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not found")
        pos, quat = body_rel_pose(model, bid)
        rpy = quat_wxyz_to_rpy(quat)
        axis = np.asarray(model.jnt_axis[jid], dtype=np.float64)
        lo, hi = [float(v) for v in model.jnt_range[jid]]
        child_link = body_name
        lines.append(f'  <link name="{child_link}"/>')
        lines.append(f'  <joint name="{jname}" type="revolute">')
        lines.append(f'    <parent link="{parent_link}"/>')
        lines.append(f'    <child link="{child_link}"/>')
        lines.append(
            f'    <origin xyz="{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}" '
            f'rpy="{rpy[0]:.8f} {rpy[1]:.8f} {rpy[2]:.8f}"/>'
        )
        lines.append(f'    <axis xyz="{axis[0]:.8f} {axis[1]:.8f} {axis[2]:.8f}"/>')
        lines.append(
            f'    <limit lower="{lo:.8f}" upper="{hi:.8f}" effort="10" velocity="3"/>'
        )
        lines.append("  </joint>")
        parent_link = child_link

    ee_rpy = quat_wxyz_to_rpy(ee_quat)
    lines.append(f'  <link name="{_EE_LINK}"/>')
    lines.append('  <joint name="ee_fixed" type="fixed">')
    lines.append('    <parent link="th_ds"/>')
    lines.append(f'    <child link="{_EE_LINK}"/>')
    lines.append(
        f'    <origin xyz="{ee_pos[0]:.8f} {ee_pos[1]:.8f} {ee_pos[2]:.8f}" '
        f'rpy="{ee_rpy[0]:.8f} {ee_rpy[1]:.8f} {ee_rpy[2]:.8f}"/>'
    )
    lines.append("  </joint>")
    lines.append("</robot>")
    return "\n".join(lines) + "\n"

def _make_pyroki_solver(robot, target_link_index: int):
    """Build a once-compiled PyRoki LM solver for ``robot``."""
    import jax.numpy as jnp
    import jax_dataclasses as jdc
    import jaxlie
    import jaxls
    import pyroki as pk

    link_idx = jnp.asarray(int(target_link_index), dtype=jnp.int32)

    @jdc.jit
    def _solve(target_wxyz_j, target_pos_j, init_j):
        joint_var = robot.joint_var_cls(0)
        costs = [
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(target_wxyz_j), target_pos_j
                ),
                link_idx,
                pos_weight=50.0,
                ori_weight=0.0,
            ),
            pk.costs.limit_cost(robot, joint_var, 100.0),
            pk.costs.rest_cost(joint_var, init_j, 1.0),
        ]
        problem = jaxls.LeastSquaresProblem(
            costs=costs, variables=[joint_var]
        ).analyze()
        sol = problem.solve(
            initial_vals=jaxls.VarValues.make((joint_var.with_value(init_j),)),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
        return sol[joint_var]

    def solve(
        target_wxyz: np.ndarray,
        target_position: np.ndarray,
        initial_cfg: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            _solve(
                jnp.asarray(target_wxyz),
                jnp.asarray(target_position),
                jnp.asarray(initial_cfg),
            ),
            dtype=np.float64,
        )

    return solve


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
    """Drive the thumb toward the handle with PyRoki LM IK."""
    del gain, damping  # absolute LM solve; rate-limited via max_dq
    try:
        import pyroki as pk
        import yourdfpy
    except ImportError as exc:  # pragma: no cover
        venv = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
        raise ImportError(
            "IK backend 'pyroki' failed to import "
            f"({exc}). Use {venv} — system JAX 0.4.14 (DreamerV3) cannot load pyroki."
        ) from exc

    from util.flex_util import flex_id, flex_surface_center

    if not ee_site:
        raise ValueError("pyroki backend requires ee_site")

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

    urdf_xml = build_thumb_urdf_xml(
        model, data, ee_site=ee_site, joint_names=names
    )
    with tempfile.TemporaryDirectory(prefix="leap_thumb_urdf_") as tmp:
        urdf_path = Path(tmp) / "thumb.urdf"
        urdf_path.write_text(urdf_xml)
        urdf = yourdfpy.URDF.load(urdf_path.as_posix())
        q0 = read_joint_qpos(model, data, names)
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=q0)
        target_link_index = robot.links.names.index(_EE_LINK)
        solve_ik = _make_pyroki_solver(robot, target_link_index)
        print("  pyroki: compiling LM solver (first thumb_ik step may be slow)...")

        mid = mocap_id(model, goal_mocap) if goal_mocap else -1
        reached = False
        # Identity orientation (position-only cost weight on ori is 0).
        target_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        compiled = False

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

            target_base = world_to_body(data, base_body_id, target)
            q_now = read_joint_qpos(model, data, names)
            try:
                q_sol = solve_ik(target_wxyz, target_base, q_now)
                if not compiled:
                    print("  pyroki: LM solver ready")
                    compiled = True
            except Exception:
                # Fall back to holding current ctrl on rare solver failures.
                q_sol = q_now

            dq = q_sol - data.ctrl[act_ids]
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
