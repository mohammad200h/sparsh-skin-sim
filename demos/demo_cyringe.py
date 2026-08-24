"""Load the Leap+XELA flex-sensor scene with the cyringe and open the viewer."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from _bootstrap import SCENE_XML
from util.flex_util import (
    AllFlexForceEstimator,
    add_flex_center_ee_site,
    list_flex_names,
)
from util.ik_common import (
    CYRINGE_MF_RF_DIP,
    CYRINGE_MF_RF_MCP,
    CYRINGE_MF_RF_PIP,
    DEFAULT_DELTA_IK_Z,
    DEFAULT_HANDLE_SITE,
    DEFAULT_IK_GOAL_MOCAP,
    DEFAULT_IK_REACH_TOL,
    DEFAULT_THUMB_EE_SITE,
    DEFAULT_THUMB_FLEX,
    IK_BACKENDS,
    add_ik_goal_mocap,
    load_ik_backend,
)
from util.motion_util import DEFAULT_CLOSE_DURATION
from util.objects_util import add_cyringe
from util.trajectory_generation import (
    DEFAULT_TRAJ_STROKE,
    DEFAULT_TRAJ_WAYPOINTS,
    add_trajectory_mocaps,
)
# Match the flex-sensor generator: the shipped MJX XML keeps iterations=5,
# which under-converges the soft skins so contact barely moves the joints.
SOLVER_ITERATIONS = 50
TH_AXL_ACT_INITIAL = 1.6
FINGERTIP_FLEX_NAMES: dict[str, str] = {
    "if": "flex_if_tip",
    "mf": "flex_mf_tip",
    "rf": "flex_rf_tip",
    "th": "flex_th_tip",
}
FINGERTIP_BODY_NAMES: dict[str, str] = {
    "if": "if_ds",
    "mf": "mf_ds",
    "rf": "rf_ds",
    "th": "th_ds",
}


def _set_actuator_initial(
    model: mj.MjModel,
    data: mj.MjData,
    actuator_name: str,
    value: float,
) -> None:
    """Set ``ctrl`` and linked joint ``qpos`` so the pose starts at ``value``."""
    act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if act_id < 0:
        raise ValueError(f"Actuator '{actuator_name}' not found")
    joint_id = int(model.actuator_trnid[act_id, 0])
    data.ctrl[act_id] = float(value)
    data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)


def _geom_ids_for_body(model: mj.MjModel, body_name: str) -> list[int]:
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return []
    adr = int(model.body_geomadr[body_id])
    n = int(model.body_geomnum[body_id])
    return list(range(adr, adr + n))


def _cyringe_geom_ids(model: mj.MjModel, housing_name: str) -> np.ndarray:
    prefix = housing_name.rsplit("/", 1)[0] + "/" if "/" in housing_name else ""
    ids: list[int] = []
    for body_id in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if name is None:
            continue
        if name == housing_name or name.startswith(prefix):
            adr = int(model.body_geomadr[body_id])
            n = int(model.body_geomnum[body_id])
            ids.extend(range(adr, adr + n))
    return np.asarray(ids, dtype=np.int32)


def _cache_fingertip_cyringe_ids(
    model: mj.MjModel, housing_name: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    keys = tuple(FINGERTIP_FLEX_NAMES)
    finger_index = {name: i for i, name in enumerate(keys)}
    flex_finger = np.full(model.nflex, -1, dtype=np.int8)
    geom_finger = np.full(model.ngeom, -1, dtype=np.int8)
    for finger, flex_name in FINGERTIP_FLEX_NAMES.items():
        flex_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, flex_name)
        if flex_id >= 0:
            flex_finger[flex_id] = finger_index[finger]
    for finger, body_name in FINGERTIP_BODY_NAMES.items():
        for geom_id in _geom_ids_for_body(model, body_name):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
            if name is not None and "tip" in name:
                geom_finger[geom_id] = finger_index[finger]
    object_geoms = _cyringe_geom_ids(model, housing_name)
    return keys, flex_finger, geom_finger, object_geoms


def _fingertip_cyringe_contacts(
    model: mj.MjModel,
    data: mj.MjData,
    keys: tuple[str, ...],
    flex_finger: np.ndarray,
    geom_finger: np.ndarray,
    object_geom_ids: np.ndarray,
) -> dict[str, bool]:
    values = np.zeros(len(keys), dtype=bool)
    ncon = int(data.ncon)
    if ncon == 0 or object_geom_ids.size == 0:
        return dict(zip(keys, values.tolist()))
    geoms = data.contact.geom[:ncon]
    flexes = data.contact.flex[:ncon]
    finger = np.where(
        flexes >= 0,
        flex_finger[np.clip(flexes, 0, flex_finger.size - 1)],
        np.where(
            geoms >= 0,
            geom_finger[np.clip(geoms, 0, geom_finger.size - 1)],
            -1,
        ),
    )
    hit = finger[(finger >= 0) & np.isin(geoms[:, ::-1], object_geom_ids)]
    if hit.size:
        values[hit] = True
    return dict(zip(keys, values.tolist()))


def _cyringe_motion_ids(
    model: mj.MjModel, housing_name: str
) -> tuple[int, int]:
    """Return ``(housing_body_id, squeeze_dofadr)``; ``-1`` if missing."""
    housing_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, housing_name)
    prefix = housing_name.rsplit("/", 1)[0] + "/" if "/" in housing_name else ""
    squeeze_name = f"{prefix}squeeze"
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, squeeze_name)
    dofadr = int(model.jnt_dofadr[jid]) if jid >= 0 else -1
    return int(housing_id), dofadr


def _cyringe_is_moving(
    data: mj.MjData,
    housing_body_id: int,
    squeeze_dofadr: int,
    *,
    housing_speed_tol: float = 1e-3,
    squeeze_speed_tol: float = 1e-4,
) -> tuple[bool, float, float]:
    """Housing COM linear speed [m/s] and squeeze joint speed [m/s]."""
    housing_speed = 0.0
    if housing_body_id >= 0:
        # cvel is (rot, lin) at the body COM.
        housing_speed = float(np.linalg.norm(data.cvel[housing_body_id, 3:6]))
    squeeze_speed = 0.0
    if squeeze_dofadr >= 0:
        squeeze_speed = abs(float(data.qvel[squeeze_dofadr]))
    moving = (
        housing_speed > housing_speed_tol or squeeze_speed > squeeze_speed_tol
    )
    return moving, housing_speed, squeeze_speed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leap+XELA cyringe demo")
    parser.add_argument(
        "--ik",
        type=str,
        default="dls",
        choices=list(IK_BACKENDS),
        help="Thumb IK backend: dls (custom), mink, or pyroki",
    )
    parser.add_argument(
        "--visualize-force",
        action="store_true",
        help="Color plots by Kelvin-Voigt estimated force instead of displacement",
    )
    parser.add_argument(
        "--no-visualize-force",
        action="store_true",
        help="Color plots by displacement (default)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Fixed color-scale max (metres for displacement, Newtons for force)",
    )
    parser.add_argument(
        "--flex",
        type=str,
        default=None,
        help=(
            "Spawn the cyringe above this flex "
            "(e.g. mf_tip or flex_uspa46_1). Default: all palm pads."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale of the cyringe (default: 1.0)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(0.01, 0.05, -0.12),
        help="XYZ offset added to the flex/palm spawn centre (metres)",
    )
    parser.add_argument(
        "--delta-x",
        type=float,
        default=0.0,
        help="Extra spawn offset along world X [m] (added to --offset)",
    )
    parser.add_argument(
        "--delta-y",
        type=float,
        default=0.0,
        help="Extra spawn offset along world Y [m] (added to --offset)",
    )
    parser.add_argument(
        "--delta-z",
        type=float,
        default=0.0,
        help="Extra spawn offset along world Z [m] (added to --offset)",
    )
    parser.add_argument(
        "--delta-rot-z",
        type=float,
        default=0.0,
        help="Extra rotation about world Z [rad] (yaw), added to the default 90° side-lay pose",
    )
    parser.add_argument(
        "--no-fk-viz",
        action="store_true",
        help="Disable the Open3D taxel FK visualizer",
    )
    parser.add_argument(
        "--sticky",
        action="store_true",
        default=False,
        help=(
            "Fix the cyringe housing in world (remove freejoint); "
            "shaft can still slide under contact"
        ),
    )
    parser.add_argument(
        "--close-duration",
        type=float,
        default=DEFAULT_CLOSE_DURATION,
        help="Seconds to ramp MF/RF to grasp targets before thumb IK",
    )
    parser.add_argument(
        "--mcp-target",
        type=float,
        default=CYRINGE_MF_RF_MCP,
        help=f"MF/RF MCP target [rad] (default: {CYRINGE_MF_RF_MCP})",
    )
    parser.add_argument(
        "--pip-target",
        type=float,
        default=CYRINGE_MF_RF_PIP,
        help=f"MF/RF PIP target [rad] (default: {CYRINGE_MF_RF_PIP})",
    )
    parser.add_argument(
        "--dip-target",
        type=float,
        default=CYRINGE_MF_RF_DIP,
        help=f"MF/RF DIP target [rad] (default: {CYRINGE_MF_RF_DIP})",
    )
    parser.add_argument(
        "--handle-site",
        type=str,
        default=DEFAULT_HANDLE_SITE,
        help=f"Target site for thumb IK (default: {DEFAULT_HANDLE_SITE})",
    )
    parser.add_argument(
        "--delta-ik-z",
        type=float,
        default=DEFAULT_DELTA_IK_Z,
        help=(
            "Offset [m] added to the handle site along its local +Z "
            "(plunger axis) so the IK goal sits ahead and the thumb pushes. "
            "With --trajectory, unused unless --traj-stroke is omitted "
            "(then |delta-ik-z| becomes the stroke if non-zero)."
        ),
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help=(
            "Precompute a push trajectory along handle +Z and have IK follow "
            "it (visualised as a series of mocap spheres)"
        ),
    )
    parser.add_argument(
        "--traj-stroke",
        type=float,
        default=None,
        help=(
            f"Trajectory length along handle +Z [m] "
            f"(default: {DEFAULT_TRAJ_STROKE}, or |delta-ik-z| if set)"
        ),
    )
    parser.add_argument(
        "--traj-waypoints",
        type=int,
        default=DEFAULT_TRAJ_WAYPOINTS,
        help=f"Number of trajectory waypoints / mocaps (default: {DEFAULT_TRAJ_WAYPOINTS})",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Disable the two-phase grasp motion (viewer only)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No MuJoCo viewer / flex / FK windows (for batch IK comparison)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many simulation seconds (required-ish for reports)",
    )
    parser.add_argument(
        "--log-ik",
        type=str,
        default=None,
        help="CSV path for per-step IK error log (time, phase, dx, dy, dz, dist)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Markdown report path summarizing IK tracking errors",
    )
    return parser.parse_args()


def _summarize_ik_log(rows: list[dict[str, float | str]]) -> dict[str, float]:
    ik_rows = [r for r in rows if r["phase"] == "thumb_ik"]
    if not ik_rows:
        return {
            "n_steps": 0.0,
            "n_ik_steps": 0.0,
            "final_dist": float("nan"),
            "min_dist": float("nan"),
            "mean_dist": float("nan"),
            "mean_abs_dx": float("nan"),
            "mean_abs_dy": float("nan"),
            "mean_abs_dz": float("nan"),
            "t_reach": float("nan"),
        }
    dists = np.asarray([float(r["dist"]) for r in ik_rows], dtype=np.float64)
    dx = np.asarray([float(r["dx"]) for r in ik_rows], dtype=np.float64)
    dy = np.asarray([float(r["dy"]) for r in ik_rows], dtype=np.float64)
    dz = np.asarray([float(r["dz"]) for r in ik_rows], dtype=np.float64)
    reached = [r for r in ik_rows if bool(r["reached"])]
    t_reach = float(reached[0]["time"]) if reached else float("nan")
    return {
        "n_steps": float(len(rows)),
        "n_ik_steps": float(len(ik_rows)),
        "final_dist": float(dists[-1]),
        "min_dist": float(np.nanmin(dists)),
        "mean_dist": float(np.nanmean(dists)),
        "mean_abs_dx": float(np.nanmean(np.abs(dx))),
        "mean_abs_dy": float(np.nanmean(np.abs(dy))),
        "mean_abs_dz": float(np.nanmean(np.abs(dz))),
        "t_reach": t_reach,
    }


def _write_report(
    path: Path,
    *,
    ik_backend: str,
    summary: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Cyringe IK report — `{ik_backend}`",
        "",
        "## Config",
        "",
        f"- IK backend: `{ik_backend}`",
        f"- sticky: `{args.sticky}`",
        f"- close_duration: `{args.close_duration}` s",
        f"- delta_ik_z: `{args.delta_ik_z}` m",
        f"- reach_tol: `{DEFAULT_IK_REACH_TOL}` m",
        f"- duration: `{args.duration}` s",
        "",
        "## Tracking error (thumb EE → IK goal)",
        "",
        f"| metric | value |",
        f"| --- | --- |",
        f"| IK steps | {int(summary['n_ik_steps'])} |",
        f"| mean ‖e‖ [m] | {summary['mean_dist']:.6f} |",
        f"| min ‖e‖ [m] | {summary['min_dist']:.6f} |",
        f"| final ‖e‖ [m] | {summary['final_dist']:.6f} |",
        f"| mean |dx| [m] | {summary['mean_abs_dx']:.6f} |",
        f"| mean |dy| [m] | {summary['mean_abs_dy']:.6f} |",
        f"| mean |dz| [m] | {summary['mean_abs_dz']:.6f} |",
        f"| time to reach [s] | {summary['t_reach']:.4f} |",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    visualize_force = bool(args.visualize_force) and not bool(args.no_visualize_force)
    headless = bool(args.headless)
    ik_mod = load_ik_backend(args.ik)

    if not SCENE_XML.is_file():
        raise FileNotFoundError(f"Scene not found: {SCENE_XML}")

    # Includes and mesh assets are resolved relative to the XML directory.
    spec = mj.MjSpec.from_file(SCENE_XML.as_posix())
    # Cyringe long axis is +Z; default pitch 90° about Y lays it on its side.
    # --delta-rot-z yaws about world Z after that.
    spawn_offset = (
        float(args.offset[0]) + float(args.delta_x),
        float(args.offset[1]) + float(args.delta_y),
        float(args.offset[2]) + float(args.delta_z),
    )
    spawn_euler = (0.0, np.pi / 2, -np.pi / 2 + float(args.delta_rot_z))
    housing, spawn_pos = add_cyringe(
        spec,
        above_palm=args.flex is None,
        flex_name=args.flex,
        scale=args.scale,
        offset=spawn_offset,
        euler=spawn_euler,
        sticky=args.sticky,
    )
    ee_site = add_flex_center_ee_site(
        spec,
        DEFAULT_THUMB_FLEX,
        site_name=DEFAULT_THUMB_EE_SITE,
    )
    goal_mocap = add_ik_goal_mocap(spec, name=DEFAULT_IK_GOAL_MOCAP)
    traj_mocap_names: list[str] | None = None
    if args.trajectory:
        traj_mocap_names = add_trajectory_mocaps(
            spec, n_waypoints=int(args.traj_waypoints)
        )
    spawn_target = args.flex if args.flex is not None else "palm"
    print(
        f"  cyringe spawn pos={tuple(np.round(spawn_pos, 4))} "
        f"name={housing.name} scale={args.scale} pitch=90deg on={spawn_target} "
        f"offset={tuple(np.round(spawn_offset, 4))} "
        f"delta_xyz=({args.delta_x}, {args.delta_y}, {args.delta_z}) "
        f"delta_rot_z={args.delta_rot_z} sticky={args.sticky}"
        f" ee_site={ee_site} goal_mocap={goal_mocap} ik={args.ik}"
        f" trajectory={bool(args.trajectory)}"
    )

    model = spec.compile()
    model.opt.iterations = SOLVER_ITERATIONS
    model.opt.tolerance = 0.0
    data = mj.MjData(model)
    _set_actuator_initial(model, data, "th_axl_act", TH_AXL_ACT_INITIAL)
    mj.mj_forward(model, data)

    print(f"Loaded: {SCENE_XML.name}")
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}")
    print(f"  flexes ({model.nflex}): {', '.join(list_flex_names(model))}")
    print(f"  visualize_force={visualize_force}  headless={headless}")
    if args.sticky:
        print("  Tip: housing is fixed in world; shaft still slides under contact.")
    else:
        print("  Tip: pause the viewer (space) then drag the cyringe onto a pad/tip.")

    viz = None
    force_est = None
    fk_viz = None
    if not headless:
        from util.flex_visualizer import visualize_all_flexes_live
        from util.fk_taxel_util import (
            compute_fk_taxels,
            create_fk_taxel_visualizer,
            flex_forces_to_taxel_forces,
            read_leap_joint_angles,
        )

        viz = visualize_all_flexes_live(
            model,
            data,
            channel="magnitude",
            vmax=args.vmax,
            update_hz=15.0,
            visualize_force=visualize_force,
        )
        force_est = AllFlexForceEstimator(model, window=5, use_qvel=True)
        if not args.no_fk_viz:
            fk_viz = create_fk_taxel_visualizer()
            print("  FK visualizer: Open3D window (F deform, V vectors, Q quit)")

    contact_keys, flex_finger, geom_finger, cyringe_geoms = (
        _cache_fingertip_cyringe_ids(model, housing.name)
    )
    prev_contacts = {k: False for k in contact_keys}
    housing_body_id, squeeze_dofadr = _cyringe_motion_ids(model, housing.name)
    cyringe_was_moving = False
    last_motion_print_t = -1.0

    motion = None
    last_phase = None
    reached_announced = False
    motion_complete_announced = False
    if not args.no_motion:
        motion_kwargs = dict(
            close_duration=args.close_duration,
            mcp_target=args.mcp_target,
            pip_target=args.pip_target,
            dip_target=args.dip_target,
            handle_site=args.handle_site,
            ee_site=ee_site,
            reach_tol=DEFAULT_IK_REACH_TOL,
            delta_ik_z=args.delta_ik_z,
            goal_mocap=goal_mocap,
        )
        if args.trajectory:
            motion_kwargs.update(
                traj_stroke=args.traj_stroke,
                traj_waypoints=int(args.traj_waypoints),
                traj_mocap_names=traj_mocap_names,
            )
        motion = ik_mod.cyringe_two_phase_motion(model, data, **motion_kwargs)
        if args.trajectory:
            stroke = (
                args.traj_stroke
                if args.traj_stroke is not None
                else (
                    abs(args.delta_ik_z)
                    if abs(args.delta_ik_z) > 1e-9
                    else DEFAULT_TRAJ_STROKE
                )
            )
            print(
                f"  motion: phase1 MF/RF close "
                f"(mcp={args.mcp_target}, pip={args.pip_target}, "
                f"dip={args.dip_target}, {args.close_duration}s), "
                f"then thumb IK [{args.ik}] follows precomputed trajectory "
                f"along {args.handle_site} +Z "
                f"(stroke={stroke} m, waypoints={args.traj_waypoints}, "
                f"tol={DEFAULT_IK_REACH_TOL} m)"
            )
        else:
            print(
                f"  motion: phase1 MF/RF close "
                f"(mcp={args.mcp_target}, pip={args.pip_target}, "
                f"dip={args.dip_target}, {args.close_duration}s), "
                f"then thumb IK [{args.ik}] → {args.handle_site} "
                f"+ z*{args.delta_ik_z} (tol={DEFAULT_IK_REACH_TOL} m)"
            )

    log_rows: list[dict[str, float | str | bool]] = []
    log_path = Path(args.log_ik) if args.log_ik else None
    report_path = Path(args.report) if args.report else None
    duration = float(args.duration) if args.duration is not None else None
    if headless and duration is None:
        duration = 5.0
        print(f"  headless default duration={duration}s")

    def _step_once() -> bool:
        nonlocal last_phase, reached_announced, motion_complete_announced
        nonlocal prev_contacts, cyringe_was_moving, last_motion_print_t
        if motion is not None:
            info = next(motion)
            phase = str(info["phase"])
            if phase != last_phase:
                print(f"  motion phase → {phase}")
                last_phase = phase
            if (
                phase == "thumb_ik"
                and bool(info["reached"])
                and not reached_announced
            ):
                print(
                    f"  thumb flex touched handle "
                    f"(dist={float(info['dist']):.4f} m, "
                    f"vertex={int(info['ee_vertex'])})"
                )
                reached_announced = True
            if (
                phase == "thumb_ik"
                and bool(info["reached"])
                and not motion_complete_announced
            ):
                print("  motion complete")
                motion_complete_announced = True
            err = np.asarray(info.get("err_xyz", [np.nan, np.nan, np.nan]), dtype=np.float64)
            log_rows.append(
                {
                    "time": float(data.time),
                    "phase": phase,
                    "dx": float(err[0]),
                    "dy": float(err[1]),
                    "dz": float(err[2]),
                    "dist": float(info["dist"]),
                    "reached": bool(info["reached"]),
                }
            )
        mj.mj_step(model, data)
        contacts = _fingertip_cyringe_contacts(
            model,
            data,
            contact_keys,
            flex_finger,
            geom_finger,
            cyringe_geoms,
        )
        for finger, hitting in contacts.items():
            if hitting and not prev_contacts[finger]:
                print(
                    f"  contact: {finger} fingertip ↔ cyringe "
                    f"(t={float(data.time):.3f} s)"
                )
        prev_contacts = contacts
        moving, v_house, v_squeeze = _cyringe_is_moving(
            data, housing_body_id, squeeze_dofadr
        )
        t = float(data.time)
        if moving and (
            not cyringe_was_moving or t - last_motion_print_t >= 0.25
        ):
            print(
                f"  cyringe moving: True  housing |v|={v_house:.4f} m/s  "
                f"squeeze qvel={v_squeeze:.4f} m/s  (t={t:.3f} s)"
            )
            last_motion_print_t = t
        elif cyringe_was_moving and not moving:
            print(
                f"  cyringe moving: False  housing |v|={v_house:.4f} m/s  "
                f"squeeze qvel={v_squeeze:.4f} m/s  (t={t:.3f} s)"
            )
            last_motion_print_t = t
        cyringe_was_moving = moving
        if duration is not None and float(data.time) >= duration:
            return False
        return True

    try:
        if headless:
            while _step_once():
                pass
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    if fk_viz is not None and not fk_viz.poll():
                        break
                    step_start = time.time()
                    if not _step_once():
                        break
                    viewer.sync()
                    if force_est is not None and viz is not None:
                        forces = force_est.update(model, data)
                        if fk_viz is not None:
                            from util.fk_taxel_util import (
                                compute_fk_taxels,
                                flex_forces_to_taxel_forces,
                                read_leap_joint_angles,
                            )

                            joint_angles = read_leap_joint_angles(model, data)
                            taxel_forces = flex_forces_to_taxel_forces(model, forces)
                            fk_result = compute_fk_taxels(joint_angles, taxel_forces)
                            fk_viz.update(fk_result)
                        if visualize_force:
                            viz.update(model, data, forces=forces)
                        else:
                            viz.update(model, data)
                    leftover = model.opt.timestep - (time.time() - step_start)
                    if leftover > 0:
                        time.sleep(leftover)
    finally:
        if viz is not None:
            viz.close()
        if fk_viz is not None:
            fk_viz.close()

    if log_path is not None and log_rows:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["time", "phase", "dx", "dy", "dz", "dist", "reached"]
            )
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"  wrote IK log: {log_path}")

    summary = _summarize_ik_log(log_rows)
    if report_path is not None:
        _write_report(report_path, ik_backend=args.ik, summary=summary, args=args)
        print(f"  wrote IK report: {report_path}")
    elif log_rows:
        print(
            f"  IK summary [{args.ik}]: mean_dist={summary['mean_dist']:.4f} m "
            f"final={summary['final_dist']:.4f} m "
            f"mean_|dxyz|=({summary['mean_abs_dx']:.4f}, "
            f"{summary['mean_abs_dy']:.4f}, {summary['mean_abs_dz']:.4f})"
        )


if __name__ == "__main__":
    main()
