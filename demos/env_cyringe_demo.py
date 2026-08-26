"""LeapFlexCyringeEnv demo: MF/RF close + pyroki thumb IK, fed as env actions."""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

import _bootstrap  # noqa: F401
from env import LeapFlexCyringeEnv
from util.ik_common import (
    CYRINGE_MF_RF_DIP,
    CYRINGE_MF_RF_MCP,
    CYRINGE_MF_RF_PIP,
    DEFAULT_DELTA_IK_Z,
    DEFAULT_HANDLE_SITE,
    DEFAULT_IK_REACH_TOL,
    load_ik_backend,
)
from util.motion_util import DEFAULT_CLOSE_DURATION
from util.trajectory_generation import DEFAULT_TRAJ_STROKE, DEFAULT_TRAJ_WAYPOINTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LeapFlexCyringeEnv: same two-phase grasp as demo_cyringe.py, "
            "with pyroki thumb IK and MF/RF close, written as env actions"
        )
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000000,
        help="Env steps per episode",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of reset→rollout episodes (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Passed to env.reset for the first episode",
    )
    parser.add_argument(
        "--n-substeps",
        type=int,
        default=1,
        help="Physics substeps per env action (use 1 to match demo_cyringe.py)",
    )
    parser.add_argument(
        "--sticky",
        action="store_true",
        help="Fix the cyringe housing in world (shaft can still slide)",
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
        help=f"Thumb IK target site (default: {DEFAULT_HANDLE_SITE})",
    )
    parser.add_argument(
        "--delta-ik-z",
        type=float,
        default=DEFAULT_DELTA_IK_Z,
        help="Offset [m] along handle +Z for the live IK goal",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        default=True,
        help="Follow a precomputed push trajectory along handle +Z (default on)",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Live handle+δz goal instead of a waypoint trajectory",
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
        help=f"Trajectory waypoint / mocap count (default: {DEFAULT_TRAJ_WAYPOINTS})",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless for --steps instead of opening the MuJoCo viewer",
    )
    return parser.parse_args()


def _peak_force(obs: dict) -> float:
    peak = 0.0
    for force in obs["flex_force"].values():
        if force.size:
            peak = max(peak, float(np.max(np.linalg.norm(force, axis=1))))
    return peak


def _ctrl_to_action(env: LeapFlexCyringeEnv) -> np.ndarray:
    return np.asarray(env.data.ctrl[env._actuator_ids], dtype=np.float32)


def main() -> None:
    args = parse_args()
    use_trajectory = bool(args.trajectory) and not bool(args.no_trajectory)
    ik_mod = load_ik_backend("pyroki")

    env = LeapFlexCyringeEnv(
        n_substeps=args.n_substeps,
        cyringe={"sticky": True} if args.sticky else None,
        ik_helpers=True,
        n_traj_waypoints=int(args.traj_waypoints) if use_trajectory else None,
    )
    n_episodes = max(1, int(args.episodes))
    print(f"Loaded LeapFlexCyringeEnv  nu={env.model.nu}  nq={env.model.nq}")
    print(f"  housing: {env._cyringe_housing_name}")
    print(f"  ee_site={env.ee_site}  goal_mocap={env.goal_mocap}")
    print(f"  episodes={n_episodes}  steps/episode={args.steps}")

    motion_kwargs = dict(
        close_duration=args.close_duration,
        mcp_target=args.mcp_target,
        pip_target=args.pip_target,
        dip_target=args.dip_target,
        handle_site=args.handle_site,
        ee_site=env.ee_site,
        reach_tol=DEFAULT_IK_REACH_TOL,
        delta_ik_z=args.delta_ik_z,
        goal_mocap=env.goal_mocap,
    )
    if use_trajectory:
        motion_kwargs.update(
            traj_stroke=args.traj_stroke,
            traj_waypoints=int(args.traj_waypoints),
            traj_mocap_names=env.traj_mocap_names,
        )

    if use_trajectory:
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
            f"  motion: MF/RF close "
            f"(mcp={args.mcp_target}, pip={args.pip_target}, "
            f"dip={args.dip_target}, {args.close_duration}s), "
            f"then thumb IK [pyroki] along {args.handle_site} +Z "
            f"(stroke={stroke} m, waypoints={args.traj_waypoints}, "
            f"tol={DEFAULT_IK_REACH_TOL} m)"
        )
    else:
        print(
            f"  motion: MF/RF close "
            f"(mcp={args.mcp_target}, pip={args.pip_target}, "
            f"dip={args.dip_target}, {args.close_duration}s), "
            f"then thumb IK [pyroki] → {args.handle_site} "
            f"+ z*{args.delta_ik_z} (tol={DEFAULT_IK_REACH_TOL} m)"
        )

    def _run_episode(ep: int, viewer=None) -> None:
        obs, info = env.reset(seed=None if ep > 0 else args.seed)
        print(f"episode {ep + 1}/{n_episodes}  spawn={info['cyringe_spawn']}")
        print(f"  peak |flex_force| at reset: {_peak_force(obs):.4g} N")
        motion = ik_mod.cyringe_two_phase_motion(env.model, env.data, **motion_kwargs)
        last_phase = None
        reached_announced = False
        for t in range(args.steps):
            if viewer is not None and not viewer.is_running():
                return
            step_start = time.time()
            info_m = next(motion)
            phase = str(info_m["phase"])
            if phase != last_phase:
                print(f"  motion phase → {phase}")
                last_phase = phase
            if phase == "thumb_ik" and bool(info_m["reached"]) and not reached_announced:
                print(
                    f"  thumb flex touched handle "
                    f"(dist={float(info_m['dist']):.4f} m, "
                    f"vertex={int(info_m['ee_vertex'])})"
                )
                reached_announced = True
            action = _ctrl_to_action(env)
            obs, reward, terminated, truncated, step_info = env.step(action)
            if viewer is not None:
                viewer.sync()
                leftover = env.model.opt.timestep * env._n_substeps - (
                    time.time() - step_start
                )
                if leftover > 0:
                    time.sleep(leftover)
            elif t % 100 == 0:
                print(
                    f"  t={t:4d}  reward={reward:.3f}  "
                    f"peak_force={_peak_force(obs):.4g} N"
                )
            if terminated or truncated:
                cause = step_info.get("termination_cause")
                print(
                    f"  episode ended at t={t}  terminated={terminated} "
                    f"truncated={truncated}  cause={cause}"
                )
                break

    try:
        if args.no_viewer:
            for ep in range(n_episodes):
                _run_episode(ep)
        else:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                for ep in range(n_episodes):
                    if not viewer.is_running():
                        break
                    _run_episode(ep, viewer=viewer)
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(env.model.opt.timestep * env._n_substeps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
