"""LeapFlexCyringeEnv demo: two-phase grasp with a live plot of ``info["metrics"]``.

Motion matches ``demos/env_cyringe_demo.py`` (MF/RF close, then pyroki thumb IK
along the handle +Z), while the fingertip positions and hand joint angles coming
back in ``info["metrics"]`` are plotted live.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import matplotlib.pyplot as plt
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

FINGERS = ("if", "mf", "rf", "th")
FINGER_COLORS = {"if": "tab:blue", "mf": "tab:orange", "rf": "tab:green", "th": "tab:red"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LeapFlexCyringeEnv two-phase grasp (same motion as env_cyringe_demo.py) "
            "with a live plot of the fingertip positions and hand joint angles from "
            "info['metrics']."
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
        "--history",
        type=int,
        default=400,
        help="Number of steps kept in the rolling plot window",
    )
    parser.add_argument(
        "--plot-hz",
        type=float,
        default=10.0,
        help="Max plot refresh rate (0 = redraw every step)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without the MuJoCo viewer (the live plot still opens)",
    )
    return parser.parse_args()


class MetricsLivePlot:
    """Rolling live plot of fingertip positions and hand joint angles."""

    def __init__(
        self,
        tip_names: list[str],
        joint_names: list[str],
        *,
        history: int = 400,
        update_hz: float = 20.0,
    ) -> None:
        self.tip_names = tip_names
        self.joint_names = joint_names
        self._min_dt = 1.0 / update_hz if update_hz > 0 else 0.0
        self._last_draw = -np.inf

        self._steps: deque[int] = deque(maxlen=history)
        self._tip_hist = {
            name: tuple(deque(maxlen=history) for _ in range(3)) for name in tip_names
        }
        self._joint_hist = {name: deque(maxlen=history) for name in joint_names}

        plt.ion()
        self.fig = plt.figure(figsize=(14, 7), layout="constrained")
        try:
            self.fig.canvas.manager.set_window_title("cyringe env metrics")
        except Exception:
            pass
        grid = self.fig.add_gridspec(2, 4)

        self._tip_axes = [self.fig.add_subplot(grid[0, i]) for i in range(3)]
        self._tip_lines: dict[str, list] = {}
        for axis, label in zip(self._tip_axes, "xyz"):
            axis.set_title(f"fingertip {label} (world)", fontsize=9)
            axis.set_ylabel("m")
            axis.grid(True, alpha=0.3)
        for name in tip_names:
            color = FINGER_COLORS.get(name.split("_", 1)[0])
            self._tip_lines[name] = [
                axis.plot([], [], color=color, label=name)[0] for axis in self._tip_axes
            ]
        self._tip_axes[0].legend(loc="upper left", fontsize=7, ncol=2)

        self._ax_spread = self.fig.add_subplot(grid[0, 3])
        self._ax_spread.set_title("tip displacement since reset", fontsize=9)
        self._ax_spread.set_ylabel("m")
        self._ax_spread.grid(True, alpha=0.3)
        self._tip_origin: dict[str, np.ndarray] = {}
        self._disp_hist = {name: deque(maxlen=history) for name in tip_names}
        self._disp_lines = {
            name: self._ax_spread.plot(
                [], [], color=FINGER_COLORS.get(name.split("_", 1)[0]), label=name
            )[0]
            for name in tip_names
        }

        self._joint_axes = {}
        self._joint_lines = {}
        for i, finger in enumerate(FINGERS):
            axis = self.fig.add_subplot(grid[1, i])
            axis.set_title(f"{finger} joints", fontsize=9)
            axis.set_xlabel("env step")
            axis.grid(True, alpha=0.3)
            if i == 0:
                axis.set_ylabel("rad")
            self._joint_axes[finger] = axis
            for name in joint_names:
                if not name.startswith(f"{finger}_"):
                    continue
                self._joint_lines[name] = axis.plot(
                    [], [], label=name.split("_", 1)[1]
                )[0]
            axis.legend(loc="upper left", fontsize=7, ncol=2)

        self._status = self.fig.suptitle("")
        self.fig.show()
        self.fig.canvas.draw()
        # Constrained layout re-solves the whole grid on every draw, which costs
        # more than the sim step itself. Freeze it once the panels are placed.
        self.fig.set_layout_engine("none")
        self.fig.canvas.flush_events()

    @property
    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def mark_reset(self, metrics: dict | None = None) -> None:
        """Re-anchor the displacement panel at the start of an episode."""
        self._tip_origin = {}
        if metrics is not None:
            self._tip_origin = {
                name: np.asarray(pos, dtype=np.float64)
                for name, pos in metrics["fingertips_pos"].items()
            }

    def push(self, step: int, metrics: dict) -> None:
        tips = metrics["fingertips_pos"]
        joints = metrics["hand_joint_values"]
        if not self._tip_origin:
            self._tip_origin = {
                name: np.asarray(pos, dtype=np.float64) for name, pos in tips.items()
            }
        self._steps.append(step)
        for name, pos in tips.items():
            pos = np.asarray(pos, dtype=np.float64)
            for axis_idx in range(3):
                self._tip_hist[name][axis_idx].append(float(pos[axis_idx]))
            self._disp_hist[name].append(
                float(np.linalg.norm(pos - self._tip_origin[name]))
            )
        for name, value in joints.items():
            self._joint_hist[name].append(float(value))

    def draw(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_draw) < self._min_dt:
            return
        if not self.is_open:
            return
        steps = np.asarray(self._steps, dtype=np.float64)
        for name, lines in self._tip_lines.items():
            for axis_idx, line in enumerate(lines):
                line.set_data(steps, np.asarray(self._tip_hist[name][axis_idx]))
        for name, line in self._disp_lines.items():
            line.set_data(steps, np.asarray(self._disp_hist[name]))
        for name, line in self._joint_lines.items():
            line.set_data(steps, np.asarray(self._joint_hist[name]))
        for axis in (*self._tip_axes, self._ax_spread, *self._joint_axes.values()):
            axis.relim()
            axis.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._last_draw = now

    def set_status(self, text: str) -> None:
        self._status.set_text(text)

    def close(self) -> None:
        if self.is_open:
            plt.close(self.fig)


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

    obs, _ = env.reset(seed=args.seed)
    metrics0 = env._get_metrics()
    plot = MetricsLivePlot(
        list(metrics0["fingertips_pos"]),
        list(metrics0["hand_joint_values"]),
        history=args.history,
        update_hz=args.plot_hz,
    )

    global_step = 0

    def _run_episode(ep: int, viewer=None) -> bool:
        """Return False when the user closed the viewer or the plot window."""
        nonlocal global_step
        if ep > 0:
            env.reset()
        plot.mark_reset(env._get_metrics())
        motion = ik_mod.cyringe_two_phase_motion(env.model, env.data, **motion_kwargs)
        last_phase = None
        reached_announced = False
        for t in range(args.steps):
            if viewer is not None and not viewer.is_running():
                return False
            if not plot.is_open:
                return False
            step_start = time.time()
            info_m = next(motion)
            phase = str(info_m["phase"])
            if phase != last_phase:
                print(f"\n  motion phase → {phase}")
                last_phase = phase
            if phase == "thumb_ik" and bool(info_m["reached"]) and not reached_announced:
                print(
                    f"\n  thumb flex touched handle "
                    f"(dist={float(info_m['dist']):.4f} m, "
                    f"vertex={int(info_m['ee_vertex'])})"
                )
                reached_announced = True
            action = _ctrl_to_action(env)
            _obs, reward, terminated, truncated, step_info = env.step(action)
            global_step += 1

            plot.push(global_step, step_info["metrics"])
            plot.set_status(
                f"episode {ep + 1}/{n_episodes}  t={t + 1}  phase={phase}  "
                f"reward={reward:.3f}"
            )
            plot.draw()

            if viewer is not None:
                viewer.sync()
                leftover = env.model.opt.timestep * env._n_substeps - (
                    time.time() - step_start
                )
                if leftover > 0:
                    time.sleep(leftover)
            if terminated or truncated:
                print(
                    f"\n  episode ended at t={t}  terminated={terminated} "
                    f"truncated={truncated}  "
                    f"cause={step_info.get('termination_cause')}"
                )
                break
        plot.draw(force=True)
        return True

    try:
        if args.no_viewer:
            for ep in range(n_episodes):
                if not _run_episode(ep):
                    break
        else:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                for ep in range(n_episodes):
                    if not viewer.is_running() or not _run_episode(ep, viewer=viewer):
                        break
        if plot.is_open:
            print("\nrollout done — close the plot window to exit")
            plt.ioff()
            plt.show()
    finally:
        plot.close()
        env.close()


if __name__ == "__main__":
    main()
