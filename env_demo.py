"""Demo: LeapFlexEnv with a tetris piece on the palm and random joint actions."""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from env import LeapFlexEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LeapFlexEnv demo: tetris on palm + random joint actions"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of env steps (ignored with --viewer until the window closes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for random actions",
    )
    parser.add_argument(
        "--n-substeps",
        type=int,
        default=5,
        help="Physics substeps per env action",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Tetris piece scale",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 0.0),
        help="World-frame spawn offset for the tetris piece (metres)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless for --steps instead of opening the MuJoCo viewer",
    )
    parser.add_argument(
        "--action-hold",
        type=int,
        default=40,
        help="Reuse each random action for this many env steps (smoother motion)",
    )
    return parser.parse_args()


def _peak_force(obs: dict) -> float:
    peak = 0.0
    for force in obs["flex_force"].values():
        if force.size:
            peak = max(peak, float(np.max(np.linalg.norm(force, axis=1))))
    return peak


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env = LeapFlexEnv(
        n_substeps=args.n_substeps,
        spawn_tetris=True,
        tetris_shape="T",
        tetris_scale=args.scale,
        tetris_offset=tuple(args.offset),
        tetris_flex=None,  # above palm pads
    )
    obs, _ = env.reset(seed=args.seed)
    print(f"Loaded LeapFlexEnv  nu={env.model.nu}  nq={env.model.nq}")
    print(f"  tetris body: {env._tetris_body_name}")
    print(f"  action range: [{env.action_space.low.min():.2f}, {env.action_space.high.max():.2f}]")
    print(f"  peak |flex_force| at reset: {_peak_force(obs):.4g} N")

    action = env.action_space.sample()
    hold_left = 0

    try:
        if args.no_viewer:
            for t in range(args.steps):
                if hold_left <= 0:
                    action = rng.uniform(env.action_space.low, env.action_space.high)
                    hold_left = max(1, args.action_hold)
                obs, _, _, _, _ = env.step(action)
                hold_left -= 1
                if t % 100 == 0:
                    print(
                        f"  t={t:4d}  peak_force={_peak_force(obs):.4g} N  "
                        f"fk_pos[0]={np.round(obs['flex_taxel_fk']['positions'][0], 3)}"
                    )
        else:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    step_start = time.time()
                    if hold_left <= 0:
                        action = rng.uniform(
                            env.action_space.low, env.action_space.high
                        ).astype(np.float32)
                        hold_left = max(1, args.action_hold)
                    env.step(action)
                    hold_left -= 1
                    viewer.sync()
                    leftover = env.model.opt.timestep * env._n_substeps - (
                        time.time() - step_start
                    )
                    if leftover > 0:
                        time.sleep(leftover)
    finally:
        env.close()


if __name__ == "__main__":
    main()
