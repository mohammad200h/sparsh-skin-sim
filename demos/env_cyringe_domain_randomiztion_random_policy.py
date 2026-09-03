"""LeapFlexCyringeEnv demo: domain-randomized reset + random joint actions."""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

import _bootstrap  # noqa: F401
from env import LeapFlexCyringeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LeapFlexCyringeEnv with a random joint policy. "
            "Pass --hand-pose and/or --object-pose to randomize those on reset."
        )
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Env steps per episode",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of reset→rollout episodes",
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
        help="Physics substeps per env action",
    )
    parser.add_argument(
        "--hand-pose",
        action="store_true",
        help="Enable hand-pose domain randomization on reset",
    )
    parser.add_argument(
        "--object-pose",
        action="store_true",
        help="Enable cyringe-pose domain randomization on reset",
    )
    parser.add_argument(
        "--sticky",
        action="store_true",
        help="Weld the cyringe housing in world (shaft can still slide)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless for --episodes × --steps instead of opening the viewer",
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


def _fmt_xyz(values: list[float] | tuple[float, ...] | None) -> str:
    if values is None:
        return "None"
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_episodes = max(1, int(args.episodes))
    env_kwargs: dict = {}
    if args.hand_pose or args.object_pose:
        env_kwargs["randomize_hand"] = bool(args.hand_pose)
        env_kwargs["randomize_cyringe"] = bool(args.object_pose)

    env = LeapFlexCyringeEnv(
        n_substeps=args.n_substeps,
        max_timestep=args.steps,
        ik_helpers=False,
        cyringe={"sticky": True} if args.sticky else None,
        **env_kwargs,
    )
    print(f"Loaded LeapFlexCyringeEnv  nu={env.model.nu}  nq={env.model.nq}")
    print(f"  housing: {env._cyringe_housing_name}")
    print(f"  sticky: {bool(args.sticky)}")
    print(f"  episodes={n_episodes}  steps/episode={args.steps}")
    if env._domain_randomization is None:
        print("  domain randomization: off")
    else:
        dr = env._domain_randomization
        print(
            "  domain randomization: "
            f"hand={dr.randomize_hand}  "
            f"cyringe={dr.randomize_cyringe}"
        )
        if dr.randomize_cyringe:
            print(f"  object offset low={list(dr.offset_low)}  high={list(dr.offset_high)}")
            print(
                f"  object euler Δ low={list(dr.euler_delta_low)}  "
                f"high={list(dr.euler_delta_high)}"
            )

    action = env.action_space.sample()
    hold_left = 0

    def _run_episode(ep: int, viewer=None) -> None:
        nonlocal action, hold_left
        obs, info = env.reset(seed=None if ep > 0 else args.seed)
        dr = info.get("domain_randomization") or {}
        print(f"episode {ep + 1}/{n_episodes}")
        print(f"  cyringe offset Δ={_fmt_xyz(dr.get('cyringe_offset'))}")
        print(f"  cyringe euler Δ={_fmt_xyz(dr.get('cyringe_euler_delta'))}")
        print(f"  cyringe pos={_fmt_xyz(dr.get('cyringe_pos'))}")
        hand = dr.get("hand_qpos")
        if hand is not None:
            print(
                f"  hand qpos[:4]={_fmt_xyz(hand[:4])}  "
                f"(16-DoF, peak |q|={max(abs(v) for v in hand):.3f})"
            )
        else:
            print("  hand qpos: default (th_axl_initial)")
        print(f"  peak |flex_force| at reset: {_peak_force(obs):.4g} N")
        hold_left = 0
        for t in range(args.steps):
            if viewer is not None and not viewer.is_running():
                return
            step_start = time.time()
            if hold_left <= 0:
                action = rng.uniform(
                    env.action_space.low, env.action_space.high
                ).astype(np.float32)
                hold_left = max(1, args.action_hold)
            obs, reward, terminated, truncated, step_info = env.step(action)
            hold_left -= 1
            flex_exploded = bool(step_info.get("flex_exploded"))
            sim_unstable = bool(step_info.get("sim_unstable"))
            if viewer is not None:
                viewer.sync()
                leftover = env.model.opt.timestep * env._n_substeps - (
                    time.time() - step_start
                )
                if leftover > 0:
                    time.sleep(leftover)
            elif t % 50 == 0:
                print(
                    f"  t={t:4d}  reward={reward:.3f}  "
                    f"peak_force={_peak_force(obs):.4g} N  "
                    f"flex exploded: {flex_exploded}  "
                    f"sim unstable: {sim_unstable}"
                )
            if terminated or truncated:
                cause = step_info.get("termination_cause")
                print(
                    f"  episode ended at t={t}  terminated={terminated} "
                    f"truncated={truncated}  cause={cause}"
                )
                print(f"  flex exploded: {flex_exploded}")
                print(f"  sim unstable: {sim_unstable}")
                break
        else:
            print("  flex exploded: False")
            print("  sim unstable: False")

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
