"""Load the Leap+XELA MJX flex-sensor scene and open the MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

from flex_util import AllFlexForceEstimator, list_flex_names
from flex_visualizer import visualize_all_flexes_live
from objects import add_tetris_part


SCENE_XML = (
    Path(__file__).resolve().parent
    / "leapXELA_model"
    / "scene_mjx_cube_CoACD_mjx_flex_sensor.xml"
)

# Match the flex-sensor generator: the shipped MJX XML keeps iterations=5,
# which under-converges the soft skins so contact barely moves the joints.
SOLVER_ITERATIONS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leap+XELA flex sensor demo")
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
            "Spawn the tetris piece above this flex "
            "(e.g. mf_tip or flex_uspa46_1). Default: all palm pads."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Uniform scale of the tetris piece (default: 1.5)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 0.0),
        help="XYZ offset added to the flex/palm spawn centre (metres)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_force = bool(args.visualize_force) and not bool(args.no_visualize_force)

    if not SCENE_XML.is_file():
        raise FileNotFoundError(f"Scene not found: {SCENE_XML}")

    # Includes and mesh assets are resolved relative to the XML directory.
    spec = mj.MjSpec.from_file(SCENE_XML.as_posix())
    piece = add_tetris_part(
        spec,
        shape="T",
        above_palm=args.flex is None,
        flex_name=args.flex,
        scale=args.scale,
        offset=args.offset,
        euler=(0.0, 0.0, np.pi / 4),
    )
    spawn_target = args.flex if args.flex is not None else "palm"
    print(
        f"  tetris spawn pos={tuple(np.round(piece.pos, 4))} "
        f"name={piece.name} scale={args.scale} yaw=45deg on={spawn_target} "
        f"offset={tuple(args.offset)}"
    )
    model = spec.compile()
    model.opt.iterations = SOLVER_ITERATIONS
    model.opt.tolerance = 0.0
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    print(f"Loaded: {SCENE_XML.name}")
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}")
    print(f"  flexes ({model.nflex}): {', '.join(list_flex_names(model))}")
    print(f"  visualize_force={visualize_force}")
    print("  Tip: pause the viewer (space) then drag the tetris piece onto a pad/tip.")

    viz = visualize_all_flexes_live(
        model,
        data,
        channel="magnitude",
        vmax=args.vmax,
        update_hz=15.0,
        visualize_force=visualize_force,
    )
    force_est = AllFlexForceEstimator(model, window=5, use_qvel=True)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                mj.mj_step(model, data)
                viewer.sync()
                forces = force_est.update(model, data)
                if visualize_force:
                    viz.update(model, data, forces=forces)
                else:
                    viz.update(model, data)
                leftover = model.opt.timestep - (time.time() - step_start)
                if leftover > 0:
                    time.sleep(leftover)
    finally:
        viz.close()


if __name__ == "__main__":
    main()
