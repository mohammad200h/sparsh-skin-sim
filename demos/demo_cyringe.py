"""Load the Leap+XELA flex-sensor scene with the cyringe and open the viewer."""

from __future__ import annotations

import argparse
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

from _bootstrap import SCENE_XML
from util.flex_visualizer import visualize_all_flexes_live
from util.flex_util import AllFlexForceEstimator, list_flex_names
from util.fk_taxel_util import (
    compute_fk_taxels,
    create_fk_taxel_visualizer,
    flex_forces_to_taxel_forces,
    read_leap_joint_angles,
)
from util.objects_util import add_cyringe

# Match the flex-sensor generator: the shipped MJX XML keeps iterations=5,
# which under-converges the soft skins so contact barely moves the joints.
SOLVER_ITERATIONS = 50
TH_AXL_ACT_INITIAL = 1.6


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leap+XELA cyringe demo")
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
        "--no-fk-viz",
        action="store_true",
        help="Disable the Open3D taxel FK visualizer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_force = bool(args.visualize_force) and not bool(args.no_visualize_force)

    if not SCENE_XML.is_file():
        raise FileNotFoundError(f"Scene not found: {SCENE_XML}")

    # Includes and mesh assets are resolved relative to the XML directory.
    spec = mj.MjSpec.from_file(SCENE_XML.as_posix())
    # Cyringe long axis is +Z; yaw (euler Z) barely changes appearance.
    # Pitch 90° about Y lays it on its side above the palm.
    housing, spawn_pos = add_cyringe(
        spec,
        above_palm=args.flex is None,
        flex_name=args.flex,
        scale=args.scale,
        offset=args.offset,
        euler=(0.0, np.pi / 2, -np.pi / 2),
    )
    spawn_target = args.flex if args.flex is not None else "palm"
    print(
        f"  cyringe spawn pos={tuple(np.round(spawn_pos, 4))} "
        f"name={housing.name} scale={args.scale} pitch=90deg on={spawn_target} "
        f"offset={tuple(args.offset)}"
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
    print(f"  visualize_force={visualize_force}")
    print("  Tip: pause the viewer (space) then drag the cyringe onto a pad/tip.")

    viz = visualize_all_flexes_live(
        model,
        data,
        channel="magnitude",
        vmax=args.vmax,
        update_hz=15.0,
        visualize_force=visualize_force,
    )
    force_est = AllFlexForceEstimator(model, window=5, use_qvel=True)

    fk_viz = None
    if not args.no_fk_viz:
        fk_viz = create_fk_taxel_visualizer()
        print("  FK visualizer: Open3D window (F deform, V vectors, Q quit)")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if fk_viz is not None and not fk_viz.poll():
                    break
                step_start = time.time()
                mj.mj_step(model, data)
                viewer.sync()
                forces = force_est.update(model, data)
                if fk_viz is not None:
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
        viz.close()
        if fk_viz is not None:
            fk_viz.close()


if __name__ == "__main__":
    main()
