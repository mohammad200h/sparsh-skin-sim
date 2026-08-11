"""Load the Leap+XELA MJX flex-sensor scene and open the MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

from util.flex_visualizer import visualize_all_flexes_live
from util.flex_util import AllFlexForceEstimator, list_flex_names
from util.fk_taxel_util import (
    compute_fk_taxels,
    create_fk_taxel_visualizer,
    flex_forces_to_taxel_forces,
    flex_vertex_for_fk_grid,
    read_leap_joint_angles,
)
from util.objects_util import (
    add_cube_on_taxel,
    add_sticky_cube_on_taxel,
    add_tetris_part,
    infer_flex_grid_size,
)


SCENE_XML = (
    Path(__file__).resolve().parent
    / "leapXELA_model"
    / "scene_mjx_cube_CoACD_mjx_flex_sensor.xml"
)

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
            "Target flex for object spawn (e.g. mf_tip or flex_uspa46_1). "
            "With --taxel-row/--taxel-col: drop a cube on that taxel "
            "(use --sticky to weld it). "
            "Otherwise: tetris piece above this flex (default: all palm pads)."
        ),
    )
    parser.add_argument(
        "--taxel-row",
        type=int,
        default=None,
        help="Taxel grid row (0-indexed). Requires --flex and --taxel-col.",
    )
    parser.add_argument(
        "--taxel-col",
        type=int,
        default=None,
        help="Taxel grid column (0-indexed). Requires --flex and --taxel-row.",
    )
    parser.add_argument(
        "--sticky",
        action="store_true",
        default=False,
        help=(
            "Weld the taxel cube to the vertex body (default: free-falling cube). "
            "Requires --taxel-row/--taxel-col."
        ),
    )
    parser.add_argument(
        "--taxel-mass",
        type=float,
        default=None,
        help="Mass of the taxel cube in kg",
    )
    parser.add_argument(
        "--cube-half-size",
        type=float,
        default=None,
        help="Taxel cube half-extent in metres",
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
        help=(
            "XYZ offset added to spawn pose (metres). "
            "Taxel cubes (sticky or free): taxel-local frame; tetris: world frame."
        ),
    )
    parser.add_argument(
        "--no-fk-viz",
        action="store_true",
        help="Disable the Open3D taxel FK visualizer",
    )
    args = parser.parse_args()
    taxel_args = (args.taxel_row, args.taxel_col)
    if any(v is not None for v in taxel_args) and not all(
        v is not None for v in taxel_args
    ):
        parser.error("--taxel-row and --taxel-col must be given together")
    if args.taxel_row is not None and args.flex is None:
        parser.error("--flex is required when placing a cube on a taxel")
    if args.sticky and args.taxel_row is None:
        parser.error("--sticky requires --taxel-row and --taxel-col")
    return args


def main() -> None:
    args = parse_args()
    use_taxel_cube = args.taxel_row is not None
    visualize_force = bool(args.visualize_force) and not bool(args.no_visualize_force)
    if use_taxel_cube and not args.visualize_force and not args.no_visualize_force:
        visualize_force = True

    if not SCENE_XML.is_file():
        raise FileNotFoundError(f"Scene not found: {SCENE_XML}")

    # Includes and mesh assets are resolved relative to the XML directory.
    spec = mj.MjSpec.from_file(SCENE_XML.as_posix())
    if use_taxel_cube:
        assert args.flex is not None
        assert args.taxel_row is not None and args.taxel_col is not None
        n_cols, n_rows = infer_flex_grid_size(args.flex)
        model_probe = spec.compile()
        vertex_index, taxel_id = flex_vertex_for_fk_grid(
            model_probe, args.flex, args.taxel_row, args.taxel_col
        )
        cube_kwargs: dict = {
            "flex_name": args.flex,
            "grid_row": args.taxel_row,
            "grid_col": args.taxel_col,
            "offset": args.offset,
        }
        if args.taxel_mass is not None:
            cube_kwargs["mass"] = args.taxel_mass
        if args.cube_half_size is not None:
            cube_kwargs["half_size"] = args.cube_half_size
        if args.sticky:
            piece = add_sticky_cube_on_taxel(spec, **cube_kwargs)
            mode = "sticky cube"
        else:
            # Spawn against the same initial hand pose used after compile
            # (sticky cubes follow the taxel; free cubes need the posed location).
            cube_kwargs["configure"] = (
                lambda model, data: _set_actuator_initial(
                    model, data, "th_axl_act", TH_AXL_ACT_INITIAL
                )
            )
            piece = add_cube_on_taxel(spec, **cube_kwargs)
            mode = "taxel cube"
        print(
            f"  {mode} pos={tuple(np.round(piece.pos, 4))} "
            f"name={piece.name} flex={args.flex} "
            f"grid=({args.taxel_row}, {args.taxel_col}) "
            f"size={n_rows}x{n_cols} vertex={vertex_index} taxel={taxel_id} "
            f"offset={tuple(args.offset)}"
        )
    else:
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
    _set_actuator_initial(model, data, "th_axl_act", TH_AXL_ACT_INITIAL)
    mj.mj_forward(model, data)

    print(f"Loaded: {SCENE_XML.name}")
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}")
    print(f"  flexes ({model.nflex}): {', '.join(list_flex_names(model))}")
    print(f"  visualize_force={visualize_force}")
    if use_taxel_cube:
        if args.sticky:
            print(
                "  Tip: sticky cube is welded to the chosen taxel; compare MuJoCo flex "
                "colours with the Open3D FK panel at the same grid cell."
            )
        else:
            print(
                "  Tip: cube falls onto the chosen taxel; compare MuJoCo flex colours "
                "with the Open3D FK panel at the same grid cell."
            )
    else:
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
