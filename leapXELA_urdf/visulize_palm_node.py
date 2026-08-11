from __future__ import annotations

import argparse
import time
from pathlib import Path


def _default_urdf_path() -> str:
    return str(Path(__file__).resolve().parent / "palm.urdf")


def _run_pybullet(urdf_path: str, use_gui: bool = True) -> None:
    try:
        import pybullet as p  # type: ignore
        import pybullet_data  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyBullet is required. Install it (e.g. `sudo apt install python3-pybullet` "
            "or `pip install pybullet`)."
        ) from e

    if not str(urdf_path).strip():
        urdf_path = _default_urdf_path()

    urdf = Path(urdf_path).expanduser().resolve()
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    if urdf.is_dir():
        raise IsADirectoryError(
            f"URDF path points to a directory, not a file: {urdf}"
        )

    cid = p.connect(p.GUI if use_gui else p.DIRECT)
    try:
        data_path = Path(pybullet_data.getDataPath()).resolve()
        p.setAdditionalSearchPath(str(data_path))
        p.setAdditionalSearchPath(str(urdf.parent))
        p.setGravity(0, 0, -9.81)
        try:
            plane_path = data_path / "plane.urdf"
            p.loadURDF(str(plane_path) if plane_path.exists() else "plane.urdf")
        except Exception:
            pass

        p.loadURDF(
            str(urdf),
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
        )

        if use_gui:
            while p.isConnected(cid):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
        else:
            for _ in range(240):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
    finally:
        try:
            p.disconnect(cid)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize palm URDF in PyBullet.")
    parser.add_argument(
        "--urdf",
        default=None,
        help="Path to URDF (default: palm.urdf in this directory)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use DIRECT connection instead of GUI",
    )
    args = parser.parse_args()

    urdf_path = args.urdf or _default_urdf_path()
    use_gui = not args.direct
    print(f"Loading URDF in PyBullet: {urdf_path}")
    print(f"PyBullet GUI: {use_gui}")
    _run_pybullet(urdf_path=urdf_path, use_gui=use_gui)


if __name__ == "__main__":
    main()
