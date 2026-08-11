from __future__ import annotations

import time
from pathlib import Path

import rclpy
from rclpy.node import Node


def _default_urdf_path() -> str:
    from ament_index_python.packages import get_package_share_directory

    share_dir = Path(get_package_share_directory("xela_description"))
    return str(share_dir / "robot.urdf")


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
        # Some container installs miss optional pybullet_data assets; don't hard-fail on the ground plane.
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


class VisulizePybulletNode(Node):
    def __init__(self) -> None:
        super().__init__("visulize_pybullet")

        self.declare_parameter("urdf_path", _default_urdf_path())
        self.declare_parameter("use_gui", True)

        urdf_path = str(self.get_parameter("urdf_path").value)
        use_gui = bool(self.get_parameter("use_gui").value)

        self.get_logger().info(f"Loading URDF in PyBullet: {urdf_path}")
        self.get_logger().info(f"PyBullet GUI: {use_gui}")

        _run_pybullet(urdf_path=urdf_path, use_gui=use_gui)


def main() -> None:
    rclpy.init()
    node: VisulizePybulletNode | None = None
    try:
        node = VisulizePybulletNode()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

