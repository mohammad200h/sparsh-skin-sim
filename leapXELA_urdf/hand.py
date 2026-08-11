from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

try:
    # When imported as part of the `xela_description` Python package.
    from .list_to_string import list_to_string
    from .finger import generate_finger, joint_urdf as finger_joint_urdf, link_urdf as finger_link_urdf
    from .thumb import generate_thumb, joint_urdf as thumb_joint_urdf, link_urdf as thumb_link_urdf
    from .palm import get_palm_constant
except ImportError:  # pragma: no cover
    # When executed directly as a script.
    from list_to_string import list_to_string
    from finger import generate_finger, joint_urdf as finger_joint_urdf, link_urdf as finger_link_urdf
    from thumb import generate_thumb, joint_urdf as thumb_joint_urdf, link_urdf as thumb_link_urdf
    from palm import get_palm_constant

def base_world_xml():
    return f""" <link name="base"/>
  <joint name="base_to_base" type="fixed">
    <parent link="base"/>
    <child link="leap_hand_xela_back_cover"/>
    <origin rpy="0 0 0" xyz="-0.007 -0.028 -0.039"/>
  </joint>
"""

JOINT_CONFIG_FILE = "../joint_config.json"

def load_joint_config(file_path: str) -> dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading joint config: {e}")
        return None

class BasePalmJoint:
    def joint_type(self) -> str:
        return "revolute"

    def parent_link_name(self) -> str:
        return "leap_hand_xela_back_cover"

    def axis(self) -> list[float]:
        return [0.0, 0.0, 1.0]


class FingersMCPJoint(BasePalmJoint):
    def __init__(self, prefix: str, pos: list[float], joint_config: dict[str, Any]):
        self.prefix = prefix
        valid_prefixes = ["rf", "mf", "if"]
        if prefix not in valid_prefixes:
            raise ValueError(f"Invalid prefix: {prefix} valid prefixes are {valid_prefixes}")
        self.joint_config = joint_config
        self.pos = pos

    def joint_name(self) -> str:
        return f"{self.prefix}_mcp"

    def child_link_name(self) -> str:
        return f"{self.prefix}_mcp"

    def origin(self) -> dict[str, Any]:
        return {"xyz": self.pos, "rpy": [3.14159, -1.5708, 0.0]}

    def limit(self) -> dict[str, Any]:
        return {"effort": self.joint_config["effort"], "velocity": self.joint_config["velocity"], "lower": self.joint_config["lower"], "upper": self.joint_config["upper"]}



class ThumbCMCJoint(BasePalmJoint):
    def __init__(self, joint_config: dict[str, Any]):
        self.joint_config = joint_config
    def joint_name(self) -> str:
        return "th_cmc"

    def child_link_name(self) -> str:
        return "cmc"

    def origin(self) -> dict[str, Any]:
        return {"xyz": [0.0480627, -0.013, 0.07], "rpy": [-3.14159, 0.0, 3.14159]}

    def limit(self) -> dict[str, Any]:
        return {"effort": self.joint_config["effort"], "velocity": self.joint_config["velocity"], "lower": self.joint_config["lower"], "upper": self.joint_config["upper"]}



def generate_mf_rf_if_links_and_joints() -> tuple[str, str]:
    rf_finger = generate_finger("rf" ,0.0, teleop=True)
    mf_finger = generate_finger("mf",0.0, teleop=True)
    if_finger = generate_finger("if",0.0, teleop=True)

    fingers = [rf_finger, mf_finger, if_finger]

    fingers_links = "\n".join(finger_link_urdf(link) for finger in fingers for link in finger.links)
    joints_urdf = "\n".join(finger_joint_urdf(joint) for finger in fingers for joint in finger.joints)

    return fingers_links, joints_urdf

def generate_thumb_links_and_joints() -> tuple[str, str]:
    thumb = generate_thumb(teleop = True)
    thumb_links = "\n".join(thumb_link_urdf(link) for link in thumb.links)
    thumb_joints = "\n".join(thumb_joint_urdf(joint) for joint in thumb.joints)
    return thumb_links, thumb_joints

def joint_urdf(joint: Any) -> str:
    limit = joint.limit()
    origin = joint.origin()
    return f"""
  <joint name="{joint.joint_name()}" type="{joint.joint_type()}">
    <origin xyz="{list_to_string(origin["xyz"])}" rpy="{list_to_string(origin["rpy"])}"/>
    <parent link="{joint.parent_link_name()}"/>
    <child link="{joint.child_link_name()}"/>
    <axis xyz="{list_to_string(joint.axis())}"/>
    <limit effort="{limit["effort"]}" velocity="{limit["velocity"]}" lower="{limit["lower"]}" upper="{limit["upper"]}"/>
  </joint>
""".rstrip("\n")

def generate_palm_joints() -> str:
    joint_config = load_joint_config(JOINT_CONFIG_FILE)["leapXela"]["sim"]
    fingers_config = joint_config["fingers"]["mcp"]
    thumb_config = joint_config["thumb"]["cmc"]
    palm_joints = [
        FingersMCPJoint("if", [0.06773, -0.0165, 0.1436] , fingers_config), 
        FingersMCPJoint("mf", [0.02228, -0.0165, 0.1436] ,fingers_config), 
        FingersMCPJoint("rf", [-0.02317, -0.0165, 0.1436],fingers_config),
        ThumbCMCJoint(thumb_config)
    ]
    return "\n".join(joint_urdf(j) for j in palm_joints)

def render_hand_joints_urdf() -> str:
    fingers_links, finger_joints = generate_mf_rf_if_links_and_joints()
    thumb_links, thumb_joints = generate_thumb_links_and_joints()
    palm_link = get_palm_constant()
    palm_joints = generate_palm_joints()
    return f"""
  <?xml version="1.0" ?>
  <robot name="xela_palm_generated">
  {base_world_xml()}
  {palm_link}
  {fingers_links}
  {thumb_links}
  {thumb_joints}
  {finger_joints}
  {palm_joints}

  </robot>
  """.rstrip("\n")

def write_hand_urdf(file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(render_hand_joints_urdf())


if __name__ == "__main__":
    import os

    out_path = os.environ.get("OUT") or "hand.urdf"
    write_hand_urdf(out_path)

