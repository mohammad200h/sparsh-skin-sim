from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

try:
    # When imported as part of the `xela_description` Python package.
    from .list_to_string import list_to_string
except ImportError:  # pragma: no cover
    # When executed directly as a script.
    from list_to_string import list_to_string

JOINT_CONFIG_FILE = "../joint_config.json"

def load_joint_config(file_path: str) -> dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading joint config: {e}")
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class CMCLink:
    def link_name(self) -> str:
        return "cmc"

    def inertial(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [-0.000232587, 0.00882444, 0.0117326], "rpy": [0.0, 0.0, 0.0]},
            "mass": 0.032,
            "inertia": {
                "ixx": 3.63585e-06,
                "ixy": 2.0599e-08,
                "ixz": 8.412e-09,
                "iyy": 2.65288e-06,
                "iyz": -1.4817e-08,
                "izz": 4.33226e-06,
            },
        }

    def visual(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [0.0, 0.0075, 0.0], "rpy": [-1.5708, -1.5708, 0.0]},
            "geometry": {"mesh": {"filename": "package://assets/thp2_unified.stl"}},
            "material": {
                "name": "thp2_unified_material",
                "color": {"rgba": [0.980392, 0.713725, 0.00392157, 1.0]},
            },
        }

    def collision(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [0.0, 0.0075, 0.0], "rpy": [-1.5708, -1.5708, 0.0]},
            "geometry": {"mesh": {"filename": "package://assets/thp2_unified.stl"}},
        }


class AXLLink:
    def link_name(self) -> str:
        return "axl"

    def inertial(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [0.000911557, -0.00108086, 0.00811114], "rpy": [0.0, 0.0, 0.0]},
            "mass": 0.003,
            "inertia": {
                "ixx": 4.40805e-07,
                "ixy": -4.612e-09,
                "ixz": -4.6465e-08,
                "iyy": 1.184e-06,
                "iyz": 4.476e-09,
                "izz": 1.12942e-06,
            },
        }

    # `clipper` has two visuals/collisions in `robot.urdf`.
    def visual(self) -> list[dict[str, Any]]:
        return [
            {
                "origin": {"xyz": [-0.0173, 0.0, 0.019], "rpy": [1.5708, -0.0, 0.0]},
                "geometry": {"mesh": {"filename": "package://assets/thconnector.stl"}},
                "material": {
                    "name": "thconnector_material",
                    "color": {"rgba": [0.792157, 0.819608, 0.933333, 1.0]},
                },
            },
            {
                "origin": {"xyz": [0.0155, -0.006, 0.019], "rpy": [-3.14159, 0.0, 1.5708]},
                "geometry": {"mesh": {"filename": "package://assets/clipper.stl"}},
                "material": {
                    "name": "clipper_material",
                    "color": {"rgba": [0.792157, 0.819608, 0.933333, 1.0]},
                },
            },
        ]

    def collision(self) -> list[dict[str, Any]]:
        return [
            {
                "origin": {"xyz": [-0.0173, 0.0, 0.019], "rpy": [1.5708, -0.0, 0.0]},
                "geometry": {"mesh": {"filename": "package://assets/thconnector.stl"}},
            },
            {
                "origin": {"xyz": [0.0155, -0.006, 0.019], "rpy": [-3.14159, 0.0, 1.5708]},
                "geometry": {"mesh": {"filename": "package://assets/clipper.stl"}},
            },
        ]


class MCPLink:
    def link_name(self) -> str:
        return "th_mcp"

    def inertial(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [0.00144426, 0.0112332, 0.0144179], "rpy": [0.0, 0.0, 0.0]},
            "mass": 0.038,
            "inertia": {
                "ixx": 7.98976e-06,
                "ixy": 8.1084e-08,
                "ixz": -9.5788e-08,
                "iyy": 4.90108e-06,
                "iyz": 1.43138e-07,
                "izz": 7.77953e-06,
            },
        }

    def visual(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [-0.0, 0.0245, 0.0145], "rpy": [-0.0, -0.0, -0.0]},
            "geometry": {"mesh": {"filename": "package://assets/thp1_unified.stl"}},
            "material": {
                "name": "thp1_unified_material",
                "color": {"rgba": [0.647059, 0.647059, 0.647059, 1.0]},
            },
        }

    def collision(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [-0.0, 0.0245, 0.0145], "rpy": [-0.0, -0.0, -0.0]},
            "geometry": {"mesh": {"filename": "package://assets/thp1_unified.stl"}},
        }


class IPLLink:
    def link_name(self) -> str:
        return "th_ipl"

    def inertial(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [-0.00210586, -0.0274158, 0.0145397], "rpy": [0.0, 0.0, 0.0]},
            "mass": 0.049,
            "inertia": {
                "ixx": 3.16029e-05,
                "ixy": 2.85926e-07,
                "ixz": 1.022e-07,
                "iyy": 8.51257e-06,
                "iyz": -8.066e-08,
                "izz": 3.25689e-05,
            },
        }

    def visual(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [-0.0, -0.0275, 0.014504], "rpy": [1.5708, 0.0, -0.0]},
            "geometry": {"mesh": {"filename": "package://assets/thfingertip_unified.stl"}},
            "material": {
                "name": "thfingertip_unified_material",
                "color": {"rgba": [0.498039, 0.498039, 0.498039, 1.0]},
            },
        }

    def collision(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [-0.0, -0.0275, 0.014504], "rpy": [1.5708, 0.0, -0.0]},
            "geometry": {"mesh": {"filename": "package://assets/thfingertip_unified.stl"}},
        }


class THREALTIPLINK:
    def link_name(self) -> str:
        return "th_realtip"

    def inertial(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            "mass": 0.000001,
            "inertia": {
                "ixx": 0.000025,
                "ixy": 0.0,
                "ixz": 0.0,
                "iyy": 0.000025,
                "iyz": 0.0,
                "izz": 0.000025,
            },
        }
    def visual(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            "geometry": {"sphere": {"radius": 0.005}},
            "material": {
                "name": "ball_material",
                "color": {"rgba": [1.0, 0.0, 0.0, 1.0]},
            },
        }
    def collision(self) -> dict[str, Any]:
        return {
            "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            "geometry": {"sphere": {"radius": 0.005}},
        }


class THAXLJOINT:
    def __init__(self, joint_config: dict[str, Any]):
        self.joint_config = joint_config

    def joint_name(self) -> str:
        return "th_axl"

    def joint_type(self) -> str:
        return "revolute"

    def origin(self) -> dict[str, Any]:
        return {"xyz": [-0.0144, -0.0, 0.013], "rpy": [-1.5708, 0.0, 1.5708]}

    def parent_link_name(self) -> str:
        return "cmc"

    def child_link_name(self) -> str:
        return "axl"

    def axis(self) -> list[float]:
        return [0.0, 0.0, 1.0]

    def limit(self) -> dict[str, Any]:
        return {"effort": self.joint_config["effort"], "velocity": self.joint_config["velocity"], "lower": self.joint_config["lower"], "upper": self.joint_config["upper"]}


class THMCPJOINT:
    def __init__(self, joint_config: dict[str, Any]):
        self.joint_config = joint_config
    def joint_name(self) -> str:
        return "th_mcp"

    def joint_type(self) -> str:
        return "revolute"

    def origin(self) -> dict[str, Any]:
        return {"xyz": [0.0145, -0.0, 0.019], "rpy": [1.5708, 0.0, -1.5708]}

    def parent_link_name(self) -> str:
        return "axl"

    def child_link_name(self) -> str:
        return "th_mcp"

    def axis(self) -> list[float]:
        return [0.0, 0.0, 1.0]

    def limit(self) -> dict[str, Any]:
        return {"effort": self.joint_config["effort"], "velocity": self.joint_config["velocity"], "lower": self.joint_config["lower"], "upper": self.joint_config["upper"]}


class THIPLJOINT:
    def __init__(self, joint_config: dict[str, Any]):
        self.joint_config = joint_config
    def joint_name(self) -> str:
        return "th_ipl"

    def joint_type(self) -> str:
        return "revolute"

    def origin(self) -> dict[str, Any]:
        return {"xyz": [-0.0, 0.0446, 0.0], "rpy": [-0.0, -0.0, -3.14159]}

    def parent_link_name(self) -> str:
        return "th_mcp"

    def child_link_name(self) -> str:
        return "th_ipl"

    def axis(self) -> list[float]:
        return [0.0, 0.0, 1.0]

    def limit(self) -> dict[str, Any]:
        return {"effort": self.joint_config["effort"], "velocity": self.joint_config["velocity"], "lower": self.joint_config["lower"], "upper": self.joint_config["upper"]}


class THREALTIPJOINT:
    def joint_name(self) -> str:
        return "th_realtip"

    def joint_type(self) -> str:
        return "fixed"

    def origin(self) -> dict[str, Any]:
        return {"xyz": [0.0, -0.07, 0.015], "rpy": [0.0, 0.0, 0.0]}

    def parent_link_name(self) -> str:
        return "th_ipl"

    def child_link_name(self) -> str:
        return "th_realtip"

    def axis(self) -> list[float]:
        return [0.0, 0.0, -1.0]

    def limit(self) -> dict[str, Any]:
        return None

@dataclass(frozen=True)
class Thumb:
    links: list[Any]
    joints: list[Any]


def generate_thumb(teleop: bool = False) -> Thumb:
    joint_config = load_joint_config(JOINT_CONFIG_FILE)["leapXela"]["sim"]["thumb"]
    links = [CMCLink(), AXLLink(), MCPLink(), IPLLink()]

    joints = [ THAXLJOINT(joint_config["axl"]), THMCPJOINT(joint_config["mcp"]), THIPLJOINT(joint_config["ipl"])]
    if teleop:
        links += [THREALTIPLINK()]
        joints += [THREALTIPJOINT()]
    return Thumb(links=links, joints=joints)


def link_urdf(link: Any) -> str:
    inertial = link.inertial()
    visuals = _as_list(link.visual())
    collisions = _as_list(link.collision())

    visuals_xml = ""
    collisions_xml = ""
    for v in visuals:
        if "mesh" in v["geometry"].keys():
            mesh_visual_xml = f"""<mesh filename="{v["geometry"]["mesh"]["filename"]}"/>"""
        else:
            mesh_visual_xml = f"""<sphere radius="{v["geometry"]["sphere"]["radius"]}"/>"""

        visuals_xml += f"""\n
    <visual>
      <origin xyz="{list_to_string(v["origin"]["xyz"])}" rpy="{list_to_string(v["origin"]["rpy"])}"/>
      <geometry>
        {mesh_visual_xml}
      </geometry>
      <material name="{v["material"]["name"]}">
        <color rgba="{list_to_string(v["material"]["color"]["rgba"])}"/>
      </material>
    </visual>
    """
    for c in collisions:
        if "mesh" in c["geometry"].keys():
            mesh_collision_xml = f"""<mesh filename="{c["geometry"]["mesh"]["filename"]}"/>"""
        else:
            mesh_collision_xml = f"""<sphere radius="{c["geometry"]["sphere"]["radius"]}"/>"""
        collisions_xml += f"""\n
    <collision>
      <origin xyz="{list_to_string(c["origin"]["xyz"])}" rpy="{list_to_string(c["origin"]["rpy"])}"/>
      <geometry>
        {mesh_collision_xml}
      </geometry>
    </collision>
    """

    return f"""
  <link name="{link.link_name()}">
    <inertial>
      <origin xyz="{list_to_string(inertial["origin"]["xyz"])}" rpy="{list_to_string(inertial["origin"]["rpy"])}"/>
      <mass value="{inertial["mass"]}"/>
      <inertia ixx="{inertial["inertia"]["ixx"]}" ixy="{inertial["inertia"]["ixy"]}" ixz="{inertial["inertia"]["ixz"]}" iyy="{inertial["inertia"]["iyy"]}" iyz="{inertial["inertia"]["iyz"]}" izz="{inertial["inertia"]["izz"]}"/>
    </inertial>
{visuals_xml}
{collisions_xml}
  </link>
""".rstrip("\n")


def joint_urdf(joint: Any) -> str:
    limit = joint.limit()
    origin = joint.origin()

    limit_xml = ""
    if limit is not None:
        limit_xml = f"""<limit effort="{limit["effort"]}" velocity="{limit["velocity"]}" lower="{limit["lower"]}" upper="{limit["upper"]}"/>"""
    return f"""
  <joint name="{joint.joint_name()}" type="{joint.joint_type()}">
    <origin xyz="{list_to_string(origin["xyz"])}" rpy="{list_to_string(origin["rpy"])}"/>
    <parent link="{joint.parent_link_name()}"/>
    <child link="{joint.child_link_name()}"/>
    <axis xyz="{list_to_string(joint.axis())}"/>
    {limit_xml}
  </joint>
""".rstrip("\n")


def render_thumb_urdf(thumb: Thumb) -> str:
    links_xml = "\n".join(link_urdf(l) for l in thumb.links)
    joints_xml = "\n".join(joint_urdf(j) for j in thumb.joints)

    return f"""<?xml version="1.0" ?>
<robot name="xela_thumb_generated">
{links_xml}
{joints_xml}
<link name="base"/>
<joint name="base" type="fixed">
  <origin xyz="0 0 0.2" rpy="1.57 0 0"/>
  <parent link="base"/>
  <child link="{thumb.links[0].link_name()}"/>
</joint>
</robot>
"""


def write_thumb_urdf(file_path: str, thumb: Thumb) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(render_thumb_urdf(thumb))


if __name__ == "__main__":
    import os

    thumb = generate_thumb()
    out_path = os.environ.get("OUT") or "thumb.urdf"
    write_thumb_urdf(out_path, thumb)
    print(f"Wrote {out_path}")
