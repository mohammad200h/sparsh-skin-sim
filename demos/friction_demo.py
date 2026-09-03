"""Slope + object. Friction [sliding, torsional, rolling]. Keys: 1/2  3/4  5/6  B/S/C."""

import time

import mujoco as mj
import mujoco.viewer

import _bootstrap  # noqa: F401

XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.05" rgba="0.3 0.3 0.35 1"/>
    <geom name="slope" type="box" size="1.5 0.4 0.03" pos="0 0 0.7" euler="0 -25 0"
          rgba="0.6 0.5 0.4 1" condim="6" friction="0.3 0.005 0.0001"/>
    <body name="obj" pos="1.05 0 1.32">
      <freejoint/>
      <geom name="obj" type="box" size="0.08 0.08 0.08" density="250"
            rgba="0.85 0.25 0.2 1" condim="6" friction="0.3 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""

# "box", "sphere", or "cylinder"
shape = "box"
# sliding, torsional, rolling
friction = [0.3, 0.005, 0.0001]

SHAPES = {
    "b": ("box", mj.mjtGeom.mjGEOM_BOX, [0.08, 0.08, 0.08]),
    "s": ("sphere", mj.mjtGeom.mjGEOM_SPHERE, [0.08, 0, 0]),
    "c": ("cylinder", mj.mjtGeom.mjGEOM_CYLINDER, [0.08, 0.08, 0]),
}


def reset():
    data.qpos[:3] = [1.05, 0, 1.32]
    data.qvel[:] = 0
    # cylinder lies on its side so it can roll down the slope
    if shape == "cylinder":
        data.qpos[3:7] = [0.7071, 0.7071, 0, 0]
    else:
        data.qpos[3:7] = [1, 0, 0, 0]


def set_shape(key):
    global shape
    shape, geom_type, size = SHAPES[key]
    model.geom_type[obj] = geom_type
    model.geom_size[obj] = size
    reset()
    print("shape =", shape)


def on_key(keycode):
    key = chr(keycode).lower() if 0 <= keycode < 256 else ""
    if key in SHAPES:
        set_shape(key)
        return
    if key == "1":
        friction[0] = max(0.0, friction[0] - 0.05)
    elif key == "2":
        friction[0] += 0.05
    elif key == "3":
        friction[1] *= 0.5
    elif key == "4":
        friction[1] *= 2.0
    elif key == "5":
        friction[2] *= 0.5
    elif key == "6":
        friction[2] *= 2.0
    elif key == "r":
        reset()
        return
    else:
        return
    print("friction [slide, torsion, roll] =", friction)


model = mj.MjModel.from_xml_string(XML)
data = mj.MjData(model)
obj = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "obj")
slope = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "slope")

set_shape({"box": "b", "sphere": "s", "cylinder": "c"}[shape])
print("B box   S sphere   C cylinder   R reset")
print("1/2 sliding   3/4 torsional   5/6 rolling")
with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    while viewer.is_running():
        model.geom_friction[obj] = friction
        model.geom_friction[slope] = friction
        mj.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
