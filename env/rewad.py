from __future__ import annotations

import sys
from collections.abc import Callable

import mujoco as mj
import numpy as np

from util.motion_util import GRASP_PATTERNS

FINGERTIP_FLEX_NAMES: dict[str, str] = {
    "if": "flex_if_tip",
    "mf": "flex_mf_tip",
    "rf": "flex_rf_tip",
    "th": "flex_th_tip",
}
FINGERTIP_BODY_NAMES: dict[str, str] = {
    "if": "if_ds",
    "mf": "mf_ds",
    "rf": "rf_ds",
    "th": "th_ds",
}


class SqueezeReward:
    def __init__(self, motion_type: str) -> None:
        if motion_type not in GRASP_PATTERNS:
            raise NotImplementedError(f"valid rewards are {GRASP_PATTERNS}")
        self._motion_type = motion_type

        self._reward_scale = {
            "squeeze": 0.1,
        }

    def get_reward_func(self):
        if self._motion_type == "squeeze":
            return self._reward_squeeze
        raise NotImplementedError(f"valid rewards are {GRASP_PATTERNS}")

    def _reward_squeeze(
        self,
        contact_func: Callable[[], tuple[dict[str, bool], np.ndarray]],
    ) -> float:
        _, contacts = contact_func()
        return float(contacts.sum() * self._reward_scale["squeeze"])


class SqueezeCyringeReward:
    """Fingertip–cyringe contacts plus plunger (``squeeze``) travel from range lo."""

    def __init__(self) -> None:
        self._current_handle_joint_value: float | None = None
        self._squeeze_jnt_id: int = -1
        self._squeeze_qposadr: int = -1
        self._fingertip_keys: tuple[str, ...] = ()
        self._flex_finger = np.zeros(0, dtype=np.int8)
        self._geom_finger = np.zeros(0, dtype=np.int8)
        self._object_geom_ids = np.zeros(0, dtype=np.int32)
        self._reward_scale = {
            "squeeze": 0.1,
            "displacement": 1e6
        }

    def bind(self, model: mj.MjModel, housing_name: str | None) -> None:
        """Cache squeeze-joint range and fingertip/cyringe contact ids."""
        self._cache_squeeze_joint(model, housing_name)
        self._cache_contact_ids(model, housing_name)

    def reset(self, model: mj.MjModel, data: mj.MjData) -> None:
        """Put the plunger at range lo and use that as the displacement baseline."""
        if self._squeeze_qposadr < 0:
            return
        lo = float(model.jnt_range[self._squeeze_jnt_id, 0])
        data.qpos[self._squeeze_qposadr] = lo
        self._current_handle_joint_value = lo

    def __call__(self, model: mj.MjModel, data: mj.MjData) -> float:
        del model
        contact_term = self._fingertips_contacts(data)
        disp_term = self._handle_displacement(data)

        contact = contact_term * self._reward_scale["squeeze"]
        disp = disp_term * self._reward_scale["displacement"]
        # Erase the whole line then rewrite so the status stays on one row.
        sys.stdout.write(
            f"\r\033[2Kreward -> contact_term: {contact:.4g}, disp_term: {disp:.6g}"
        )
        sys.stdout.flush()

        return (
            self._reward_scale["squeeze"] * contact_term
            + self._reward_scale["displacement"] * disp_term
        )

    def _cache_squeeze_joint(
        self, model: mj.MjModel, housing_name: str | None
    ) -> None:
        housing = housing_name or ""
        prefix = housing.rsplit("/", 1)[0] + "/" if "/" in housing else ""
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{prefix}squeeze")
        self._squeeze_jnt_id = int(jid)
        if jid < 0:
            self._squeeze_qposadr = -1
            self._current_handle_joint_value = None
            return
        self._squeeze_qposadr = int(model.jnt_qposadr[jid])
        self._current_handle_joint_value = float(model.jnt_range[jid, 0])

    def _geom_ids_for_body(self, model: mj.MjModel, body_name: str) -> list[int]:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return []
        adr = int(model.body_geomadr[body_id])
        n = int(model.body_geomnum[body_id])
        return list(range(adr, adr + n))

    def _cyringe_geom_ids(self, model: mj.MjModel, housing_name: str) -> np.ndarray:
        prefix = housing_name.rsplit("/", 1)[0] + "/" if "/" in housing_name else ""
        ids: list[int] = []
        for body_id in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
            if name is None:
                continue
            if name == housing_name or name.startswith(prefix):
                adr = int(model.body_geomadr[body_id])
                n = int(model.body_geomnum[body_id])
                ids.extend(range(adr, adr + n))
        return np.asarray(ids, dtype=np.int32)

    def _cache_contact_ids(
        self, model: mj.MjModel, housing_name: str | None
    ) -> None:
        self._fingertip_keys = tuple(FINGERTIP_FLEX_NAMES)
        finger_index = {name: i for i, name in enumerate(self._fingertip_keys)}
        self._flex_finger = np.full(model.nflex, -1, dtype=np.int8)
        self._geom_finger = np.full(model.ngeom, -1, dtype=np.int8)

        for finger, flex_name in FINGERTIP_FLEX_NAMES.items():
            flex_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, flex_name)
            if flex_id >= 0:
                self._flex_finger[flex_id] = finger_index[finger]

        for finger, body_name in FINGERTIP_BODY_NAMES.items():
            for geom_id in self._geom_ids_for_body(model, body_name):
                name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
                if name is not None and "tip" in name:
                    self._geom_finger[geom_id] = finger_index[finger]

        if housing_name is not None:
            self._object_geom_ids = self._cyringe_geom_ids(model, housing_name)
        else:
            self._object_geom_ids = np.asarray([], dtype=np.int32)

    def _get_contact_between_fingers_and_object(
        self, data: mj.MjData
    ) -> tuple[dict[str, bool], np.ndarray]:
        values = np.zeros(len(self._fingertip_keys), dtype=bool)
        ncon = int(data.ncon)
        if ncon == 0 or self._object_geom_ids.size == 0:
            return dict(zip(self._fingertip_keys, values.tolist())), values

        geoms = data.contact.geom[:ncon]
        flexes = data.contact.flex[:ncon]
        finger = np.where(
            flexes >= 0,
            self._flex_finger[np.clip(flexes, 0, self._flex_finger.size - 1)],
            np.where(
                geoms >= 0,
                self._geom_finger[np.clip(geoms, 0, self._geom_finger.size - 1)],
                -1,
            ),
        )
        hit = finger[(finger >= 0) & np.isin(geoms[:, ::-1], self._object_geom_ids)]
        if hit.size:
            values[hit] = True
        contacts = dict(zip(self._fingertip_keys, values.tolist()))
        return contacts, values

    def _fingertips_contacts(self, data: mj.MjData) -> float:
        _, contacts = self._get_contact_between_fingers_and_object(data)
        return float(np.asarray(contacts).sum())

    def _handle_displacement(self, data: mj.MjData) -> float:
        if self._squeeze_qposadr < 0 or self._current_handle_joint_value is None:
            return 0.0
        q = float(data.qpos[self._squeeze_qposadr])
    
        delta = q - self._current_handle_joint_value
        self._current_handle_joint_value = q
        noise_in_sensor_value = 2e-7
        return delta if delta > noise_in_sensor_value else 0.0
