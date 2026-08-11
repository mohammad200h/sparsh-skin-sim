"""Live matplotlib visualization for flex displacements and forces."""

from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np

from .flex_util import (
    all_flex_rest_vertex_positions,
    flex_joint_ids,
    flex_rest_vertex_positions,
    list_flex_names,
)

_CHANNEL_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "magnitude": None,
}

# Reference color-scale floors for adaptive (peak-hold) mode. Kept large enough
# that near-zero residual noise stays dark after contact ends.
_DISP_CLIM_REF = 5e-4   # 0.5 mm
_FORCE_CLIM_REF = 5e-2  # 50 mN
_CLIM_DECAY = 0.997     # per visual update; slow fall of the high-water mark


def _project_to_plane(points: np.ndarray) -> np.ndarray:
    """Project 3D points onto their best-fit 2D plane via PCA."""
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _channel_values(displacements: np.ndarray, channel: str) -> np.ndarray:
    if channel not in _CHANNEL_INDEX:
        raise ValueError(
            f"Unknown channel '{channel}'; choose from {sorted(_CHANNEL_INDEX)}"
        )
    index = _CHANNEL_INDEX[channel]
    if index is None:
        return np.linalg.norm(displacements, axis=1)
    if displacements.shape[1] <= index:
        raise ValueError(
            f"Channel '{channel}' needs joint axis {index}, but displacements "
            f"have shape {displacements.shape}"
        )
    return displacements[:, index]


def _adaptive_clim(peak: float, *, visualize_force: bool, vmax: float | None) -> float:
    """Fixed vmax if given; otherwise 1.25× peak with a reference floor.

    Prefer ``ClimTracker`` for live plots so the scale does not collapse when
    the signal returns to zero (which would paint residual noise as "max").
    """
    if vmax is not None:
        return float(vmax)
    ref = _FORCE_CLIM_REF if visualize_force else _DISP_CLIM_REF
    return max(float(peak) * 1.25, ref)


class ClimTracker:
    """Expand-on-rise, slow-decay color limit so zeros stay visually dark.

    Shrinking the scale every frame to ``1.25 * peak`` makes a near-zero flex
    look fully saturated from residual millinewton / sub-mm noise. This tracker
    keeps a high-water mark that only decays slowly toward a reference floor.
    """

    def __init__(
        self,
        *,
        visualize_force: bool,
        vmax: float | None = None,
        decay: float = _CLIM_DECAY,
    ) -> None:
        self.vmax = vmax
        self.decay = float(decay)
        self.ref = _FORCE_CLIM_REF if visualize_force else _DISP_CLIM_REF
        self.hold = self.ref

    def update(self, peak: float) -> float:
        if self.vmax is not None:
            return float(self.vmax)
        peak = float(peak)
        target = max(peak * 1.25, self.ref)
        if target > self.hold:
            self.hold = target
        else:
            toward = max(peak * 1.25, self.ref)
            self.hold = max(toward, self.hold * self.decay)
        return self.hold


def _short_flex_label(flex_name: str) -> str:
    return flex_name.removeprefix("flex_")


def _grid_shape(n: int) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("Need at least one flex to visualize")
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


class FlexLiveVisualizer:
    """Live matplotlib view of flex joint displacements or estimated forces.

    Left panel: rest-pose vertex layout colored by the selected channel.
    Right panel: rolling history of max |x|, |y|, |z|, and magnitude.
    """

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        flex_name: str,
        *,
        channel: str = "magnitude",
        history_len: int = 300,
        vmax: float | None = None,
        update_hz: float = 30.0,
        visualize_force: bool = False,
    ) -> None:
        self.flex_name = flex_name
        self.channel = channel
        self.history_len = int(history_len)
        self.visualize_force = bool(visualize_force)
        # None => peak-hold adaptive clim (see ClimTracker).
        self.vmax = vmax
        self._clim = ClimTracker(visualize_force=self.visualize_force, vmax=vmax)
        self._min_dt = 1.0 / float(update_hz) if update_hz > 0 else 0.0
        self._last_draw_time = -np.inf
        self._quantity = "force" if self.visualize_force else "displacement"
        self._unit = "N" if self.visualize_force else "m"

        rest = flex_rest_vertex_positions(model, data, flex_name)
        self._xy = _project_to_plane(rest)
        joint_ids = flex_joint_ids(model, flex_name)
        self._shape = joint_ids.shape
        self._qadr = np.asarray(
            model.jnt_qposadr[joint_ids.reshape(-1)], dtype=np.int32
        )

        self._t_hist: deque[float] = deque(maxlen=self.history_len)
        self._max_hist = {
            "x": deque(maxlen=self.history_len),
            "y": deque(maxlen=self.history_len),
            "z": deque(maxlen=self.history_len),
            "magnitude": deque(maxlen=self.history_len),
        }

        plt.ion()
        self.fig, (self.ax_map, self.ax_hist) = plt.subplots(
            1, 2, figsize=(10, 4), constrained_layout=True
        )
        try:
            self.fig.canvas.manager.set_window_title(
                f"flex: {flex_name} [{self._quantity}]"
            )
        except Exception:
            pass

        values0 = np.zeros(self._shape[0])
        clim = self._clim.update(0.0)
        self._scatter = self.ax_map.scatter(
            self._xy[:, 0],
            self._xy[:, 1],
            c=values0,
            s=80,
            cmap="inferno",
            vmin=-clim if channel != "magnitude" else 0.0,
            vmax=clim,
            edgecolors="k",
            linewidths=0.3,
        )
        self._cbar = self.fig.colorbar(self._scatter, ax=self.ax_map, fraction=0.046)
        self._cbar.set_label(self._unit)
        self.ax_map.set_aspect("equal")
        self.ax_map.set_title(f"{flex_name}  [{channel}]  ({self._quantity})")
        self.ax_map.set_xlabel("plane u (m)")
        self.ax_map.set_ylabel("plane v (m)")

        self._lines = {
            name: self.ax_hist.plot([], [], label=f"max |{name}|")[0]
            for name in ("x", "y", "z", "magnitude")
        }
        self.ax_hist.set_title(f"max |{self._quantity}|")
        self.ax_hist.set_xlabel("time (s)")
        self.ax_hist.set_ylabel(self._unit)
        self.ax_hist.legend(loc="upper right", fontsize=8)
        self.ax_hist.grid(True, alpha=0.3)

        self.fig.show()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    def set_channel(self, channel: str) -> None:
        _channel_values(np.zeros((1, 3)), channel)  # validate
        self.channel = channel
        self.ax_map.set_title(
            f"{self.flex_name}  [{channel}]  ({self._quantity})"
        )

    def _displacements(self, model: mj.MjModel, data: mj.MjData) -> np.ndarray:
        return np.asarray(
            data.qpos[self._qadr] - model.qpos0[self._qadr], dtype=np.float64
        ).reshape(self._shape)

    def update(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        *,
        force_redraw: bool = False,
        force_field: np.ndarray | None = None,
    ) -> None:
        """Refresh the plot from the current MuJoCo state.

        When ``visualize_force`` is True, pass ``force_field`` with shape
        ``(n_vert, 3)`` from ``FlexForceEstimator.update``.
        """
        if not force_redraw and (data.time - self._last_draw_time) < self._min_dt:
            return
        if not plt.fignum_exists(self.fig.number):
            return

        if self.visualize_force:
            if force_field is None:
                raise ValueError(
                    "visualize_force=True requires force_field=(n_vert, 3)"
                )
            field = np.asarray(force_field, dtype=np.float64)
            if field.shape != self._shape:
                raise ValueError(
                    f"force_field shape {field.shape} != expected {self._shape}"
                )
        else:
            field = self._displacements(model, data)

        values = _channel_values(field, self.channel)
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        clim = self._clim.update(peak)

        self._scatter.set_array(values)
        if self.channel == "magnitude":
            self._scatter.set_clim(0.0, clim)
        else:
            self._scatter.set_clim(-clim, clim)
        self._cbar.update_normal(self._scatter)

        self._t_hist.append(float(data.time))
        self._max_hist["x"].append(float(np.max(np.abs(field[:, 0]))))
        self._max_hist["y"].append(float(np.max(np.abs(field[:, 1]))))
        self._max_hist["z"].append(float(np.max(np.abs(field[:, 2]))))
        self._max_hist["magnitude"].append(
            float(np.max(np.linalg.norm(field, axis=1)))
        )

        t = np.asarray(self._t_hist)
        for name, line in self._lines.items():
            line.set_data(t, np.asarray(self._max_hist[name]))
        self.ax_hist.relim()
        self.ax_hist.autoscale_view()

        if self.visualize_force:
            peak_txt = f"peak={1e3 * peak:.2f} mN"
        else:
            peak_txt = f"peak={1e3 * peak:.2f} mm"
        self.ax_map.set_title(
            f"{self.flex_name}  [{self.channel}]  ({self._quantity})  {peak_txt}"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        self._last_draw_time = float(data.time)

    def close(self) -> None:
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)


def visualize_flex_live(
    model: mj.MjModel,
    data: mj.MjData,
    flex_name: str,
    **kwargs,
) -> FlexLiveVisualizer:
    """Create a live matplotlib visualizer for a named flex."""
    return FlexLiveVisualizer(model, data, flex_name, **kwargs)


class AllFlexLiveVisualizer:
    """Live matplotlib grid of flex displacements or estimated forces.

    Each subplot shows that flex's rest-pose vertex layout colored by the
    selected channel. A shared color scale keeps patches comparable, and a
    bottom strip tracks the global max over time.
    """

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        flex_names: list[str] | None = None,
        *,
        channel: str = "magnitude",
        history_len: int = 300,
        vmax: float | None = None,
        update_hz: float = 20.0,
        visualize_force: bool = False,
    ) -> None:
        self.flex_names = (
            list(flex_names) if flex_names is not None else list_flex_names(model)
        )
        if not self.flex_names:
            raise ValueError("Model has no flexes to visualize")

        _channel_values(np.zeros((1, 3)), channel)  # validate
        self.channel = channel
        self.history_len = int(history_len)
        self.visualize_force = bool(visualize_force)
        # None => peak-hold adaptive clim (see ClimTracker).
        self.vmax = vmax
        self._clim = ClimTracker(visualize_force=self.visualize_force, vmax=vmax)
        self._min_dt = 1.0 / float(update_hz) if update_hz > 0 else 0.0
        self._last_draw_time = -np.inf
        self._quantity = "force" if self.visualize_force else "displacement"
        self._unit = "N" if self.visualize_force else "m"

        rest_by_name = all_flex_rest_vertex_positions(model, data, self.flex_names)
        self._panels: list[dict] = []
        for name in self.flex_names:
            joint_ids = flex_joint_ids(model, name)
            self._panels.append(
                {
                    "name": name,
                    "xy": _project_to_plane(rest_by_name[name]),
                    "shape": joint_ids.shape,
                    "qadr": np.asarray(
                        model.jnt_qposadr[joint_ids.reshape(-1)], dtype=np.int32
                    ),
                }
            )

        self._t_hist: deque[float] = deque(maxlen=self.history_len)
        self._max_hist = {
            "x": deque(maxlen=self.history_len),
            "y": deque(maxlen=self.history_len),
            "z": deque(maxlen=self.history_len),
            "magnitude": deque(maxlen=self.history_len),
        }

        rows, cols = _grid_shape(len(self._panels))
        plt.ion()
        self.fig = plt.figure(figsize=(3.2 * cols, 2.6 * rows + 1.8), layout="constrained")
        try:
            self.fig.canvas.manager.set_window_title(
                f"all flexes ({len(self._panels)}) [{self._quantity}/{channel}]"
            )
        except Exception:
            pass

        grid = self.fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [0.55])
        clim = self._clim.update(0.0)
        vmin = 0.0 if channel == "magnitude" else -clim
        self._scatters = []
        self._axes = []
        for index, panel in enumerate(self._panels):
            r, c = divmod(index, cols)
            ax = self.fig.add_subplot(grid[r, c])
            scatter = ax.scatter(
                panel["xy"][:, 0],
                panel["xy"][:, 1],
                c=np.zeros(panel["shape"][0]),
                s=55,
                cmap="inferno",
                vmin=vmin,
                vmax=clim,
                edgecolors="k",
                linewidths=0.2,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(_short_flex_label(panel["name"]), fontsize=8)
            self._axes.append(ax)
            self._scatters.append(scatter)

        for index in range(len(self._panels), rows * cols):
            r, c = divmod(index, cols)
            ax = self.fig.add_subplot(grid[r, c])
            ax.set_axis_off()

        self._cbar = self.fig.colorbar(
            self._scatters[0],
            ax=self._axes,
            fraction=0.02,
            pad=0.02,
        )
        self._cbar.set_label(self._unit)

        self.ax_hist = self.fig.add_subplot(grid[-1, :])
        self._lines = {
            name: self.ax_hist.plot([], [], label=f"max |{name}|")[0]
            for name in ("x", "y", "z", "magnitude")
        }
        self.ax_hist.set_title(
            f"global max |{self._quantity}| across all flexes", fontsize=9
        )
        self.ax_hist.set_xlabel("time (s)")
        self.ax_hist.set_ylabel(self._unit)
        self.ax_hist.legend(loc="upper right", fontsize=8, ncol=4)
        self.ax_hist.grid(True, alpha=0.3)
        self._status = self.fig.suptitle("")

        self.fig.show()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def set_channel(self, channel: str) -> None:
        _channel_values(np.zeros((1, 3)), channel)
        self.channel = channel
        self._cbar.set_label(self._unit)
        try:
            self.fig.canvas.manager.set_window_title(
                f"all flexes ({len(self._panels)}) [{self._quantity}/{channel}]"
            )
        except Exception:
            pass

    def update(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        *,
        force_redraw: bool = False,
        forces: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Refresh every flex subplot from the current MuJoCo state.

        When ``visualize_force`` is True, pass ``forces`` as
        ``{flex_name: (n_vert, 3)}`` from ``AllFlexForceEstimator.update``.
        """
        if not force_redraw and (data.time - self._last_draw_time) < self._min_dt:
            return
        if not plt.fignum_exists(self.fig.number):
            return

        if self.visualize_force and forces is None:
            raise ValueError(
                "visualize_force=True requires forces={flex_name: (n_vert, 3)}"
            )

        global_peak = 0.0
        global_abs = {name: 0.0 for name in ("x", "y", "z", "magnitude")}

        for panel, scatter in zip(self._panels, self._scatters):
            if self.visualize_force:
                assert forces is not None
                field = np.asarray(forces[panel["name"]], dtype=np.float64)
                if field.shape != panel["shape"]:
                    raise ValueError(
                        f"forces['{panel['name']}'] shape {field.shape} != "
                        f"expected {panel['shape']}"
                    )
            else:
                field = np.asarray(
                    data.qpos[panel["qadr"]] - model.qpos0[panel["qadr"]],
                    dtype=np.float64,
                ).reshape(panel["shape"])

            values = _channel_values(field, self.channel)
            scatter.set_array(values)
            if values.size:
                global_peak = max(global_peak, float(np.max(np.abs(values))))
            global_abs["x"] = max(global_abs["x"], float(np.max(np.abs(field[:, 0]))))
            global_abs["y"] = max(global_abs["y"], float(np.max(np.abs(field[:, 1]))))
            global_abs["z"] = max(global_abs["z"], float(np.max(np.abs(field[:, 2]))))
            global_abs["magnitude"] = max(
                global_abs["magnitude"],
                float(np.max(np.linalg.norm(field, axis=1))),
            )

        clim = self._clim.update(global_peak)
        if self.channel == "magnitude":
            vmin, vmax = 0.0, clim
        else:
            vmin, vmax = -clim, clim
        for scatter in self._scatters:
            scatter.set_clim(vmin, vmax)
        self._cbar.update_normal(self._scatters[0])

        self._t_hist.append(float(data.time))
        for name, value in global_abs.items():
            self._max_hist[name].append(value)

        t = np.asarray(self._t_hist)
        for name, line in self._lines.items():
            line.set_data(t, np.asarray(self._max_hist[name]))
        self.ax_hist.relim()
        self.ax_hist.autoscale_view()

        if self.visualize_force:
            status = (
                f"[{self.channel}/force]  "
                f"global peak = {1e3 * global_abs['magnitude']:.2f} mN  "
                f"|  clim = {1e3 * clim:.2f} mN  "
                f"|  ncon = {data.ncon}"
            )
        else:
            status = (
                f"[{self.channel}/displacement]  "
                f"global peak = {1e3 * global_abs['magnitude']:.2f} mm  "
                f"|  clim = {1e3 * clim:.2f} mm  "
                f"|  ncon = {data.ncon}"
            )
        self._status.set_text(status)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        self._last_draw_time = float(data.time)

    def close(self) -> None:
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)


def visualize_all_flexes_live(
    model: mj.MjModel,
    data: mj.MjData,
    flex_names: list[str] | None = None,
    **kwargs,
) -> AllFlexLiveVisualizer:
    """Create a live matplotlib grid visualizer for all (or selected) flexes."""
    return AllFlexLiveVisualizer(model, data, flex_names, **kwargs)
