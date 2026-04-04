"""3D viewer widget — multi-tracker unified view, inspired by Three.js implementation."""

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal
from typing import Optional, Dict

from ...core.transforms import Transform3D


# ---------------------------------------------------------------------------
# Palette couleurs par tracker (miroir de TV3D_COLORS)
# ---------------------------------------------------------------------------

TRACKER_COLORS: Dict[str, tuple] = {
    "head":  (0.357, 0.490, 0.961, 1.0),   # #5b7df5 — bleu
    "left":  (0.133, 0.827, 0.525, 1.0),   # #22d386 — vert
    "right": (0.961, 0.773, 0.259, 1.0),   # #f5c542 — jaune
}
_DEFAULT_COLOR = (0.780, 0.800, 0.957, 1.0)  # #cdd6f4


def _tracker_color(name: str) -> tuple:
    return TRACKER_COLORS.get(name, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Gradient pour trajectoire (fade alpha de bas en haut)
# ---------------------------------------------------------------------------

def _gradient(n: int, c0: tuple, c1: tuple) -> np.ndarray:
    colors = np.ones((n, 4), dtype=np.float32)
    for i in range(3):
        colors[:, i] = np.linspace(c0[i], c1[i], n)
    colors[:, 3] = np.linspace(0.12, 0.95, n)
    return colors


_TRAJ_PALETTES = {
    "Couleur tracker":  None,          # utilise la couleur propre du tracker
    "Cyan → Blanc":   lambda n: _gradient(n, (0.2, 0.8, 1.0), (1.0, 1.0, 1.0)),
    "Violet → Rose":  lambda n: _gradient(n, (0.6, 0.2, 1.0), (1.0, 0.4, 0.7)),
    "Vert → Jaune":   lambda n: _gradient(n, (0.2, 1.0, 0.3), (1.0, 1.0, 0.2)),
    "Rouge → Orange": lambda n: _gradient(n, (0.9, 0.1, 0.1), (1.0, 0.7, 0.1)),
}


# ---------------------------------------------------------------------------
# Viewer3DWidget — unified multi-tracker viewer
# ---------------------------------------------------------------------------

class Viewer3DWidget(QWidget):
    """Unified 3D viewer displaying all trackers together.

    Mirrors the Three.js TV3D implementation:
    - Spherical orbital camera, auto-centred on data centroid
    - Per-tracker coloured trajectories (progressive drawRange)
    - Per-tracker sphere cursors

    Public API
    ----------
    build(trajectories)        -- dict name→np.ndarray(N,3), call once per session
    update_cursors(states, idx) -- dict name→Transform3D, call every frame
    apply_settings(cfg)        -- called by centralised param panel
    clear()                    -- reset all items
    """

    swap_requested = pyqtSignal(str)   # kept for compatibility

    _BTN_STYLE = (
        "QPushButton { background: #313244; color: #cdd6f4; border: none; "
        "border-radius: 2px; padding: 0 6px; font-size: 11px; }"
        "QPushButton:hover { background: #45475a; }"
    )

    # Axis remaps: CSV (x,y,z) → GL axes
    AXIS_REMAPS = {
        "X→X  Y→Z  Z→Y  (SteamVR→GL)": (0, 2, 1),
        "X  Y  Z  (identité)":          (0, 1, 2),
        "X  Z  Y":                       (0, 2, 1),
        "Y  Z  X":                       (1, 2, 0),
        "Z  X  Y":                       (2, 0, 1),
        "Y  X  Z":                       (1, 0, 2),
        "Z  Y  X":                       (2, 1, 0),
    }

    _DEF = dict(
        sphere_size=14,
        traj_width=2.0,
        palette="Couleur tracker",
        axis_len=0.12,
        axis_width=3.0,
        show_axes=True,
        show_grid=True,
        show_world_axis=True,
        cam_distance=2.0,
        axis_remap="X→X  Y→Z  Z→Y  (SteamVR→GL)",
    )

    def __init__(self, tracker_name: str = "", parent=None):
        super().__init__(parent)
        self.tracker_name = tracker_name   # kept for compat (single-tracker label)
        self._cfg = dict(self._DEF)

        # Per-tracker data
        self._raw_trajectories: Dict[str, np.ndarray] = {}   # original CSV coords
        self._traj_pts_gl: Dict[str, np.ndarray] = {}        # remapped GL coords
        self._traj_n: Dict[str, int] = {}                    # total point count
        self._traj_end: Dict[str, int] = {}                  # current draw end

        # Per-tracker GL items
        self._traj_items: Dict[str, gl.GLLinePlotItem] = {}
        self._sphere_items: Dict[str, gl.GLScatterPlotItem] = {}
        self._ax_x_items: Dict[str, gl.GLLinePlotItem] = {}
        self._ax_y_items: Dict[str, gl.GLLinePlotItem] = {}
        self._ax_z_items: Dict[str, gl.GLLinePlotItem] = {}

        # Scene helpers
        self._grid: Optional[gl.GLGridItem] = None
        self._world_axis: Optional[gl.GLAxisItem] = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar (kept minimal — tracker_name shows which slot this is)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(2, 1, 2, 1)
        title_row.setSpacing(4)

        self._title_label = QLabel("Vue 3D — Trackers")
        self._title_label.setStyleSheet(
            "color: #a6adc8; background: #1e1e2e; font-size: 10px; padding: 1px 4px;"
        )
        title_row.addWidget(self._title_label, stretch=1)

        # Legend dots (head / left / right)
        for role, color in TRACKER_COLORS.items():
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            dot = QLabel(f"● {role}")
            dot.setStyleSheet(
                f"color: {hex_color}; font-size: 9px; padding: 0 4px; background: #1e1e2e;"
            )
            title_row.addWidget(dot)

        self._swap_btn = QPushButton("⇄")
        self._swap_btn.setToolTip("Échanger ce tracker")
        self._swap_btn.setFixedHeight(18)
        self._swap_btn.setStyleSheet(self._BTN_STYLE)
        self._swap_btn.setVisible(False)
        self._swap_btn.clicked.connect(lambda: self.swap_requested.emit(self.tracker_name))
        title_row.addWidget(self._swap_btn)

        title_w = QWidget()
        title_w.setFixedHeight(20)
        title_w.setStyleSheet("background: #1e1e2e;")
        title_w.setLayout(title_row)
        root.addWidget(title_w)

        # GLViewWidget
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((13, 13, 42, 255))  # #0d0d2a — mirrors Three.js 0x13132a
        self.view.setCameraPosition(distance=self._cfg["cam_distance"],
                                    elevation=30, azimuth=45)
        root.addWidget(self.view, stretch=1)

        # World grid
        self._grid = gl.GLGridItem()
        self._grid.scale(0.1, 0.1, 0.1)
        self._grid.setColor((37, 37, 69, 255))  # #252545
        self.view.addItem(self._grid)

        # World axis helper
        self._world_axis = gl.GLAxisItem()
        self._world_axis.setSize(0.3, 0.3, 0.3)
        self.view.addItem(self._world_axis)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remap(self, pts: np.ndarray) -> np.ndarray:
        """Reorder CSV x/y/z columns to GL axes per current mapping."""
        i, j, k = self.AXIS_REMAPS.get(
            self._cfg["axis_remap"],
            self.AXIS_REMAPS["X→X  Y→Z  Z→Y  (SteamVR→GL)"]
        )
        return pts[..., [i, j, k]]

    def _remap1(self, v: np.ndarray) -> np.ndarray:
        return self._remap(v.reshape(1, 3))[0]

    def _traj_color_array(self, name: str, n: int) -> np.ndarray:
        """Return (n,4) color array for trajectory."""
        palette_fn = _TRAJ_PALETTES.get(self._cfg["palette"])
        if palette_fn is None:
            # "Couleur tracker" — solid tracker color with alpha fade
            c = _tracker_color(name)
            arr = np.ones((n, 4), dtype=np.float32)
            arr[:, 0] = c[0]; arr[:, 1] = c[1]; arr[:, 2] = c[2]
            arr[:, 3] = np.linspace(0.12, 0.95, n)
            return arr
        return palette_fn(n)

    def _get_or_create_tracker_items(self, name: str) -> None:
        """Ensure GL items exist for tracker `name`."""
        if name in self._traj_items:
            return
        c = _tracker_color(name)
        stub = np.zeros((2, 3), dtype=np.float32)

        traj = gl.GLLinePlotItem(
            pos=stub.copy(),
            color=c,
            width=self._cfg["traj_width"],
            antialias=True,
            mode='line_strip',
        )
        self.view.addItem(traj)
        self._traj_items[name] = traj

        sphere = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=c,
            size=self._cfg["sphere_size"],
            pxMode=True,
        )
        self.view.addItem(sphere)
        self._sphere_items[name] = sphere

        ax_x = gl.GLLinePlotItem(pos=stub.copy(), color=(1, 0.2, 0.2, 1),
                                  width=self._cfg["axis_width"], antialias=True)
        ax_y = gl.GLLinePlotItem(pos=stub.copy(), color=(0.2, 1, 0.2, 1),
                                  width=self._cfg["axis_width"], antialias=True)
        ax_z = gl.GLLinePlotItem(pos=stub.copy(), color=(0.3, 0.5, 1, 1),
                                  width=self._cfg["axis_width"], antialias=True)
        self.view.addItem(ax_x)
        self.view.addItem(ax_y)
        self.view.addItem(ax_z)
        self._ax_x_items[name] = ax_x
        self._ax_y_items[name] = ax_y
        self._ax_z_items[name] = ax_z

    def _remove_tracker_items(self, name: str) -> None:
        for d in (self._traj_items, self._sphere_items,
                  self._ax_x_items, self._ax_y_items, self._ax_z_items):
            item = d.pop(name, None)
            if item is not None:
                self.view.removeItem(item)

    def _update_axes_for(self, name: str, transform: Transform3D) -> None:
        L = self._cfg["axis_len"]
        w = self._cfg["axis_width"]
        o  = self._remap1(transform.position.astype(np.float32))
        tx = self._remap1(transform.transform_point(np.array([L, 0, 0])).astype(np.float32))
        ty = self._remap1(transform.transform_point(np.array([0, L, 0])).astype(np.float32))
        tz = self._remap1(transform.transform_point(np.array([0, 0, L])).astype(np.float32))
        self._ax_x_items[name].setData(pos=np.array([o, tx]), width=w)
        self._ax_y_items[name].setData(pos=np.array([o, ty]), width=w)
        self._ax_z_items[name].setData(pos=np.array([o, tz]), width=w)

    def _recenter_scene(self) -> None:
        """Move grid/axis to data centroid and adjust camera distance — mirrors tv3dBuild."""
        if not self._traj_pts_gl:
            return
        all_pts = np.concatenate(list(self._traj_pts_gl.values()), axis=0)
        cx, cy, cz = float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean()), float(all_pts[:, 2].mean())
        # Radius
        dists = np.sqrt(((all_pts - np.array([cx, cy, cz])) ** 2).sum(axis=1))
        R = max(float(dists.max()), 0.05)

        # Reposition grid under data
        if self._grid:
            self._grid.resetTransform()
            self._grid.scale(R * 0.4, R * 0.4, R * 0.4)
            self._grid.translate(cx, cy - R, cz)

        # Reposition world axis on centroid
        if self._world_axis:
            self._world_axis.resetTransform()
            self._world_axis.setSize(R * 0.5, R * 0.5, R * 0.5)
            self._world_axis.translate(cx, cy, cz)

        # Camera distance proportional to radius, looking at centroid
        dist = R * 3.5
        self.view.setCameraPosition(distance=dist, elevation=30, azimuth=45)
        # Pan to centroid
        self.view.opts['center'] = __import__('pyqtgraph', fromlist=['Vector']).Vector(cx, cy, cz)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, trajectories: Dict[str, np.ndarray]) -> None:
        """Load all tracker trajectories and rebuild the scene.

        Parameters
        ----------
        trajectories : dict  name → np.ndarray(N, 3) in CSV coords (x, y, z)
        """
        # Remove items for trackers that disappeared
        gone = set(self._traj_items) - set(trajectories)
        for name in gone:
            self._remove_tracker_items(name)
            self._raw_trajectories.pop(name, None)
            self._traj_pts_gl.pop(name, None)
            self._traj_n.pop(name, None)
            self._traj_end.pop(name, None)

        for name, pts in trajectories.items():
            if pts is None or len(pts) < 2:
                continue
            raw = pts.astype(np.float32)
            self._raw_trajectories[name] = raw
            gl_pts = self._remap(raw)
            self._traj_pts_gl[name] = gl_pts
            n = len(gl_pts)
            self._traj_n[name] = n
            self._traj_end[name] = 1

            self._get_or_create_tracker_items(name)
            colors = self._traj_color_array(name, n)
            self._traj_items[name].setData(
                pos=gl_pts[:1],
                color=colors[:1],
                width=self._cfg["traj_width"],
            )
            # Initialise sphere at first position
            self._sphere_items[name].setData(pos=gl_pts[0:1])

        self._recenter_scene()

    def update_cursors(self, states: Dict[str, Transform3D], traj_idx: int) -> None:
        """Move all tracker cursors to current position and reveal trajectory up to traj_idx."""
        for name, transform in states.items():
            if name not in self._sphere_items:
                continue
            pos_gl = self._remap1(transform.position.astype(np.float32))
            self._sphere_items[name].setData(pos=pos_gl.reshape(1, 3))

            if self._cfg["show_axes"] and name in self._ax_x_items:
                self._update_axes_for(name, transform)

            # Progressive trajectory reveal (mirrors drawRange update)
            if name in self._traj_pts_gl:
                n = self._traj_n[name]
                end = max(1, min(traj_idx + 1, n))
                self._traj_end[name] = end
                gl_pts = self._traj_pts_gl[name]
                colors = self._traj_color_array(name, n)
                self._traj_items[name].setData(
                    pos=gl_pts[:end],
                    color=colors[:end],
                )

    # ── Compat methods (single-tracker usage from _CameraColumn) ─────

    def set_tracker_name(self, name: str) -> None:
        self.tracker_name = name

    def set_trajectory(self, trajectory: np.ndarray) -> None:
        """Single-tracker compat shim — delegates to build()."""
        if self.tracker_name:
            self.build({self.tracker_name: trajectory})

    def set_current_index(self, idx: int) -> None:
        """Single-tracker compat shim — delegates to update_cursors() without moving sphere."""
        for name in list(self._traj_pts_gl):
            if name in self._traj_pts_gl:
                n = self._traj_n[name]
                end = max(1, min(idx + 1, n))
                self._traj_end[name] = end
                gl_pts = self._traj_pts_gl[name]
                colors = self._traj_color_array(name, n)
                self._traj_items[name].setData(pos=gl_pts[:end], color=colors[:end])

    def update_transform(self, transform: Transform3D) -> None:
        """Single-tracker compat shim."""
        if not self.tracker_name:
            return
        name = self.tracker_name
        if name not in self._sphere_items:
            self._get_or_create_tracker_items(name)
        pos_gl = self._remap1(transform.position.astype(np.float32))
        self._sphere_items[name].setData(pos=pos_gl.reshape(1, 3))
        if self._cfg["show_axes"] and name in self._ax_x_items:
            self._update_axes_for(name, transform)

    def apply_settings(self, cfg: dict) -> None:
        """Apply settings dict from centralised param panel."""
        self._cfg.update(cfg)
        c = self._cfg

        # Remap trajectories with potentially new axis mapping
        for name, raw in self._raw_trajectories.items():
            self._traj_pts_gl[name] = self._remap(raw)

        for name in list(self._traj_items):
            self._traj_items[name].setData(width=c["traj_width"])
        for name in list(self._sphere_items):
            self._sphere_items[name].setData(size=c["sphere_size"])
        for name in list(self._ax_x_items):
            show = c["show_axes"]
            self._ax_x_items[name].setVisible(show)
            self._ax_y_items[name].setVisible(show)
            self._ax_z_items[name].setVisible(show)

        if self._grid:
            self._grid.setVisible(c["show_grid"])
        if self._world_axis:
            self._world_axis.setVisible(c["show_world_axis"])

        # Restore trajectory render with new palette / mapping
        for name in list(self._traj_pts_gl):
            n = self._traj_n[name]
            end = self._traj_end[name]
            gl_pts = self._traj_pts_gl[name]
            colors = self._traj_color_array(name, n)
            self._traj_items[name].setData(pos=gl_pts[:end], color=colors[:end])

        self.view.setCameraPosition(distance=c["cam_distance"])

    def clear(self) -> None:
        for name in list(self._traj_items):
            self._remove_tracker_items(name)
        self._raw_trajectories.clear()
        self._traj_pts_gl.clear()
        self._traj_n.clear()
        self._traj_end.clear()
