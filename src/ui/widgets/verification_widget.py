"""Verification mode widget — clock-driven multi-camera player.

Architecture
============
A single QTimer (the "master clock") drives playback.  At each tick it
increments a shared frame counter and asks every camera thread to decode
that exact frame index.  There is no sequential read() — every frame
displayed is the result of an explicit seek, which eliminates inter-camera
drift completely.

Temporal alignment
==================
Each camera has an optional array of absolute capture timestamps (ns) loaded
from its .jsonl file.  When seeking by absolute time (timeline scrub) the
widget converts t_ns → per-camera frame index via np.searchsorted on those
arrays.  If a camera has no jsonl the master index is used directly.

Thread model
============
_VideoDecodeThread owns one cv2.VideoCapture.  It accepts seek commands via
a thread-safe queue, decodes the frame, rotates it 180°, and emits
frame_ready(position, frame_idx, ndarray) back to the main thread via a
Qt queued connection.  Only the most recent pending seek is kept — stale
seeks are dropped automatically.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QScrollArea, QSlider, QCheckBox, QSplitter, QVBoxLayout, QWidget,
)

from .viewer_3d_widget import Viewer3DWidget, _TRAJ_PALETTES
from ...core.transforms import Transform3D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_POSITIONS = ["left", "head", "right"]

# Palette canonique : left=vert, right=jaune, head=bleu
_CAM_COLORS: dict[str, str] = {
    "left":  "#22d386",
    "head":  "#5b7df5",
    "right": "#f5c542",
}
_TRACKER_COLORS: dict[str, QColor] = {k: QColor(v) for k, v in _CAM_COLORS.items()}
_DEFAULT_TRACKER_COLOR = QColor("#cdd6f4")

_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
_DEFAULT_SPEED_IDX = 2  # 1.0×

_COMBO_STYLE = """
    QComboBox {
        background: #313244; color: #cdd6f4;
        border: 1px solid #45475a; border-radius: 3px;
        padding: 1px 6px; font-size: 11px; font-weight: bold;
    }
    QComboBox:hover { border: 1px solid #89b4fa; }
    QComboBox::drop-down { border: none; width: 16px; }
    QComboBox QAbstractItemView {
        background: #313244; color: #cdd6f4;
        selection-background-color: #45475a;
        border: 1px solid #45475a;
    }
"""
_BTN = (
    "QPushButton { background: #313244; color: #cdd6f4; border: none; "
    "border-radius: 4px; padding: 4px 10px; font-size: 13px; }"
    "QPushButton:hover { background: #45475a; }"
    "QPushButton:pressed { background: #585b70; }"
    "QPushButton:disabled { background: #1e1e2e; color: #45475a; }"
    "QPushButton:checked { background: #89b4fa; color: #1e1e2e; }"
)
_BTN_OK = (
    "QPushButton { background: #40a02b; color: white; border: none; "
    "border-radius: 6px; padding: 8px 28px; font-size: 13px; font-weight: bold; }"
    "QPushButton:hover { background: #4cb832; }"
    "QPushButton:pressed { background: #389926; }"
    "QPushButton:disabled { background: #313244; color: #585b70; }"
)
_BTN_KO = (
    "QPushButton { background: #e64553; color: white; border: none; "
    "border-radius: 6px; padding: 8px 28px; font-size: 13px; font-weight: bold; }"
    "QPushButton:hover { background: #f04a5a; }"
    "QPushButton:pressed { background: #c73545; }"
    "QPushButton:disabled { background: #313244; color: #585b70; }"
)


# ---------------------------------------------------------------------------
# _VideoDecodeThread
# ---------------------------------------------------------------------------

class _VideoDecodeThread(QThread):
    """Owns one cv2.VideoCapture; decodes frames on demand via a seek queue.

    Only the most recently posted seek is kept — older ones are silently
    dropped so the UI always shows the latest requested position even under
    rapid scrubbing.
    """

    frame_ready = pyqtSignal(str, int, object)  # (position, frame_idx, ndarray)

    def __init__(self, position: str, video_path: str,
                 frame_count: int, parent=None):
        super().__init__(parent)
        self._position   = position
        self._video_path = video_path
        self._frame_count = max(1, frame_count)

        self._cond    = threading.Condition()
        self._pending: Optional[int] = None   # next frame to decode
        self._running = True
        self._current_idx = -1

    # ------------------------------------------------------------------
    # Public API  (main thread)
    # ------------------------------------------------------------------

    def seek(self, frame_idx: int) -> None:
        """Request decoding of frame_idx.  Replaces any pending request."""
        idx = max(0, min(int(frame_idx), self._frame_count - 1))
        with self._cond:
            self._pending = idx
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._pending = None
            self._cond.notify()

    # ------------------------------------------------------------------
    # Thread loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            return

        try:
            while True:
                with self._cond:
                    while self._pending is None and self._running:
                        self._cond.wait(timeout=1.0)
                    if not self._running:
                        break
                    target = self._pending
                    self._pending = None

                if target is None:
                    continue

                # Skip seek if we're already there (sequential playback optimisation)
                if target != self._current_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)

                ok, frame = cap.read()
                if not ok or frame is None:
                    # One retry after explicit re-seek
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    ok, frame = cap.read()

                if ok and frame is not None:
                    self._current_idx = target
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    self.frame_ready.emit(self._position, target, frame)
        finally:
            cap.release()

    @property
    def current_idx(self) -> int:
        return self._current_idx


# ---------------------------------------------------------------------------
# _VideoCell
# ---------------------------------------------------------------------------

class _VideoCell(QLabel):
    """Displays a single video frame with aspect ratio preservation."""

    _RATIO_W, _RATIO_H = 1920, 1200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #11111b;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(100)
        self._src: Optional[QPixmap] = None
        self._frame_idx   = 0
        self._frame_count = 0

    def hasHeightForWidth(self) -> bool:
        return False

    def set_frame(self, frame: np.ndarray, idx: int) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._src = QPixmap.fromImage(img)
        self._frame_idx = idx
        self._render()

    def set_frame_count(self, n: int) -> None:
        self._frame_count = n

    def clear_frame(self) -> None:
        self._src = None
        super().clear()

    def _render(self) -> None:
        if self._src is None:
            return
        scaled = self._src.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Overlay: frame number bottom-left
        out = QPixmap(scaled)
        p = QPainter(out)
        p.setFont(QFont("Monospace", 9))
        txt = (f"{self._frame_idx} / {self._frame_count - 1}"
               if self._frame_count > 0 else str(self._frame_idx))
        p.setPen(QColor(0, 0, 0, 180))
        p.drawText(6, out.height() - 6, txt)
        p.setPen(QColor("#cdd6f4"))
        p.drawText(5, out.height() - 7, txt)
        p.end()
        super().setPixmap(out)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()


# ---------------------------------------------------------------------------
# _TrackerCell
# ---------------------------------------------------------------------------

class _TrackerCell(QWidget):
    """Top-down XZ trajectory view for one tracker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name      = ""
        self._positions: Optional[np.ndarray] = None
        self._cur_idx   = 0
        self.setStyleSheet("background: #11111b;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(60)

    def hasHeightForWidth(self) -> bool:
        return False

    def set_source(self, name: str, positions: Optional[np.ndarray]) -> None:
        self._name = name
        if positions is not None and len(positions) >= 2:
            mask = np.isfinite(positions[:, 0]) & np.isfinite(positions[:, 2])
            positions = positions[mask] if mask.sum() >= 2 else None
        self._positions = positions
        self._cur_idx = 0
        self.update()

    def set_current_index(self, idx: int) -> None:
        if idx != self._cur_idx:
            self._cur_idx = idx
            self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#11111b"))

        if self._positions is None or len(self._positions) < 2:
            p.setPen(QColor("#585b70"))
            p.setFont(QFont("", 10))
            p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter,
                       f"Tracker {self._name}\n(aucune donnée)" if self._name
                       else "(aucune donnée)")
            return

        xs, zs = self._positions[:, 0], self._positions[:, 2]
        mg = 0.08
        xr = (xs.max() - xs.min()) or 1.0
        zr = (zs.max() - zs.min()) or 1.0
        xm, zm = xs.min(), zs.min()

        def px(xi, zi):
            return (
                int(mg * w + (xi - xm) / xr * w * (1 - 2 * mg)),
                int(h - (mg * h + (zi - zm) / zr * h * (1 - 2 * mg))),
            )

        color = _TRACKER_COLORS.get(self._name, _DEFAULT_TRACKER_COLOR)
        dim   = QColor(color.red(), color.green(), color.blue(), 55)
        n     = len(xs)
        cur   = min(max(0, self._cur_idx), n - 1)

        # Full path (dim)
        p.setPen(QPen(dim, 1))
        for i in range(1, n):
            if (np.isfinite(xs[i-1]) and np.isfinite(zs[i-1])
                    and np.isfinite(xs[i]) and np.isfinite(zs[i])):
                p.drawLine(*px(xs[i-1], zs[i-1]), *px(xs[i], zs[i]))

        # Elapsed path (bright)
        p.setPen(QPen(color, 2))
        for i in range(1, cur + 1):
            if (np.isfinite(xs[i-1]) and np.isfinite(zs[i-1])
                    and np.isfinite(xs[i]) and np.isfinite(zs[i])):
                p.drawLine(*px(xs[i-1], zs[i-1]), *px(xs[i], zs[i]))

        # Current position dot
        if np.isfinite(xs[cur]) and np.isfinite(zs[cur]):
            cx, cy = px(xs[cur], zs[cur])
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(cx - 5, cy - 5, 10, 10)

        p.setPen(QColor("#a6adc8"))
        p.setFont(QFont("", 9))
        p.drawText(6, 14, self._name)


# ---------------------------------------------------------------------------
# _TimelineBar
# ---------------------------------------------------------------------------

class _TimelineBar(QWidget):
    """Scrub bar operating in absolute nanoseconds.

    The cursor position is set programmatically during playback; it is also
    drag/click-editable by the user.  Emits cursor_moved(t_ns) only on user
    interaction (not on programmatic set_cursor calls) so there is no
    feedback loop.
    """

    cursor_moved = pyqtSignal(float)  # user interaction only

    _H      = 44
    _TRACK_H = 12
    _MARGIN  = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._start_ns  = 0.0
        self._end_ns    = 1.0
        self._cursor_ns = 0.0
        self._cam_ns: dict[str, float] = {}
        self._dragging  = False

        self.setFixedHeight(self._H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("background: #181825;")
        self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_range(self, start_ns: float, end_ns: float) -> None:
        self._start_ns  = float(start_ns)
        self._end_ns    = float(end_ns) if end_ns > start_ns else float(start_ns) + 1.0
        self._cursor_ns = self._start_ns
        self._cam_ns.clear()
        self.update()

    def set_cursor(self, t_ns: float) -> None:
        """Programmatic cursor update — does NOT emit cursor_moved."""
        clamped = max(self._start_ns, min(float(t_ns), self._end_ns))
        if clamped != self._cursor_ns:
            self._cursor_ns = clamped
            self.update()

    def set_camera_ns(self, position: str, t_ns: float) -> None:
        self._cam_ns[position] = t_ns
        self.update()

    @property
    def cursor_ns(self) -> float:
        return self._cursor_ns

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#181825"))

        m   = self._MARGIN
        tw  = w - 2 * m
        ty  = (h - self._TRACK_H) // 2
        dur = self._end_ns - self._start_ns
        if dur <= 0 or tw <= 0:
            return

        def x_of(t: float) -> int:
            return int(m + (t - self._start_ns) / dur * tw)

        # Track background
        p.fillRect(m, ty, tw, self._TRACK_H, QColor("#313244"))

        # Elapsed fill
        cx = x_of(self._cursor_ns)
        fill_w = max(0, cx - m)
        if fill_w > 0:
            p.fillRect(m, ty, fill_w, self._TRACK_H, QColor("#585b70"))

        # Per-camera tick marks
        for pos, t_ns in self._cam_ns.items():
            x = x_of(t_ns)
            p.setPen(QPen(QColor(_CAM_COLORS.get(pos, "#cdd6f4")), 2))
            p.drawLine(x, ty - 4, x, ty + self._TRACK_H + 4)

        # Second tick marks + labels
        p.setFont(QFont("", 8))
        step = 1_000_000_000  # 1 s
        t = (int(self._start_ns / step) + 1) * step
        while t < self._end_ns:
            x = x_of(t)
            p.setPen(QPen(QColor("#45475a"), 1))
            p.drawLine(x, ty, x, ty + self._TRACK_H)
            p.setPen(QColor("#585b70"))
            p.drawText(x + 2, ty - 1, f"{(t - self._start_ns) / 1e9:.0f}s")
            t += step

        # Cursor line (always on top)
        p.setPen(QPen(QColor("#cdd6f4"), 2))
        p.drawLine(cx, 2, cx, h - 2)

        # Time label next to cursor
        elapsed = (self._cursor_ns - self._start_ns) / 1e9
        total   = dur / 1e9
        label   = f"{elapsed:.3f}s / {total:.2f}s"
        p.setFont(QFont("", 9))
        p.setPen(QColor("#cdd6f4"))
        lw = p.fontMetrics().horizontalAdvance(label) + 6
        lx = cx + 4 if cx + 4 + lw < w else cx - lw - 4
        p.drawText(lx, h - 5, label)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _ns_at(self, x: float) -> float:
        tw = self.width() - 2 * self._MARGIN
        if tw <= 0:
            return self._start_ns
        frac = max(0.0, min(1.0, (x - self._MARGIN) / tw))
        return self._start_ns + frac * (self._end_ns - self._start_ns)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging  = True
            t = self._ns_at(event.position().x())
            self._cursor_ns = t
            self.update()
            self.cursor_moved.emit(t)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            t = self._ns_at(event.position().x())
            self._cursor_ns = t
            self.update()
            self.cursor_moved.emit(t)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False


# ---------------------------------------------------------------------------
# GripperTimelineWidget — rendu QPainter identique à l'annotation timeline
# ---------------------------------------------------------------------------

_GRIPPER_TRACK_H = 36
_GRIPPER_COLORS_VER: dict[str, QColor] = {
    "left":  QColor("#22d386"),   # vert
    "right": QColor("#f5c542"),   # jaune
}


class GripperTimelineWidget(QWidget):
    """Affiche les signaux gripper sous forme de waveform QPainter,
    alignés sur la timeline vidéo en nanosecondes absolues.

    Chaque track occupe _GRIPPER_TRACK_H pixels en hauteur.
    Le curseur suit set_cursor_ns().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._start_ns      = 0.0
        self._end_ns        = 1.0
        self._cursor_ns     = 0.0
        self._gripper_ref_ns = 0.0   # origine des timestamps gripper (ns Unix)
        self._gripper_data: dict = {}  # gid -> (timestamps_s, angles)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._update_height()

    # ------------------------------------------------------------------
    def _update_height(self) -> None:
        n = len(self._gripper_data)
        self.setFixedHeight(max(n * _GRIPPER_TRACK_H, 0))

    def set_range(self, start_ns: float, end_ns: float) -> None:
        self._start_ns = float(start_ns)
        self._end_ns   = float(end_ns) if end_ns > start_ns else float(start_ns) + 1.0
        self.update()

    def set_gripper_ref_ns(self, ref_ns: float) -> None:
        self._gripper_ref_ns = float(ref_ns)
        self.update()

    def set_data(self, gripper_data: dict) -> None:
        """gripper_data: gid -> (timestamps_s np.ndarray, angles np.ndarray)"""
        self._gripper_data = gripper_data or {}
        self._update_height()
        self.update()

    def set_cursor_ns(self, t_ns: float) -> None:
        self._cursor_ns = float(t_ns)
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        if not self._gripper_data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w   = self.width()
        dur = self._end_ns - self._start_ns
        if dur <= 0 or w <= 0:
            return

        margin = 10

        def ns_to_x(t_ns: float) -> int:
            return int(margin + (t_ns - self._start_ns) / dur * (w - 2 * margin))

        def ts_to_x(t_s: float) -> int:
            """Convertit un timestamp gripper (secondes depuis ref) en pixel."""
            return ns_to_x(self._gripper_ref_ns + t_s * 1e9)

        gripper_y = 0
        for gid, (timestamps, angles) in self._gripper_data.items():
            color = _GRIPPER_COLORS_VER.get(gid, QColor(200, 200, 200))

            # Background
            bg = QColor(color)
            bg.setAlpha(18)
            p.fillRect(0, gripper_y, w, _GRIPPER_TRACK_H, bg)

            # Separator
            p.setPen(QPen(QColor("#313244"), 1))
            p.drawLine(0, gripper_y, w, gripper_y)

            # Label
            label_color = QColor(color)
            label_color.setAlpha(180)
            p.setPen(label_color)
            p.setFont(QFont("Menlo", 8))
            side_label = "Left" if gid in ("left", "1") else "Right"
            p.drawText(margin + 2, gripper_y + 1, w - margin - 4, _GRIPPER_TRACK_H - 2,
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       side_label)

            # Waveform
            if timestamps is not None and angles is not None and len(timestamps) > 1:
                valid = np.isfinite(angles)
                if valid.any():
                    a_min = float(np.nanmin(angles))
                    a_max = float(np.nanmax(angles))
                    a_range = a_max - a_min if a_max > a_min else 1.0

                    inner_h   = _GRIPPER_TRACK_H - 6
                    inner_top = gripper_y + 3

                    # Filled path (fill entre la courbe et le bas)
                    fill_path = QPainterPath()
                    line_path = QPainterPath()
                    first = True
                    last_px = 0
                    for ts_val, angle_val in zip(timestamps, angles):
                        if not np.isfinite(angle_val):
                            first = True
                            continue
                        px_x = ts_to_x(float(ts_val))
                        norm  = (float(angle_val) - a_min) / a_range
                        px_y  = inner_top + int((1.0 - norm) * inner_h)
                        if first:
                            fill_path.moveTo(px_x, inner_top + inner_h)
                            fill_path.lineTo(px_x, px_y)
                            line_path.moveTo(px_x, px_y)
                            first = False
                        else:
                            fill_path.lineTo(px_x, px_y)
                            line_path.lineTo(px_x, px_y)
                        last_px = px_x
                    if not first:
                        fill_path.lineTo(last_px, inner_top + inner_h)
                        fill_path.closeSubpath()

                    fill_color = QColor(color)
                    fill_color.setAlpha(40)
                    p.setBrush(fill_color)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.drawPath(fill_path)

                    line_color = QColor(color)
                    line_color.setAlpha(220)
                    p.setPen(QPen(line_color, 1.5))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPath(line_path)

                    # Valeur courante : point + label
                    t_s_cursor = (self._cursor_ns - self._gripper_ref_ns) / 1e9
                    val = float(np.interp(t_s_cursor,
                                         timestamps.astype(float),
                                         angles.astype(float)))
                    if np.isfinite(val):
                        norm_val = (val - a_min) / a_range
                        dot_x = ns_to_x(self._cursor_ns)
                        dot_y = inner_top + int((1.0 - norm_val) * inner_h)
                        dot_color = QColor(color)
                        dot_color.setAlpha(255)
                        p.setPen(QPen(QColor("#f38ba8"), 1))
                        p.setBrush(dot_color)
                        p.drawEllipse(dot_x - 3, dot_y - 3, 6, 6)

                        val_color = QColor(color)
                        val_color.setAlpha(210)
                        p.setPen(val_color)
                        p.setFont(QFont("Menlo", 8))
                        p.drawText(dot_x + 6, gripper_y + 1, 80, _GRIPPER_TRACK_H - 2,
                                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                   f"{val:.1f} mm")

            # Curseur vertical
            cx = ns_to_x(self._cursor_ns)
            p.setPen(QPen(QColor("#f38ba8"), 1, Qt.PenStyle.SolidLine))
            p.drawLine(cx, gripper_y + 1, cx, gripper_y + _GRIPPER_TRACK_H - 1)

            gripper_y += _GRIPPER_TRACK_H

        p.end()


# ---------------------------------------------------------------------------
# _TrackerParamPanel
# ---------------------------------------------------------------------------

class _TrackerParamPanel(QWidget):
    """Panneau latéral de paramètres d'affichage des trackers 3D.

    Un seul panneau dans le VerificationWidget, partagé par toutes les colonnes.
    S'ouvre / se ferme par animation de largeur.
    """

    changed              = pyqtSignal(dict)   # émet le dict de config complet à chaque changement
    tracker_swap         = pyqtSignal(str, str)   # (tracker_a, tracker_b)
    camera_rotate        = pyqtSignal(str)         # rotation 180° d'une caméra
    camera_rename        = pyqtSignal(str, str)    # (old_name, new_name)
    gripper_visibility   = pyqtSignal(bool)        # toggle visibilité grippers
    tracker_layout_changed = pyqtSignal(str)       # "split" | "unified"

    _W_OPEN  = 210
    _W_CLOSE = 0

    _SS = """
        QWidget   { background: #181825; }
        QLabel    { color: #a6adc8; font-size: 10px; }
        QCheckBox { color: #cdd6f4; font-size: 10px; spacing: 5px; }
        QCheckBox::indicator {
            width: 13px; height: 13px;
            background: #313244; border: 1px solid #45475a; border-radius: 2px;
        }
        QCheckBox::indicator:checked {
            background: #89b4fa; border: 1px solid #89b4fa;
        }
        QSlider::groove:horizontal {
            height: 4px; background: #313244; border-radius: 2px;
        }
        QSlider::handle:horizontal {
            width: 13px; height: 13px; margin: -5px 0;
            background: #89b4fa; border-radius: 6px;
        }
        QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 2px; }
        QComboBox {
            background: #313244; color: #cdd6f4;
            border: 1px solid #45475a; border-radius: 3px;
            padding: 2px 6px; font-size: 10px;
        }
        QComboBox::drop-down { border: none; width: 14px; }
        QComboBox QAbstractItemView {
            background: #313244; color: #cdd6f4;
            selection-background-color: #45475a;
            border: 1px solid #45475a;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self._SS)
        self.setMinimumWidth(150)    # largeur minimale quand ouvert

        # Scroll area
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #181825; }")

        inner = QWidget()
        inner.setStyleSheet("background: #181825;")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ── Titre ───────────────────────────────────────────────────────
        t = QLabel("⚙  Trackers 3D")
        t.setStyleSheet("color: #cdd6f4; font-size: 12px; font-weight: bold; padding-bottom: 2px;")
        lay.addWidget(t)
        lay.addWidget(self._sep())

        # ── Sphère ──────────────────────────────────────────────────────
        lay.addWidget(self._section("Sphère"))
        self._sld_sphere, _ = self._slider("Taille", 4, 50, 14, lay,
                                            fmt=lambda v: f"{v} px")
        lay.addWidget(self._sep())

        # ── Système de coordonnées ───────────────────────────────────────
        lay.addWidget(self._section("Coordonnées"))
        info = QLabel("CSV (x,y,z) → axes GL\nSteamVR : Y=haut, Z=profondeur")
        info.setStyleSheet("color: #585b70; font-size: 9px;")
        info.setWordWrap(True)
        lay.addWidget(info)
        lay.addWidget(QLabel("Remapping des axes"))
        self._cmb_remap = QComboBox()
        self._cmb_remap.addItems(list(Viewer3DWidget.AXIS_REMAPS.keys()))
        # Défaut : SteamVR→GL
        self._cmb_remap.setCurrentText("X→X  Y→Z  Z→Y  (SteamVR→GL)")
        self._cmb_remap.currentIndexChanged.connect(self._emit)
        lay.addWidget(self._cmb_remap)
        lay.addWidget(self._sep())

        # ── Trajectoire ─────────────────────────────────────────────────
        lay.addWidget(self._section("Trajectoire"))
        self._sld_traj_w, _ = self._slider("Épaisseur", 1, 12, 2, lay,
                                            fmt=lambda v: f"{v} px")
        lay.addWidget(QLabel("Palette de couleurs"))
        self._cmb_palette = QComboBox()
        self._cmb_palette.addItems(list(_TRAJ_PALETTES.keys()))
        self._cmb_palette.currentIndexChanged.connect(self._emit)
        lay.addWidget(self._cmb_palette)
        lay.addWidget(self._sep())

        # ── Axes orientation ─────────────────────────────────────────────
        lay.addWidget(self._section("Axes d'orientation"))
        self._chk_axes = QCheckBox("Afficher")
        self._chk_axes.setChecked(True)
        self._chk_axes.stateChanged.connect(self._emit)
        lay.addWidget(self._chk_axes)
        self._sld_axis_len, _ = self._slider("Longueur", 2, 40, 12, lay,
                                              fmt=lambda v: f"{v} cm")
        self._sld_axis_w, _   = self._slider("Épaisseur", 1, 8, 3, lay,
                                              fmt=lambda v: f"{v} px")
        lay.addWidget(self._sep())

        # ── Scène ────────────────────────────────────────────────────────
        lay.addWidget(self._section("Scène"))
        self._chk_grid = QCheckBox("Grille")
        self._chk_grid.setChecked(True)
        self._chk_grid.stateChanged.connect(self._emit)
        lay.addWidget(self._chk_grid)

        self._chk_world = QCheckBox("Axes monde")
        self._chk_world.setChecked(True)
        self._chk_world.stateChanged.connect(self._emit)
        lay.addWidget(self._chk_world)

        self._sld_cam, _ = self._slider("Distance caméra", 3, 100, 20, lay,
                                         fmt=lambda v: f"{v*0.1:.1f} m")
        lay.addWidget(self._sep())

        # ── Vue 3D hauteur ───────────────────────────────────────────────
        lay.addWidget(self._section("Disposition"))
        self._sld_3d_ratio, _ = self._slider("Hauteur vue 3D", 10, 70, 35, lay,
                                              fmt=lambda v: f"{v} %")
        lay.addWidget(self._sep())

        # ── Swap trackers ────────────────────────────────────────────────
        lay.addWidget(self._section("Swap trackers"))
        _SWAPS = [("head", "left"), ("head", "right"), ("left", "right")]
        _SWAP_COLORS = {
            "head":  "#5b7df5",
            "left":  "#22d386",
            "right": "#f5c542",
        }
        for a, b in _SWAPS:
            ca, cb = _SWAP_COLORS[a], _SWAP_COLORS[b]
            label = (
                f"<span style='color:{ca};font-weight:bold'>{a}</span>"
                f" ↔ "
                f"<span style='color:{cb};font-weight:bold'>{b}</span>"
            )
            btn = QPushButton()
            btn.setFlat(True)
            lbl_w = QLabel(label)
            lbl_w.setTextFormat(Qt.TextFormat.RichText)
            lbl_w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            row_w = QWidget()
            row_w.setStyleSheet(
                "QWidget { background: #252535; border-radius: 4px; }"
                "QWidget:hover { background: #313244; }"
            )
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(6, 4, 6, 4)
            rl.addWidget(lbl_w)
            # Invisible button overlaid
            btn = QPushButton(row_w)
            btn.setFlat(True)
            btn.setStyleSheet("QPushButton { background: transparent; border: none; }")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.move(0, 0)
            btn.resize(row_w.sizeHint())
            btn.clicked.connect(lambda _, x=a, y=b: self.tracker_swap.emit(x, y))
            row_w.mousePressEvent = lambda _e, x=a, y=b: self.tracker_swap.emit(x, y)
            lay.addWidget(row_w)

        lay.addWidget(self._sep())

        # ── Caméras ──────────────────────────────────────────────────────
        lay.addWidget(self._section("Caméras"))

        # Rotation 180°
        lay.addWidget(QLabel("Rotation 180°"))
        self._cam_rotate_buttons: dict[str, QPushButton] = {}
        for cam in ("left", "head", "right"):
            color = _SWAP_COLORS.get(cam, "#cdd6f4")
            btn_rot = QPushButton(f"↺  {cam}")
            btn_rot.setStyleSheet(
                f"QPushButton {{ background: #252535; color: {color}; border: none; "
                f"border-radius: 4px; padding: 4px 8px; font-size: 11px; font-weight: bold; }}"
                f"QPushButton:hover {{ background: #313244; }}"
                f"QPushButton:pressed {{ background: #45475a; }}"
                f"QPushButton:disabled {{ color: #45475a; background: #1e1e2e; }}"
            )
            btn_rot.clicked.connect(lambda _, c=cam: self.camera_rotate.emit(c))
            lay.addWidget(btn_rot)
            self._cam_rotate_buttons[cam] = btn_rot

        lay.addSpacing(4)

        # Renommer caméra
        lay.addWidget(QLabel("Renommer caméra"))
        from PyQt6.QtWidgets import QLineEdit
        self._rename_combos: dict[str, QLineEdit] = {}
        for cam in ("left", "head", "right"):
            color = _SWAP_COLORS.get(cam, "#cdd6f4")
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            lbl_cam = QLabel(f"<span style='color:{color};font-weight:bold'>{cam}</span>")
            lbl_cam.setTextFormat(Qt.TextFormat.RichText)
            lbl_cam.setFixedWidth(34)
            edit = QLineEdit()
            edit.setPlaceholderText(cam)
            edit.setStyleSheet(
                "QLineEdit { background: #252535; color: #cdd6f4; border: 1px solid #45475a; "
                "border-radius: 3px; padding: 2px 5px; font-size: 10px; }"
                "QLineEdit:focus { border: 1px solid #89b4fa; }"
            )
            btn_ok = QPushButton("✓")
            btn_ok.setFixedWidth(22)
            btn_ok.setStyleSheet(
                "QPushButton { background: #313244; color: #a6e3a1; border: none; "
                "border-radius: 3px; font-size: 11px; }"
                "QPushButton:hover { background: #45475a; }"
            )
            btn_ok.clicked.connect(
                lambda _, c=cam, e=edit: self.camera_rename.emit(c, e.text().strip()) if e.text().strip() else None
            )
            edit.returnPressed.connect(
                lambda c=cam, e=edit: self.camera_rename.emit(c, e.text().strip()) if e.text().strip() else None
            )
            row.addWidget(lbl_cam)
            row.addWidget(edit, stretch=1)
            row.addWidget(btn_ok)
            lay.addLayout(row)
            self._rename_combos[cam] = edit

        lay.addWidget(self._sep())

        # ── Disposition trackers ──────────────────────────────────────────
        lay.addWidget(self._section("Disposition trackers"))
        self._cmb_tracker_layout = QComboBox()
        self._cmb_tracker_layout.addItem("3 vues  (par colonne)", "split")
        self._cmb_tracker_layout.addItem("1 vue   (unifiée)", "unified")
        self._cmb_tracker_layout.currentIndexChanged.connect(
            lambda _: self.tracker_layout_changed.emit(
                self._cmb_tracker_layout.currentData()
            )
        )
        lay.addWidget(self._cmb_tracker_layout)
        lay.addWidget(self._sep())

        # ── Grippers ─────────────────────────────────────────────────────
        lay.addWidget(self._section("Grippers"))
        self._chk_grippers = QCheckBox("Afficher les grippers")
        self._chk_grippers.setChecked(True)
        self._chk_grippers.stateChanged.connect(
            lambda v: self.gripper_visibility.emit(bool(v))
        )
        lay.addWidget(self._chk_grippers)

        lay.addStretch()

        scroll.setWidget(inner)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    def _emit(self) -> None:
        self.changed.emit(self.config())

    def config(self) -> dict:
        return dict(
            sphere_size  = self._sld_sphere.value(),
            traj_width   = float(self._sld_traj_w.value()),
            palette      = self._cmb_palette.currentText(),
            axis_remap   = self._cmb_remap.currentText(),
            axis_len     = self._sld_axis_len.value() * 0.01,
            axis_width   = float(self._sld_axis_w.value()),
            show_axes    = self._chk_axes.isChecked(),
            show_grid    = self._chk_grid.isChecked(),
            show_world_axis = self._chk_world.isChecked(),
            cam_distance = self._sld_cam.value() * 0.1,
        )

    @property
    def viewer_3d_ratio(self) -> int:
        """Pourcentage de hauteur accordé à la vue 3D dans le splitter."""
        return self._sld_3d_ratio.value()

    # ── Helpers UI ────────────────────────────────────────────────────

    def _section(self, text: str) -> QLabel:
        l = QLabel(text.upper())
        l.setStyleSheet(
            "color: #585b70; font-size: 9px; font-weight: bold; letter-spacing: 1px;"
        )
        return l

    def _sep(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("QFrame { color: #313244; max-height: 1px; }")
        return f

    def _slider(self, label: str, lo: int, hi: int, default: int,
                parent_lay, fmt=None):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl_n = QLabel(label)
        lbl_v = QLabel(fmt(default) if fmt else str(default))
        lbl_v.setStyleSheet("color: #cdd6f4; font-size: 10px; min-width: 40px;")
        lbl_v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl_n, stretch=1)
        row.addWidget(lbl_v)
        parent_lay.addLayout(row)

        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(lo, hi)
        sld.setValue(default)
        if fmt:
            sld.valueChanged.connect(lambda v, l=lbl_v, f=fmt: l.setText(f(v)))
        else:
            sld.valueChanged.connect(lambda v, l=lbl_v: l.setText(str(v)))
        sld.valueChanged.connect(lambda _: self._emit())
        parent_lay.addWidget(sld)
        return sld, lbl_v


# ---------------------------------------------------------------------------
# _CameraColumn
# ---------------------------------------------------------------------------

class _CameraColumn(QWidget):
    """Vertical column: video source combo + video cell + optional 3D tracker."""

    name_change_requested = pyqtSignal(str, str)  # (current_video, requested_video)

    def __init__(self, source_name: str, tracker_name: str = None, parent=None):
        super().__init__(parent)
        self._source      = source_name
        self._block_combo = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        self._combo = QComboBox()
        self._combo.addItems(_ALL_POSITIONS)
        self._combo.setCurrentText(source_name)
        self._combo.setFixedHeight(22)
        self._combo.setStyleSheet(_COMBO_STYLE)
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        lay.addWidget(self._combo)

        self.viewer_3d: Viewer3DWidget | None = None
        if tracker_name is not None:
            splitter = QSplitter(Qt.Orientation.Vertical)
            splitter.setStyleSheet(
                "QSplitter::handle { background: #45475a; height: 4px; }"
            )
            self.video_cell = _VideoCell()
            splitter.addWidget(self.video_cell)

            self.viewer_3d = Viewer3DWidget(tracker_name=tracker_name)
            self.viewer_3d.setMinimumHeight(120)
            splitter.addWidget(self.viewer_3d)

            splitter.setSizes([650, 350])
            splitter.setCollapsible(0, False)
            splitter.setCollapsible(1, False)
            lay.addWidget(splitter, stretch=1)
        else:
            self.video_cell = _VideoCell()
            lay.addWidget(self.video_cell, stretch=1)

    @property
    def source(self) -> str:
        return self._source

    def set_source(self, name: str, trajectory=None) -> None:  # noqa: ARG002
        self._source      = name
        self._block_combo = True
        self._combo.setCurrentText(name)
        self._block_combo = False
        self.video_cell.clear_frame()

    def set_tracker_transform(self, transform: "Transform3D") -> None:
        if self.viewer_3d is not None:
            self.viewer_3d.update_cursors({self.viewer_3d.tracker_name: transform},
                                          self.video_cell._frame_idx)

    def set_tracker_trajectory(self, traj: np.ndarray) -> None:
        if self.viewer_3d is not None:
            self.viewer_3d.build({self.viewer_3d.tracker_name: traj})

    def _on_combo_changed(self, new_name: str) -> None:
        if not self._block_combo and new_name != self._source:
            self.name_change_requested.emit(self._source, new_name)


# ---------------------------------------------------------------------------
# VerificationWidget
# ---------------------------------------------------------------------------

class VerificationWidget(QWidget):
    """Clock-driven multi-camera verification player.

    All cameras are seeked to an absolute frame index derived from a shared
    nanosecond clock.  There is no sequential read() — every displayed frame
    is the result of an explicit, reproducible seek.
    """

    validated               = pyqtSignal()
    rejected                = pyqtSignal()
    swap_requested          = pyqtSignal(str, str)   # video swap
    tracker_swap_requested  = pyqtSignal(str, str)   # tracker swap
    camera_rotate_requested = pyqtSignal(str)         # rotation 180° d'une caméra
    camera_rename_requested = pyqtSignal(str, str)    # (old_name, new_name)

    _SLOT_ORDER = ["left", "head", "right"]

    def __init__(self, parent=None):
        super().__init__(parent)

        # Session state
        self._session       = None
        self._trajectories: dict[str, Optional[np.ndarray]] = {}
        self._capture_ns: dict[str, Optional[np.ndarray]] = {}
        self._frame_counts: dict[str, int] = {}
        self._master_pos    = "head"

        # Clock state
        self._master_idx    = 0    # current frame of the master camera
        self._start_ns      = 0.0
        self._end_ns        = 1.0
        self._gripper_ref_ns = 0.0  # _ref_time in ns — origin of gripper timestamps

        # Decoder threads
        self._decoders: dict[str, _VideoDecodeThread] = {}

        # Playback clock
        self._playing       = False
        self._speed_idx     = _DEFAULT_SPEED_IDX
        self._clock_timer   = QTimer(self)
        self._clock_timer.timeout.connect(self._clock_tick)

        # Columns
        self._columns: dict[str, _CameraColumn] = {}

        # Tracker layout mode: "split" (3 vues par colonne) ou "unified" (1 vue)
        self._tracker_layout: str = "split"
        self._viewer_unified: Viewer3DWidget | None = None

        self._setup_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setStyleSheet("background: #1e1e2e;")
        self.setMinimumSize(800, 500)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Zone principale ─────────────────────────────────────────────
        # Layout: [  colonnes vidéo (chacune avec vue 3D)  ] | [ panneau ⚙ ]
        #
        # _main_splitter (H): contenu | param_panel
        #   contenu = _vert_splitter (V): colonnes vidéo+3D / gripper
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setStyleSheet(
            "QSplitter::handle:horizontal {"
            "  width: 4px; background: #313244;"
            "}"
            "QSplitter::handle:horizontal:hover {"
            "  background: #89b4fa;"
            "}"
        )
        self._main_splitter.setHandleWidth(4)

        # Vertical splitter: vidéos (haut) / viewer 3D (bas)
        self._vert_splitter = QSplitter(Qt.Orientation.Vertical)
        self._vert_splitter.setStyleSheet(
            "QSplitter::handle:vertical { height: 4px; background: #313244; }"
            "QSplitter::handle:vertical:hover { background: #89b4fa; }"
        )
        self._vert_splitter.setHandleWidth(4)

        # ── Colonnes vidéo + trackers 3D par colonne ──────────────────
        cols_w = QWidget()
        cols_l = QHBoxLayout(cols_w)
        cols_l.setContentsMargins(0, 0, 0, 0)
        cols_l.setSpacing(4)
        for slot in self._SLOT_ORDER:
            col = _CameraColumn(slot, tracker_name=slot)
            col.name_change_requested.connect(self._on_name_change_requested)
            self._columns[slot] = col
            cols_l.addWidget(col, stretch=1)

        self._vert_splitter.addWidget(cols_w)

        # ── Viewer 3D unifié (mode "1 vue") — caché par défaut ────────────
        self._viewer_unified = Viewer3DWidget()
        self._viewer_unified.setVisible(False)
        self._vert_splitter.addWidget(self._viewer_unified)

        # ── Gripper timeline ───────────────────────────────────────────
        self._gripper_widget = GripperTimelineWidget()
        self._vert_splitter.addWidget(self._gripper_widget)

        self._vert_splitter.setSizes([900, 0, 120])
        self._vert_splitter.setCollapsible(0, False)
        self._vert_splitter.setCollapsible(1, False)
        self._vert_splitter.setCollapsible(2, True)

        self._main_splitter.addWidget(self._vert_splitter)

        # ── Panneau paramètres ─────────────────────────────────────────
        self._param_panel = _TrackerParamPanel()
        self._param_panel.changed.connect(self._on_params_changed)
        self._param_panel.tracker_swap.connect(self.tracker_swap_requested)
        self._param_panel.camera_rotate.connect(self.camera_rotate_requested)
        self._param_panel.camera_rename.connect(self.camera_rename_requested)
        self._param_panel.gripper_visibility.connect(self._on_gripper_visibility)
        self._param_panel.tracker_layout_changed.connect(self._on_tracker_layout_changed)
        self._main_splitter.addWidget(self._param_panel)

        # Panneau fermé par défaut
        self._main_splitter.setSizes([1, 0])
        self._main_splitter.setCollapsible(0, False)
        self._main_splitter.setCollapsible(1, True)

        root.addWidget(self._main_splitter, stretch=1)

        # ── Barre de transport ─────────────────────────────────────────
        tb = QWidget()
        tb.setFixedHeight(42)
        tb.setStyleSheet("background: #181825; border-radius: 4px;")
        tl = QHBoxLayout(tb)
        tl.setContentsMargins(8, 4, 8, 4)
        tl.setSpacing(6)

        def _tbtn(text: str, tip: str, w: int = 36) -> QPushButton:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setStyleSheet(_BTN)
            b.setFixedWidth(w)
            return b

        self._btn_begin = _tbtn("⏮", "Début (Home)")
        self._btn_begin.clicked.connect(self._go_begin)
        tl.addWidget(self._btn_begin)

        self._btn_prev = _tbtn("◀", "Frame précédente (←)")
        self._btn_prev.clicked.connect(lambda: self._step(-1))
        tl.addWidget(self._btn_prev)

        self._btn_play = _tbtn("▶", "Lecture / Pause (Espace)", w=44)
        self._btn_play.clicked.connect(self._toggle_play)
        tl.addWidget(self._btn_play)

        self._btn_next = _tbtn("▶|", "Frame suivante (→)")
        self._btn_next.clicked.connect(lambda: self._step(1))
        tl.addWidget(self._btn_next)

        self._btn_end = _tbtn("⏭", "Fin (End)")
        self._btn_end.clicked.connect(self._go_end)
        tl.addWidget(self._btn_end)

        tl.addSpacing(16)

        self._spd_btns: list[QPushButton] = []
        for i, spd in enumerate(_SPEEDS):
            b = QPushButton(f"{spd:.2g}×")
            b.setCheckable(True)
            b.setChecked(i == _DEFAULT_SPEED_IDX)
            b.setStyleSheet(_BTN)
            b.setFixedWidth(44)
            b.clicked.connect(lambda _, idx=i: self._set_speed(idx))
            tl.addWidget(b)
            self._spd_btns.append(b)

        tl.addStretch()

        self._frame_label = QLabel("— / —")
        self._frame_label.setStyleSheet(
            "color: #a6adc8; font-size: 11px; font-family: monospace;")
        tl.addWidget(self._frame_label)

        # Bouton ⚙ — ouvre/ferme le panneau paramètres
        self._btn_params = QPushButton("⚙")
        self._btn_params.setToolTip("Paramètres d'affichage des trackers 3D")
        self._btn_params.setCheckable(True)
        self._btn_params.setFixedWidth(34)
        self._btn_params.setStyleSheet(
            _BTN +
            "QPushButton:checked { background: #45475a; color: #89b4fa; }"
        )
        self._btn_params.clicked.connect(self._toggle_params_panel)
        tl.addWidget(self._btn_params)

        root.addWidget(tb)

        # ── Timeline ───────────────────────────────────────────────────
        self._timeline = _TimelineBar()
        self._timeline.cursor_moved.connect(self._on_user_seek)
        root.addWidget(self._timeline)

        # ── Barre du bas ───────────────────────────────────────────────
        bb = QWidget()
        bb.setFixedHeight(52)
        bl = QHBoxLayout(bb)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(12)
        self._info_label = QLabel(
            "Vérifiez les vidéos et les trajectoires, puis validez ou rejetez.")
        self._info_label.setStyleSheet("color: #a6adc8; font-size: 12px;")
        bl.addWidget(self._info_label, stretch=1)
        self._validate_btn = QPushButton("✓ Valider")
        self._validate_btn.setStyleSheet(_BTN_OK)
        self._validate_btn.clicked.connect(self._on_validate)
        bl.addWidget(self._validate_btn)
        self._reject_btn = QPushButton("✕ Rejeter")
        self._reject_btn.setStyleSheet(_BTN_KO)
        self._reject_btn.clicked.connect(self._on_reject)
        bl.addWidget(self._reject_btn)
        root.addWidget(bb)

    # ------------------------------------------------------------------
    # Panneau paramètres
    # ------------------------------------------------------------------

    def _toggle_params_panel(self) -> None:
        sizes = self._main_splitter.sizes()
        if sizes[1] == 0:
            # Ouvrir : donner 210 px au panneau
            total = sizes[0] + sizes[1]
            self._main_splitter.setSizes([max(0, total - 210), 210])
        else:
            # Fermer
            self._main_splitter.setSizes([sizes[0] + sizes[1], 0])

    def _on_params_changed(self, cfg: dict) -> None:
        """Propage les paramètres à tous les viewers 3D actifs."""
        for col in self._columns.values():
            if col.viewer_3d is not None:
                col.viewer_3d.apply_settings(cfg)
        if self._viewer_unified is not None:
            self._viewer_unified.apply_settings(cfg)

    def _on_tracker_layout_changed(self, mode: str) -> None:
        self._tracker_layout = mode
        split = (mode == "split")
        # Colonnes : montrer/cacher le viewer 3D intégré
        for col in self._columns.values():
            if col.viewer_3d is not None:
                col.viewer_3d.setVisible(split)
        # Viewer unifié
        self._viewer_unified.setVisible(not split)
        # Ajuster les tailles du vert_splitter
        total = self._vert_splitter.height() or 1000
        gripper_h = 120 if self._gripper_widget.isVisible() else 0
        if split:
            self._vert_splitter.setSizes([total - gripper_h, 0, gripper_h])
        else:
            content_h = total - gripper_h
            self._vert_splitter.setSizes([
                content_h * 55 // 100,
                content_h * 45 // 100,
                gripper_h,
            ])
        # Reconstruire les trajectoires dans le viewer qui devient visible
        if self._session is not None:
            all_pos = self._session.get_all_tracker_positions()
            if not split:
                trajectories = {n: pts for n, pts in all_pos.items()
                                if pts is not None and len(pts) >= 2}
                self._viewer_unified.build(trajectories)
                cfg = self._param_panel.config()
                self._viewer_unified.apply_settings(cfg)
            # Re-sync curseurs
            t_ns = self._timeline.cursor_ns
            self._sync_trackers(t_ns)

    def _on_gripper_visibility(self, visible: bool) -> None:
        self._gripper_widget.setVisible(visible)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        k = event.key()
        if k == Qt.Key.Key_Left:
            self._pause()
            self._step(-1)
        elif k == Qt.Key.Key_Right:
            self._pause()
            self._step(1)
        elif k == Qt.Key.Key_Space:
            self._toggle_play()
        elif k == Qt.Key.Key_Home:
            self._go_begin()
        elif k == Qt.Key.Key_End:
            self._go_end()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Clock-driven playback
    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if not self._decoders:
            return
        # Stop at last frame
        total = self._frame_counts.get(self._master_pos, 0)
        if total > 0 and self._master_idx >= total - 1:
            self._seek_master(0)   # loop back to beginning
        self._playing = True
        self._btn_play.setText("⏸")
        fps = 30.0
        if self._session is not None:
            cam = self._session.cameras.get(self._master_pos)
            if cam and cam.fps > 0:
                fps = cam.fps
        interval_ms = max(1, round(1000.0 / (fps * _SPEEDS[self._speed_idx])))
        self._clock_timer.start(interval_ms)

    def _pause(self) -> None:
        if self._playing:
            self._playing = False
            self._clock_timer.stop()
            self._btn_play.setText("▶")

    def _clock_tick(self) -> None:
        """Advance master index by 1 and seek all cameras."""
        total = self._frame_counts.get(self._master_pos, 0)
        if total <= 0:
            return
        next_idx = self._master_idx + 1
        if next_idx >= total:
            self._pause()
            return
        self._seek_master(next_idx)

    def _set_speed(self, idx: int) -> None:
        self._speed_idx = idx
        for i, b in enumerate(self._spd_btns):
            b.setChecked(i == idx)
        if self._playing:
            self._pause()
            self._play()

    # ------------------------------------------------------------------
    # Seeking — the single source of truth for all position updates
    # ------------------------------------------------------------------

    def _seek_master(self, master_idx: int) -> None:
        """Seek all cameras to the position corresponding to master_idx.

        This is the ONLY place that updates _master_idx, the timeline cursor,
        the frame label, and the tracker position.  Every navigation action
        calls this method so the UI is always consistent.
        """
        if not self._decoders:
            return

        total = self._frame_counts.get(self._master_pos, 0)
        if total <= 0:
            return
        master_idx = max(0, min(master_idx, total - 1))

        # --- Resolve master timestamp ---
        master_cap_ns = self._capture_ns.get(self._master_pos)
        if master_cap_ns is not None and len(master_cap_ns) > master_idx:
            t_ns = float(master_cap_ns[master_idx])
        else:
            # No jsonl — estimate from FPS
            fps = 30.0
            if self._session is not None:
                cam = self._session.cameras.get(self._master_pos)
                if cam and cam.fps > 0:
                    fps = cam.fps
            t_ns = self._start_ns + master_idx / fps * 1e9

        # --- Seek master decoder ---
        if self._master_pos in self._decoders:
            self._decoders[self._master_pos].seek(master_idx)

        # --- Seek slave cameras to closest timestamp ---
        for pos, thread in self._decoders.items():
            if pos == self._master_pos:
                continue
            cap_ns = self._capture_ns.get(pos)
            slave_total = self._frame_counts.get(pos, 0)
            if slave_total <= 0:
                continue
            if cap_ns is not None and len(cap_ns) > 0:
                idx = int(np.searchsorted(cap_ns, t_ns, side="left"))
                idx = max(0, min(idx, slave_total - 1))
            else:
                idx = max(0, min(master_idx, slave_total - 1))
            thread.seek(idx)

        # --- Update UI state ---
        self._master_idx = master_idx

        # Timeline cursor (programmatic — no signal)
        self._timeline.set_cursor(t_ns)

        # Per-camera timeline markers
        for pos in list(self._decoders):
            cap_ns = self._capture_ns.get(pos)
            if cap_ns is not None and len(cap_ns) > 0:
                if pos == self._master_pos:
                    self._timeline.set_camera_ns(pos, t_ns)
                else:
                    slave_total = self._frame_counts.get(pos, 0)
                    if slave_total > 0:
                        idx = int(np.searchsorted(
                            self._capture_ns[pos], t_ns, side="left"))
                        idx = max(0, min(idx, slave_total - 1))
                        self._timeline.set_camera_ns(
                            pos, float(self._capture_ns[pos][idx]))

        # Frame counter label
        self._frame_label.setText(f"{master_idx} / {total - 1}")

        # Tracker sync
        self._sync_trackers(t_ns)

    def _seek_to_ns(self, t_ns: float) -> None:
        """Seek all cameras to an absolute timestamp (user timeline interaction)."""
        if not self._decoders:
            return
        master_cap_ns = self._capture_ns.get(self._master_pos)
        total = self._frame_counts.get(self._master_pos, 0)
        if total <= 0:
            return

        if master_cap_ns is not None and len(master_cap_ns) > 0:
            idx = int(np.searchsorted(master_cap_ns, t_ns, side="left"))
            idx = max(0, min(idx, total - 1))
        else:
            fps = 30.0
            if self._session is not None:
                cam = self._session.cameras.get(self._master_pos)
                if cam and cam.fps > 0:
                    fps = cam.fps
            elapsed_s = max(0.0, (t_ns - self._start_ns) / 1e9)
            idx = max(0, min(round(elapsed_s * fps), total - 1))

        self._seek_master(idx)

    def _step(self, delta: int) -> None:
        self._seek_master(self._master_idx + delta)

    def _go_begin(self) -> None:
        self._pause()
        self._seek_master(0)

    def _go_end(self) -> None:
        self._pause()
        total = self._frame_counts.get(self._master_pos, 0)
        if total > 0:
            self._seek_master(total - 1)

    # ------------------------------------------------------------------
    # User timeline drag / click
    # ------------------------------------------------------------------

    def _on_user_seek(self, t_ns: float) -> None:
        self._pause()
        self._seek_to_ns(t_ns)

    # ------------------------------------------------------------------
    # Tracker sync
    # ------------------------------------------------------------------

    def _sync_trackers(self, t_ns: float) -> None:
        if self._session is None:
            return
        traj_idx = self._session.get_tracker_index_at_ns(t_ns)
        try:
            tracker_states = self._session.get_tracker_state_at_ns(t_ns)
        except Exception:
            tracker_states = {}

        # Build transforms dict
        transforms: dict[str, Transform3D] = {}
        for name, state in tracker_states.items():
            transforms[name] = Transform3D(
                position=state["position"],
                rotation=state["quaternion"],
                rotation_format="quaternion",
            )

        if self._tracker_layout == "split":
            # Envoyer chaque tracker au viewer de sa colonne
            for slot, col in self._columns.items():
                if col.viewer_3d is None:
                    continue
                tracker_name = col.viewer_3d.tracker_name
                t = transforms.get(tracker_name)
                if t is not None:
                    col.viewer_3d.update_cursors({tracker_name: t}, traj_idx)
        else:
            # Envoyer tous les trackers au viewer unifié
            if transforms and self._viewer_unified is not None:
                self._viewer_unified.update_cursors(transforms, traj_idx)

        # Gripper cursor
        self._gripper_widget.set_cursor_ns(t_ns)

    # ------------------------------------------------------------------
    # Decoder thread callbacks
    # ------------------------------------------------------------------

    def _on_frame_ready(self, position: str, frame_idx: int,
                        frame: np.ndarray) -> None:
        """Display the decoded frame in the matching column."""
        for col in self._columns.values():
            if col.source == position:
                col.video_cell.set_frame(frame, frame_idx)
                break

    # ------------------------------------------------------------------
    # Decoder lifecycle
    # ------------------------------------------------------------------

    def _start_decoders(self, session) -> None:
        self._stop_decoders()
        self._frame_counts.clear()
        self._capture_ns.clear()

        for pos in _ALL_POSITIONS:
            cam = session.cameras.get(pos)
            if cam is None:
                continue
            vp = cam.video_path
            if vp is None or not vp.exists():
                continue
            if cam.frame_count <= 0:
                continue

            t = _VideoDecodeThread(pos, str(vp), cam.frame_count, parent=self)
            t.frame_ready.connect(self._on_frame_ready)
            t.start()
            self._decoders[pos]     = t
            self._frame_counts[pos] = cam.frame_count
            self._capture_ns[pos]   = cam.frame_capture_ns

            for col in self._columns.values():
                if col.source == pos:
                    col.video_cell.set_frame_count(cam.frame_count)

        # Master = head if available, else first found
        self._master_pos = next(
            (p for p in ("head", "left", "right") if p in self._decoders),
            next(iter(self._decoders), "head"),
        )

    def _stop_decoders(self) -> None:
        self._pause()
        for t in self._decoders.values():
            t.stop()
        for t in self._decoders.values():
            t.wait(3000)
        self._decoders.clear()

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------

    def _on_name_change_requested(self, current: str, requested: str) -> None:
        """Swap visuel des sources vidéo entre deux colonnes."""
        other = next(
            (s for s, c in self._columns.items() if c.source == requested), None)
        origin = next(
            (s for s, c in self._columns.items() if c.source == current), None)
        if other is None or origin is None or other == origin:
            return
        self._columns[origin].set_source(requested, None)
        self._columns[other].set_source(current, None)
        self.swap_requested.emit(current, requested)


    # ------------------------------------------------------------------
    # Validate / Reject
    # ------------------------------------------------------------------

    def _on_validate(self) -> None:
        self._pause()
        self.validated.emit()

    def _on_reject(self) -> None:
        self._pause()
        self.rejected.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_session(self, session) -> None:
        self._stop_decoders()
        self._session = session
        if session is None:
            return

        # Load trajectories into all viewers
        all_pos = session.get_all_tracker_positions()
        self._trajectories = {n: all_pos.get(n) for n in _ALL_POSITIONS}
        trajectories_valid = {n: pts for n, pts in all_pos.items()
                              if pts is not None and len(pts) >= 2}
        # Per-column viewers (mode split)
        for slot, col in self._columns.items():
            if col.viewer_3d is None:
                continue
            tracker_name = col.viewer_3d.tracker_name
            traj = trajectories_valid.get(tracker_name)
            if traj is not None:
                col.viewer_3d.build({tracker_name: traj})
        # Viewer unifié (mode unified) — chargé même si caché, prêt à l'usage
        if self._viewer_unified is not None:
            self._viewer_unified.build(trajectories_valid)

        for slot in self._SLOT_ORDER:
            self._columns[slot].set_source(slot)

        # Timeline range
        self._start_ns, self._end_ns = session.get_timeline_ns()
        self._timeline.set_range(self._start_ns, self._end_ns)

        # Gripper timestamps are relative to session._ref_time (metadata.start_time)
        try:
            self._gripper_ref_ns = float(session._ref_time.value)
        except Exception:
            self._gripper_ref_ns = self._start_ns

        # Start decoders
        self._master_idx = 0
        self._start_decoders(session)

        # Activer les boutons rotation uniquement pour les caméras présentes
        for cam, btn in self._param_panel._cam_rotate_buttons.items():
            btn.setEnabled(cam in self._decoders)

        # Auto-detect axis mapping from CSV data and apply to param panel
        try:
            detected = session.detect_axis_remap()
            self._param_panel._cmb_remap.setCurrentText(detected)
            # _cmb_remap.currentIndexChanged triggers _emit → _on_params_changed
            # so all viewers get the correct mapping immediately
        except Exception:
            pass

        # Gripper timeline
        gripper_data = {}
        for side in session.metadata.gripper_sides:
            try:
                ts, openings = session.get_gripper_timeseries(side)
                if ts is not None:
                    gripper_data[side] = (ts, openings)
            except Exception:
                pass
        self._gripper_widget.set_range(self._start_ns, self._end_ns)
        self._gripper_widget.set_gripper_ref_ns(self._gripper_ref_ns)
        self._gripper_widget.set_data(gripper_data)
        self._gripper_widget.setVisible(bool(gripper_data))

        # Seek to frame 0 — also updates label, timeline cursor, trackers
        self._seek_master(0)

    def reload_trackers_only(self, session) -> None:
        """Swap CSV vient d'être appliqué — mettre à jour la session et le viewer 3D
        sans toucher aux colonnes vidéo ni relancer les décodeurs.

        À appeler après swap_trackers_on_disk pour conserver l'état vidéo courant.
        """
        self._session = session

        # Rebuild trajectories dans tous les viewers
        all_pos = session.get_all_tracker_positions()
        self._trajectories = {n: all_pos.get(n) for n in _ALL_POSITIONS}
        trajectories_valid = {n: pts for n, pts in all_pos.items()
                              if pts is not None and len(pts) >= 2}
        for slot, col in self._columns.items():
            if col.viewer_3d is None:
                continue
            tracker_name = col.viewer_3d.tracker_name
            traj = trajectories_valid.get(tracker_name)
            if traj is not None:
                col.viewer_3d.build({tracker_name: traj})
        if self._viewer_unified is not None:
            self._viewer_unified.build(trajectories_valid)

        # Re-sync les curseurs au timestamp courant
        master_cap_ns = self._capture_ns.get(self._master_pos)
        if master_cap_ns is not None and len(master_cap_ns) > self._master_idx:
            t_ns = float(master_cap_ns[self._master_idx])
        else:
            t_ns = self._timeline.cursor_ns
        self._sync_trackers(t_ns)

    def set_frames(self, frames: dict) -> None:
        """Ignored — verification widget manages its own decoding."""
        pass

    def set_current_frame(self, frame_index: int, session) -> None:
        """Seek to a frame index (called from annotation mode sync)."""
        if not self._playing:
            self._seek_master(frame_index)

    def set_info(self, text: str) -> None:
        self._info_label.setText(text)

    def set_buttons_enabled(self, enabled: bool) -> None:
        self._validate_btn.setEnabled(enabled)
        self._reject_btn.setEnabled(enabled)

    def release_decoders(self) -> None:
        """Stop and join all decoder threads, releasing every VideoCapture handle.

        Must be called before any on-disk rename/swap on Windows, where open
        file handles prevent file operations.  After joining, a small sleep
        gives the Windows kernel time to flush its handle table.
        """
        import sys, time
        self._stop_decoders()
        if sys.platform == "win32":
            time.sleep(0.15)
