"""Multi-camera video display widget.

Each camera column shows:
- Camera title
- Video frame
- 3D tracker viewer (optional)
"""

import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QFrame,
    QPushButton, QApplication, QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPoint
from PyQt6.QtGui import QImage, QPixmap, QScreen

from .viewer_3d_widget import Viewer3DWidget
from ...core.transforms import Transform3D


class FullscreenVideoWindow(QWidget):
    """Fenêtre plein écran détachée affichant le flux d'une caméra."""

    closed = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(title)
        self.setStyleSheet("background: black;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._label)

        self._pixmap = None

    def set_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_image)
        self._refresh_scaled()

    def _refresh_scaled(self) -> None:
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self._label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_scaled()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_F11):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


class VideoFrame(QLabel):
    """Video frame display widget with locked 1920×1200 (16:10) aspect ratio."""

    _RATIO_W = 1920
    _RATIO_H = 1200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #11111b;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(192, 120)  # 1/10th of target resolution
        self._pixmap = None

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return round(width * self._RATIO_H / self._RATIO_W)

    def sizeHint(self) -> QSize:
        return QSize(self._RATIO_W, self._RATIO_H)

    def set_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_image)
        self._refresh_scaled()

    def _refresh_scaled(self) -> None:
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_scaled()


class CameraColumn(QWidget):
    """Single camera column: title + video (16:10) + optional 3D tracker."""

    clicked = pyqtSignal(str)        # camera_id
    swap_requested = pyqtSignal(str) # camera_id — demande d'intervertir avec le voisin droit
    fullscreen_requested = pyqtSignal(str)  # camera_id
    maximize_requested = pyqtSignal(str)  # camera_id — demande d'agrandir la caméra
    rotate_requested = pyqtSignal(str)  # camera_id — demande de rotation 180°
    tracker_swap_requested = pyqtSignal(str)  # tracker_name — demande d'échanger ce tracker

    _BTN_STYLE = (
        "QPushButton { background: #313244; color: #cdd6f4; border: none; "
        "border-radius: 2px; padding: 0 4px; font-size: 10px; }"
        "QPushButton:hover { background: #45475a; }"
    )

    def __init__(self, camera_id: str, tracker_name: str = None, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.tracker_name = tracker_name
        self.selected = False
        self._fullscreen_win: "FullscreenVideoWindow | None" = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Title bar with swap + fullscreen buttons
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(2)

        self.title_label = QLabel(f"Camera {camera_id}")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(
            "color: #cdd6f4; background: #1e1e2e; font-size: 11px; "
            "font-weight: bold; padding: 2px; border-radius: 2px;"
        )
        title_row.addWidget(self.title_label, stretch=1)

        self._rotate_btn = QPushButton("↻")
        self._rotate_btn.setToolTip("Pivoter la vidéo 180°")
        self._rotate_btn.setFixedHeight(20)
        self._rotate_btn.setStyleSheet(self._BTN_STYLE)
        self._rotate_btn.clicked.connect(lambda: self.rotate_requested.emit(self.camera_id))
        title_row.addWidget(self._rotate_btn)

        self._swap_btn = QPushButton("⇄")
        self._swap_btn.setToolTip("Intervertir avec la caméra suivante")
        self._swap_btn.setFixedHeight(20)
        self._swap_btn.setStyleSheet(self._BTN_STYLE)
        self._swap_btn.clicked.connect(lambda: self.swap_requested.emit(self.camera_id))
        title_row.addWidget(self._swap_btn)

        self._maximize_btn = QPushButton("⛶")
        self._maximize_btn.setToolTip("Agrandir cette caméra")
        self._maximize_btn.setFixedHeight(20)
        self._maximize_btn.setStyleSheet(self._BTN_STYLE)
        self._maximize_btn.clicked.connect(lambda: self.maximize_requested.emit(self.camera_id))
        title_row.addWidget(self._maximize_btn)

        self._fs_btn = QPushButton("⛉")
        self._fs_btn.setToolTip("Afficher en plein écran (2e écran)")
        self._fs_btn.setFixedHeight(20)
        self._fs_btn.setStyleSheet(self._BTN_STYLE)
        self._fs_btn.clicked.connect(lambda: self.fullscreen_requested.emit(self.camera_id))
        title_row.addWidget(self._fs_btn)

        title_widget = QWidget()
        title_widget.setFixedHeight(20)
        title_widget.setLayout(title_row)
        layout.addWidget(title_widget)

        # Video + 3D in a vertical splitter so the user can resize
        self.viewer_3d: Viewer3DWidget | None = None
        if tracker_name is not None:
            splitter = QSplitter(Qt.Orientation.Vertical)
            splitter.setStyleSheet(
                "QSplitter::handle { background: #45475a; height: 4px; }"
            )

            # Video frame
            self.frame_label = VideoFrame()
            splitter.addWidget(self.frame_label)

            # 3D viewer
            self.viewer_3d = Viewer3DWidget(tracker_name=tracker_name)
            self.viewer_3d.setMinimumHeight(120)
            self.viewer_3d.swap_requested.connect(self.tracker_swap_requested)
            splitter.addWidget(self.viewer_3d)

            # Initial split: ~65% video / ~35% 3D
            splitter.setSizes([650, 350])
            splitter.setCollapsible(0, False)
            splitter.setCollapsible(1, False)

            layout.addWidget(splitter, stretch=1)
        else:
            # No tracker — just the video frame
            self.frame_label = VideoFrame()
            layout.addWidget(self.frame_label, stretch=1)

        self._update_border()

    def set_frame(self, frame: np.ndarray) -> None:
        self.frame_label.set_frame(frame)
        if self._fullscreen_win and self._fullscreen_win.isVisible():
            self._fullscreen_win.set_frame(frame)

    def set_tracker_transform(self, transform: Transform3D) -> None:
        """Update the 3D viewer with a new tracker transform."""
        if self.viewer_3d is not None:
            self.viewer_3d.update_transform(transform)

    def set_tracker_trajectory(self, trajectory: np.ndarray) -> None:
        """Set the full trajectory path in the 3D viewer."""
        if self.viewer_3d is not None:
            self.viewer_3d.set_trajectory(trajectory)

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self._update_border()

    def open_fullscreen(self, screen: "QScreen | None" = None) -> None:
        """Ouvre ou amène au premier plan la fenêtre plein écran."""
        if self._fullscreen_win and self._fullscreen_win.isVisible():
            self._fullscreen_win.raise_()
            return

        win = FullscreenVideoWindow(f"Camera {self.camera_id}")
        win.closed.connect(self._on_fullscreen_closed)
        if screen is not None:
            geo = screen.geometry()
            win.setGeometry(geo)
            win.showFullScreen()
        else:
            win.resize(1280, 800)
            win.show()
        self._fullscreen_win = win

    def _on_fullscreen_closed(self) -> None:
        self._fullscreen_win = None

    def close_fullscreen(self) -> None:
        if self._fullscreen_win:
            self._fullscreen_win.close()
            self._fullscreen_win = None

    def _update_border(self) -> None:
        border_color = "#89b4fa" if self.selected else "#313244"
        self.setStyleSheet(
            f"CameraColumn {{ border: 2px solid {border_color}; border-radius: 4px; "
            f"background: #181825; }}"
        )

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.camera_id)


class MultiVideoWidget(QWidget):
    """Displays all cameras side by side, each with an optional 3D tracker viewer.

    Supports:
    - Swapping two adjacent camera columns (⇄ button)
    - Maximize single camera to full width (⛶ button)
    - Fullscreen detached window on secondary screen (⛉ button)
    - Per-camera 3D tracker visualization
    """

    camera_selected = pyqtSignal(str)        # camera_id
    swap_requested = pyqtSignal(str, str)    # pos_a, pos_b — demande de swap permanent sur disque
    rotate_requested = pyqtSignal(str)       # camera_id — demande de rotation 180° permanente
    tracker_swap_requested = pyqtSignal(str) # tracker_name — demande d'échanger ce tracker

    def __init__(self, camera_ids: list = None, parent=None):
        super().__init__(parent)
        # Ordered list of logical camera ids as they appear in columns
        self._order: list[str] = []
        # column widgets keyed by camera_id (logical)
        self._columns: dict[str, CameraColumn] = {}
        self._selected_camera: str = ""
        # Mapping: logical camera_id -> source data key (swapped or not)
        self._source_map: dict[str, str] = {}
        # Track maximized state
        self._maximized_camera: str = ""
        # Mapping: camera_id -> tracker_name (set by set_cameras)
        self._tracker_map: dict[str, str] = {}

        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(4)

        if camera_ids:
            self.set_cameras(camera_ids)

    def set_cameras(self, camera_ids: list, tracker_map: dict = None) -> None:
        """Initialize camera columns.

        Args:
            camera_ids: Ordered list of camera position names.
            tracker_map: Optional dict mapping camera_id -> tracker_name.
                         Pass None or omit to show no 3D viewers.
                         Pass {"head": "head", "left": "left", "right": "right"}
                         to show a 3D viewer matching each camera.
        """
        # Close any open fullscreen windows
        for col in self._columns.values():
            col.close_fullscreen()
            self.layout_.removeWidget(col)
            col.deleteLater()
        self._columns.clear()
        self._order = list(camera_ids)
        self._source_map = {cid: cid for cid in camera_ids}
        self._maximized_camera = ""
        self._tracker_map = tracker_map or {}

        for cid in camera_ids:
            tracker_name = self._tracker_map.get(cid)
            column = CameraColumn(cid, tracker_name=tracker_name)
            column.clicked.connect(self._on_camera_clicked)
            column.swap_requested.connect(self._on_swap_requested)
            column.rotate_requested.connect(self._on_rotate_requested)
            column.fullscreen_requested.connect(self._on_fullscreen_requested)
            column.maximize_requested.connect(self._on_maximize_requested)
            column.tracker_swap_requested.connect(self.tracker_swap_requested)
            self.layout_.addWidget(column)
            self._columns[cid] = column

        if camera_ids:
            self._select(camera_ids[0])

    # ------------------------------------------------------------------
    # 3D tracker API
    # ------------------------------------------------------------------

    def update_tracker(self, camera_id: str, transform: Transform3D) -> None:
        """Update the 3D viewer for a specific camera column."""
        col = self._columns.get(camera_id)
        if col is not None:
            col.set_tracker_transform(transform)

    def set_tracker_trajectories(self, trajectories: dict) -> None:
        """Set full trajectory paths. trajectories: camera_id -> np.ndarray (N,3)."""
        for cid, traj in trajectories.items():
            col = self._columns.get(cid)
            if col is not None:
                col.set_tracker_trajectory(traj)

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------

    def _on_swap_requested(self, camera_id: str) -> None:
        """Demande un swap permanent sur disque entre camera_id et son voisin droit."""
        if camera_id not in self._order:
            return
        idx = self._order.index(camera_id)
        next_idx = (idx + 1) % len(self._order)
        pos_a = self._order[idx]
        pos_b = self._order[next_idx]
        self.swap_requested.emit(pos_a, pos_b)

    # ------------------------------------------------------------------
    # Rotate
    # ------------------------------------------------------------------

    def _on_rotate_requested(self, camera_id: str) -> None:
        self.rotate_requested.emit(camera_id)

    # ------------------------------------------------------------------
    # Maximize
    # ------------------------------------------------------------------

    def _on_maximize_requested(self, camera_id: str) -> None:
        if self._maximized_camera == camera_id:
            self._restore_normal_view()
        else:
            self._maximize_camera(camera_id)

    def _maximize_camera(self, camera_id: str) -> None:
        self._maximized_camera = camera_id
        for cid, col in self._columns.items():
            if cid == camera_id:
                col.setVisible(True)
                col._maximize_btn.setToolTip("Réduire cette caméra")
                col._maximize_btn.setText("⊟")
            else:
                col.setVisible(False)

    def _restore_normal_view(self) -> None:
        self._maximized_camera = ""
        for cid, col in self._columns.items():
            col.setVisible(True)
            col._maximize_btn.setToolTip("Agrandir cette caméra")
            col._maximize_btn.setText("⛶")

    # ------------------------------------------------------------------
    # Fullscreen
    # ------------------------------------------------------------------

    def _on_fullscreen_requested(self, camera_id: str) -> None:
        col = self._columns.get(camera_id)
        if col is None:
            return
        screens = QApplication.screens()
        screen = screens[1] if len(screens) > 1 else screens[0]
        col.open_fullscreen(screen)

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def set_frames(self, frames: dict) -> None:
        """Set frames for all cameras, applying the current swap mapping."""
        for col_id, col in self._columns.items():
            src = self._source_map.get(col_id, col_id)
            frame = frames.get(src)
            if frame is not None:
                col.set_frame(frame)

    def _on_camera_clicked(self, camera_id: str) -> None:
        self._select(camera_id)
        self.camera_selected.emit(camera_id)

    def _select(self, camera_id: str) -> None:
        self._selected_camera = camera_id
        for cid, col in self._columns.items():
            col.set_selected(cid == camera_id)

    @property
    def selected_camera(self) -> str:
        return self._selected_camera
