"""Advanced annotation timeline widget.

Features:
- Multi-track display with colored annotation intervals
- In/out point selection for interval annotations
- Precise scrubbing with frame-level accuracy
- Keyboard shortcuts for fast navigation
- Visual indication of current annotation state
- Crop cursors (start/end) for trimming the video
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QSizePolicy, QToolButton, QSpacerItem,
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QRectF, QPointF, QTimer
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QLinearGradient, QFont,
    QPainterPath, QFontMetrics, QPixmap,
)
from typing import List, Optional, Tuple

from .gripper_graph_widget import GripperGraphWidget

from ...labeling.label_manager import (
    LabelManager, Annotation, LabelType, UNLABELED_LABEL_ID, IDLE_LABEL_ID
)


# Half-width (px) of the grab zone around a crop cursor handle
_CROP_GRAB_PX = 15


class AnnotationTimelineBar(QWidget):
    """Custom-painted multi-track annotation timeline."""

    frame_selected = pyqtSignal(int)
    interval_set = pyqtSignal(int, int)  # start_frame, end_frame
    crop_changed = pyqtSignal(int, int)  # crop_start_frame, crop_end_frame
    annotation_clicked = pyqtSignal(object)  # Annotation object clicked (for editing)
    segment_selected = pyqtSignal(object)    # Annotation object selected (for labeling)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_count = 0
        self.current_frame = 0
        self.fps = 30.0

        # Annotation data
        self._annotations: List[Annotation] = []
        self._label_colors: dict = {}  # label_id -> color_hex

        # In/out selection
        self.in_point: Optional[int] = None
        self.out_point: Optional[int] = None

        # Crop cursors (frame indices); None = at extremity (no crop)
        self.crop_start: int = 0
        self.crop_end: int = 0  # will be set to frame_count-1 when count is set

        # Static background cache (ruler + annotations) — invalidated when data changes
        self._static_cache = None  # QPixmap or None

        # Interaction
        self._dragging = False
        self._hover_frame = -1
        self._drag_target: str = "playhead"  # "playhead" | "crop_start" | "crop_end"

        # Selected segment for labeling (highlight)
        self._selected_segment: Optional[Annotation] = None

        # Throttle: emit frame_selected at most once per 30ms during drag
        self._drag_throttle = QTimer()
        self._drag_throttle.setSingleShot(True)
        self._drag_throttle.setInterval(30)
        self._drag_throttle.timeout.connect(self._flush_drag)
        self._pending_drag_frame: Optional[int] = None

        # Throttle for crop drag: invalidate cache at most once per 50ms
        self._crop_cache_timer = QTimer()
        self._crop_cache_timer.setSingleShot(True)
        self._crop_cache_timer.setInterval(50)
        self._crop_cache_timer.timeout.connect(self._invalidate_crop_cache)
        self._crop_cache_needs_update = False

        self.setMinimumHeight(100)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _annotation_track_bottom(self) -> int:
        """Bottom y of the annotation area."""
        return self.height() - 20

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_frame_count(self, count: int) -> None:
        self.frame_count = count
        self.crop_start = 0
        self.crop_end = max(count - 1, 0)
        self._static_cache = None
        self.update()

    def set_current_frame(self, frame: int) -> None:
        old_x = self._frame_to_x(self.current_frame)
        self.current_frame = frame
        new_x = self._frame_to_x(frame)
        dirty_left = min(old_x, new_x) - 6
        dirty_right = max(old_x, new_x) + 6
        self.update(dirty_left, 0, dirty_right - dirty_left, self.height())

    def set_fps(self, fps: float) -> None:
        self.fps = fps
        self._static_cache = None

    def set_annotations(self, annotations: List[Annotation], label_colors: dict) -> None:
        self._annotations = annotations
        self._label_colors = label_colors
        self._static_cache = None
        self.update()

    def set_in_point(self, frame: Optional[int]) -> None:
        self.in_point = frame
        self._static_cache = None
        self.update()

    def set_out_point(self, frame: Optional[int]) -> None:
        self.out_point = frame
        self._static_cache = None
        self.update()

    def set_selected_segment(self, annotation: Optional[Annotation]) -> None:
        """Highlight the given annotation as the currently selected segment."""
        self._selected_segment = annotation
        self._static_cache = None
        self.update()

    def is_crop_active(self) -> bool:
        """True when crop cursors are not at the extremities."""
        return self.crop_start > 0 or self.crop_end < max(self.frame_count - 1, 0)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _frame_to_x(self, frame: int) -> int:
        """Convert frame index to pixel x-coordinate (adaptive with margins)."""
        if self.frame_count <= 0:
            return 0
        margin = 20  # pixels on each side
        usable_width = self.width() - 2 * margin
        # Map frame 0 -> margin, frame (count-1) -> width-margin
        return margin + int(frame * usable_width / max(1, self.frame_count - 1))

    def _frame_at(self, x: int) -> int:
        """Convert pixel x-coordinate to frame index (adaptive with margins)."""
        margin = 20  # Must match _frame_to_x margin
        usable_width = self.width() - 2 * margin
        # Clamp x to usable area
        x_clamped = max(margin, min(x, self.width() - margin))
        # Map position to frame
        frame = int((x_clamped - margin) * max(1, self.frame_count - 1) / usable_width)
        return max(0, min(frame, self.frame_count - 1))

    # ------------------------------------------------------------------
    # Static layer (ruler + annotations) — cached as QPixmap
    # ------------------------------------------------------------------

    def _render_static(self, w: int, h: int):
        """Render ruler + annotations + gripper tracks into a cached QPixmap."""
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor("#181825"))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        track_width = w
        ruler_h = 16
        margin = 20
        usable_width = track_width - 2 * margin

        def frame_to_x_static(frame: int) -> int:
            if self.frame_count <= 0:
                return margin
            return margin + int(frame * usable_width / max(1, self.frame_count - 1))

        def time_to_x_static(t: float) -> int:
            """Convert time (s) to x pixel via frame mapping."""
            if self.fps <= 0 or self.frame_count <= 0:
                return margin
            frame = min(int(t * self.fps), self.frame_count - 1)
            return frame_to_x_static(frame)

        # --- Time ruler ---
        painter.fillRect(0, 0, track_width, ruler_h, QColor("#1e1e2e"))
        painter.setPen(QPen(QColor("#585b70"), 1))
        painter.setFont(QFont("Courier", 7))

        duration = (self.frame_count - 1) / self.fps if self.fps > 0 and self.frame_count > 0 else 0
        tick_interval = self._nice_tick_interval(duration, usable_width)
        t = 0.0
        while t <= duration + 0.001:
            frame_at_t = int(t * self.fps) if self.fps > 0 else int(t)
            frame_at_t = min(frame_at_t, self.frame_count - 1) if self.frame_count > 0 else 0
            x = frame_to_x_static(frame_at_t)
            painter.drawLine(x, ruler_h - 4, x, ruler_h)
            painter.drawText(x + 2, ruler_h - 2, f"{t:.1f}s")
            t += tick_interval

        # --- Annotation track layout ---
        track_top = ruler_h + 2
        track_bottom = h - 20
        track_h = max(track_bottom - track_top, 2)
        mid_y = track_top + track_h // 2

        # --- Annotation tracks background ---
        painter.fillRect(0, track_top, track_width, track_h, QColor("#11111b"))

        # Crop shading
        if self.is_crop_active() and self.frame_count > 0:
            crop_color = QColor("#000000")
            crop_color.setAlpha(90)
            x_cs = frame_to_x_static(self.crop_start)
            x_ce = frame_to_x_static(self.crop_end)
            if x_cs > margin:
                painter.fillRect(margin, track_top, x_cs - margin, track_h, crop_color)
            if x_ce < track_width - margin:
                painter.fillRect(x_ce, track_top, track_width - margin - x_ce, track_h, crop_color)

        # Separator line between two hand rows
        painter.setPen(QPen(QColor("#313244"), 1))
        painter.drawLine(0, mid_y, track_width, mid_y)

        # Row labels
        painter.setFont(QFont("Courier", 6))
        painter.setPen(QColor("#585b70"))
        painter.drawText(2, track_top + 1, track_width - 4, mid_y - track_top - 1,
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, "R")
        painter.drawText(2, mid_y + 1, track_width - 4, track_bottom - mid_y - 1,
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, "L")

        row_h_right = mid_y - track_top - 1
        row_h_left = track_bottom - mid_y - 1

        def _draw_interval_anns(anns_list, row_top, row_h):
            for ann in anns_list:
                x1 = frame_to_x_static(ann.start_frame)
                x2 = frame_to_x_static(ann.end_frame)
                bar_w = max(x2 - x1, 2)
                is_fail = ann.metadata.get("fail", False)
                is_unlabeled = ann.label_id == UNLABELED_LABEL_ID
                is_idle = ann.label_id == IDLE_LABEL_ID
                is_selected = (
                    self._selected_segment is not None
                    and self._selected_segment.id == ann.id
                )

                if is_idle:
                    # Bloc gris uni discret pour les segments idle
                    color = QColor("#45475a")
                    color.setAlpha(140)
                    painter.fillRect(x1, row_top, bar_w, row_h, color)
                    if is_selected:
                        painter.setPen(QPen(QColor("#f9e2af"), 2))
                    else:
                        painter.setPen(QPen(QColor("#585b70"), 1))
                    painter.drawRect(x1, row_top, bar_w, row_h)
                    if bar_w > 35:
                        painter.setPen(QColor("#6c7086"))
                        painter.setFont(QFont("Courier", 7))
                        painter.drawText(
                            QRect(x1 + 2, row_top, bar_w - 4, row_h),
                            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                            "idle",
                        )
                    continue

                if is_unlabeled:
                    # Draw as hatched grey block
                    color = QColor("#3d3d5c")
                    color.setAlpha(180)
                    painter.fillRect(x1, row_top, bar_w, row_h, color)
                    # Hatching
                    hatch_color = QColor("#5a5a8a")
                    hatch_color.setAlpha(80)
                    painter.setPen(QPen(hatch_color, 1))
                    step = 8
                    for offset in range(-row_h, bar_w + row_h, step):
                        xa = x1 + offset
                        xb = x1 + offset + row_h
                        painter.drawLine(
                            max(xa, x1), row_top,
                            min(xb, x1 + bar_w), row_top + min(xb - xa, row_h),
                        )
                    if is_selected:
                        sel_border = QColor("#f9e2af")  # yellow highlight
                        painter.setPen(QPen(sel_border, 2))
                    else:
                        painter.setPen(QPen(QColor("#5a5a8a"), 1))
                    painter.drawRect(x1, row_top, bar_w, row_h)
                    if bar_w > 40:
                        painter.setPen(QColor("#a6adc8"))
                        painter.setFont(QFont("Courier", 7))
                        painter.drawText(
                            QRect(x1 + 2, row_top, bar_w - 4, row_h),
                            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                            "[ ? ]",
                        )
                    continue

                color_hex = self._label_colors.get(ann.label_id, "#6c7086")
                color = QColor(color_hex)
                color.setAlpha(160)
                painter.fillRect(x1, row_top, bar_w, row_h, color)
                if is_fail:
                    fail_color = QColor("#f38ba8")
                    fail_color.setAlpha(120)
                    pen = QPen(fail_color, 1)
                    painter.setPen(pen)
                    step = 5
                    for offset in range(-row_h, bar_w + row_h, step):
                        x_a = x1 + offset
                        x_b = x1 + offset + row_h
                        painter.drawLine(
                            max(x_a, x1), row_top,
                            min(x_b, x1 + bar_w), row_top + min(x_b - x_a, row_h),
                        )
                if is_selected:
                    border_color = QColor("#f9e2af")  # yellow highlight when selected
                    painter.setPen(QPen(border_color, 2))
                else:
                    border_color = QColor("#f38ba8") if is_fail else QColor(color_hex)
                    painter.setPen(QPen(border_color, 1 if not is_fail else 2))
                painter.drawRect(x1, row_top, bar_w, row_h)
                if bar_w > 40:
                    painter.setPen(QColor("#ffffff"))
                    painter.setFont(QFont("Courier", 7))
                    label_text = ("[fail] " if is_fail else "") + ann.label_name
                    painter.drawText(
                        QRect(x1 + 2, row_top, bar_w - 4, row_h),
                        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                        label_text,
                    )

        interval_anns = [a for a in self._annotations if a.annotation_type == LabelType.INTERVAL]
        right_interval = [a for a in interval_anns if a.metadata.get("hand") != "left"]
        left_interval = [a for a in interval_anns if a.metadata.get("hand") == "left"]
        _draw_interval_anns(right_interval, track_top, row_h_right)
        _draw_interval_anns(left_interval, mid_y + 1, row_h_left)

        frame_anns = [a for a in self._annotations if a.annotation_type == LabelType.FRAME]
        for ann in frame_anns:
            x = frame_to_x_static(ann.frame_index)
            color = QColor(self._label_colors.get(ann.label_id, "#6c7086"))
            painter.setPen(QPen(color, 2))
            if ann.metadata.get("hand") == "left":
                painter.drawLine(x, mid_y + 1, x, track_bottom)
            else:
                painter.drawLine(x, track_top, x, mid_y - 1)

        # In/Out markers
        if self.in_point is not None:
            x_in = frame_to_x_static(self.in_point)
            painter.setPen(QPen(QColor("#a6e3a1"), 2))
            painter.drawLine(x_in, track_top, x_in, track_bottom)
            painter.setFont(QFont("Courier", 7, QFont.Weight.Bold))
            painter.drawText(x_in + 3, track_top + 10, "IN")

        if self.out_point is not None:
            x_out = frame_to_x_static(self.out_point)
            painter.setPen(QPen(QColor("#f38ba8"), 2))
            painter.drawLine(x_out, track_top, x_out, track_bottom)
            painter.setFont(QFont("Courier", 7, QFont.Weight.Bold))
            painter.drawText(x_out + 3, track_top + 10, "OUT")

        if self.in_point is not None and self.out_point is not None:
            x1 = frame_to_x_static(min(self.in_point, self.out_point))
            x2 = frame_to_x_static(max(self.in_point, self.out_point))
            sel_color = QColor("#89b4fa")
            sel_color.setAlpha(30)
            painter.fillRect(x1, track_top, x2 - x1, track_h, sel_color)

        painter.end()
        return pixmap

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def _draw_crop_cursor(self, painter: QPainter, x: int, h: int, is_start: bool) -> None:
        """Draw a crop cursor handle (orange triangle + vertical line)."""
        color = QColor("#fab387")  # Catppuccin peach/orange
        painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
        painter.drawLine(x, 0, x, h)

        # Triangle handle at top, pointing toward the kept region
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        size = 7
        if is_start:
            # Triangle pointing right
            pts = [QPointF(x, 0), QPointF(x, size * 1.4), QPointF(x + size, 0)]
        else:
            # Triangle pointing left
            pts = [QPointF(x, 0), QPointF(x, size * 1.4), QPointF(x - size, 0)]
        painter.drawPolygon(pts)

        # Small label
        painter.setPen(color)
        painter.setFont(QFont("Courier", 6, QFont.Weight.Bold))
        label = "◀" if is_start else "▶"
        offset = 4 if is_start else -10
        painter.drawText(x + offset, size * 2, label)

    def paintEvent(self, event) -> None:
        w = self.width()
        h = self.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.frame_count <= 0:
            painter.fillRect(0, 0, w, h, QColor("#181825"))
            painter.setPen(QColor("#6c7086"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data loaded")
            painter.end()
            return

        # --- Static layer ---
        if self._static_cache is None or self._static_cache.width() != w or self._static_cache.height() != h:
            self._static_cache = self._render_static(w, h)

        painter.drawPixmap(0, 0, self._static_cache)

        track_top = 18  # ruler_h + 2
        track_bottom = self._annotation_track_bottom()

        # --- Hover frame indicator ---
        if 0 <= self._hover_frame < self.frame_count:
            x_hover = self._frame_to_x(self._hover_frame)
            painter.setPen(QPen(QColor(255, 255, 255, 60), 1, Qt.PenStyle.DotLine))
            painter.drawLine(x_hover, track_top, x_hover, track_bottom)

        # --- Crop cursors (drawn before playhead so playhead is always on top) ---
        x_cs = self._frame_to_x(self.crop_start)
        x_ce = self._frame_to_x(self.crop_end)
        self._draw_crop_cursor(painter, x_cs, h, is_start=True)
        self._draw_crop_cursor(painter, x_ce, h, is_start=False)

        # --- Current frame playhead ---
        x_current = self._frame_to_x(self.current_frame)
        painter.setPen(QPen(QColor("#f38ba8"), 2))
        painter.drawLine(x_current, 0, x_current, h)
        painter.setBrush(QBrush(QColor("#f38ba8")))
        painter.setPen(Qt.PenStyle.NoPen)
        triangle = [
            QPointF(x_current - 5, 0),
            QPointF(x_current + 5, 0),
            QPointF(x_current, 8),
        ]
        painter.drawPolygon(triangle)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _pick_drag_target(self, x: int) -> str:
        """Return which element is being grabbed at pixel x."""
        x_cs = self._frame_to_x(self.crop_start)
        x_ce = self._frame_to_x(self.crop_end)
        if abs(x - x_cs) <= _CROP_GRAB_PX:
            return "crop_start"
        if abs(x - x_ce) <= _CROP_GRAB_PX:
            return "crop_end"
        return "playhead"

    def _find_annotation_at(self, x: int, y: int) -> Optional[Annotation]:
        """Find annotation at given pixel position.

        Args:
            x: Horizontal position
            y: Vertical position

        Returns:
            Annotation object if found, None otherwise
        """
        frame = self._frame_at(x)

        # Timeline layout
        ruler_h = 16
        track_top = ruler_h + 2
        track_bottom = self._annotation_track_bottom()
        track_h = track_bottom - track_top
        mid_y = track_top + track_h // 2

        # Check if y is in the annotation area
        if y < track_top or y >= track_bottom:
            return None

        # Determine which hand based on y position
        in_right_track = y < mid_y
        target_hand = None if in_right_track else "left"

        # Find annotations at this frame
        for ann in self._annotations:
            # Check hand match
            ann_hand = ann.metadata.get("hand")
            if target_hand is None:  # right track
                if ann_hand == "left":
                    continue
            else:  # left track
                if ann_hand != "left":
                    continue

            # Check frame range
            if ann.annotation_type == LabelType.INTERVAL:
                if ann.start_frame <= frame <= ann.end_frame:
                    return ann
            elif ann.annotation_type == LabelType.FRAME:
                # Allow some tolerance (±2 pixels)
                ann_x = self._frame_to_x(ann.frame_index)
                if abs(x - ann_x) <= 3:
                    return ann

        return None

    def mousePressEvent(self, event) -> None:
        if self.frame_count <= 0:
            return
        x = event.pos().x()
        y = event.pos().y()

        # Right click = edit annotation
        if event.button() == Qt.MouseButton.RightButton:
            annotation = self._find_annotation_at(x, y)
            if annotation is not None:
                self.annotation_clicked.emit(annotation)
                return

        # Left click inside annotation area = select segment for labeling (without blocking drag)
        ruler_h = 16
        track_top = ruler_h + 2
        track_bottom = self._annotation_track_bottom()
        if track_top <= y < track_bottom:
            annotation = self._find_annotation_at(x, y)
            if annotation is not None and annotation.annotation_type == LabelType.INTERVAL:
                self._selected_segment = annotation
                self._static_cache = None
                self.segment_selected.emit(annotation)
                # Fall through — still move playhead to clicked position and allow drag

        # Left click = normal drag behavior (playhead / crop cursors)
        self._drag_target = self._pick_drag_target(x)
        self._dragging = True

        if self._drag_target == "playhead":
            frame = self._frame_at(x)
            self.current_frame = frame
            self.update()
            self.frame_selected.emit(frame)
        else:
            self._apply_crop_drag(x)

    def mouseMoveEvent(self, event) -> None:
        if self.frame_count <= 0:
            return
        x = event.pos().x()

        # Update hover
        self._hover_frame = int(x / self.width() * self.frame_count)
        self._hover_frame = max(0, min(self._hover_frame, self.frame_count - 1))

        # Update cursor shape based on proximity
        target = self._pick_drag_target(x) if not self._dragging else self._drag_target
        if target in ("crop_start", "crop_end"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if self._dragging:
            if self._drag_target == "playhead":
                self._schedule_drag(x)
            else:
                self._apply_crop_drag(x)
        else:
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False
        if self._drag_target == "playhead" and self._pending_drag_frame is not None:
            self._flush_drag()
        elif self._drag_target in ("crop_start", "crop_end"):
            # Force final cache update when releasing crop cursor
            self._crop_cache_timer.stop()
            self._invalidate_crop_cache()
        self._drag_target = "playhead"

    def leaveEvent(self, event) -> None:
        self._hover_frame = -1
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def resizeEvent(self, event) -> None:
        self._static_cache = None
        super().resizeEvent(event)

    def _apply_crop_drag(self, x: int) -> None:
        frame = self._frame_at(x)
        if self._drag_target == "crop_start":
            self.crop_start = min(frame, self.crop_end - 1)
        else:
            self.crop_end = max(frame, self.crop_start + 1)

        # Schedule cache invalidation (throttled for smooth dragging)
        self._crop_cache_needs_update = True
        if not self._crop_cache_timer.isActive():
            self._crop_cache_timer.start()

        # Update immediately for visual feedback
        self.update()
        self.crop_changed.emit(self.crop_start, self.crop_end)

    def _invalidate_crop_cache(self) -> None:
        """Invalidate static cache after crop drag (throttled)."""
        if self._crop_cache_needs_update:
            self._static_cache = None
            self._crop_cache_needs_update = False
            self.update()

    def _schedule_drag(self, x: int) -> None:
        self._pending_drag_frame = self._frame_at(x)
        self.current_frame = self._pending_drag_frame
        self.update()
        if not self._drag_throttle.isActive():
            self._drag_throttle.start()

    def _flush_drag(self) -> None:
        if self._pending_drag_frame is not None:
            self.frame_selected.emit(self._pending_drag_frame)
            self._pending_drag_frame = None

    @staticmethod
    def _nice_tick_interval(duration: float, width_px: int) -> float:
        target_ticks = max(width_px // 80, 2)
        raw = duration / target_ticks
        for nice in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60]:
            if raw <= nice:
                return nice
        return 60.0


class AnnotationTimeline(QWidget):
    """Full timeline widget with controls, annotation bar, and in/out selection."""

    frame_changed = pyqtSignal(int)
    play_toggled = pyqtSignal(bool)
    in_point_set = pyqtSignal(int)
    out_point_set = pyqtSignal(int)
    crop_changed = pyqtSignal(int, int)  # crop_start_frame, crop_end_frame
    annotation_clicked = pyqtSignal(object)  # Forward annotation clicks
    segment_selected = pyqtSignal(object)    # Forward segment selection

    def __init__(self, label_manager: LabelManager = None, parent=None):
        super().__init__(parent)
        self.label_manager = label_manager
        self.frame_count = 0
        self.current_frame = 0
        self.fps = 30.0
        self.is_playing = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(2)

        # Timeline bar (fixed height)
        self.timeline_bar = AnnotationTimelineBar()
        self.timeline_bar.setFixedHeight(80)
        self.timeline_bar.frame_selected.connect(self._on_bar_frame_selected)
        self.timeline_bar.crop_changed.connect(self._on_crop_changed)
        self.timeline_bar.annotation_clicked.connect(self.annotation_clicked.emit)
        self.timeline_bar.segment_selected.connect(self.segment_selected.emit)
        layout.addWidget(self.timeline_bar)

        # Gripper graph panel (height auto-géré par le widget)
        self.gripper_widget = GripperGraphWidget()
        layout.addWidget(self.gripper_widget)

        # Controls row (below timeline)
        controls = QHBoxLayout()
        controls.setSpacing(4)  # Reduced spacing between buttons

        # Play/Pause
        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(60)
        self.play_btn.setStyleSheet(
            "QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; "
            "border-radius: 3px; padding: 3px; font-size: 11px; } "
            "QPushButton:hover { background: #45475a; }"
        )
        self.play_btn.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_btn)

        # Previous frame
        prev_btn = QToolButton()
        prev_btn.setText("<")
        prev_btn.setFixedSize(28, 24)
        prev_btn.setStyleSheet("QToolButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 3px; }")
        prev_btn.clicked.connect(lambda: self._step_frame(-1))
        controls.addWidget(prev_btn)

        # Next frame
        next_btn = QToolButton()
        next_btn.setText(">")
        next_btn.setFixedSize(28, 24)
        next_btn.setStyleSheet("QToolButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 3px; }")
        next_btn.clicked.connect(lambda: self._step_frame(1))
        controls.addWidget(next_btn)

        # Skip backward 10 frames
        skip_back_btn = QToolButton()
        skip_back_btn.setText("<<")
        skip_back_btn.setFixedSize(28, 24)
        skip_back_btn.setStyleSheet("QToolButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 3px; }")
        skip_back_btn.clicked.connect(lambda: self._step_frame(-10))
        controls.addWidget(skip_back_btn)

        # Skip forward 10 frames
        skip_fwd_btn = QToolButton()
        skip_fwd_btn.setText(">>")
        skip_fwd_btn.setFixedSize(28, 24)
        skip_fwd_btn.setStyleSheet("QToolButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 3px; }")
        skip_fwd_btn.clicked.connect(lambda: self._step_frame(10))
        controls.addWidget(skip_fwd_btn)

        controls.addSpacing(12)

        # In/Out buttons
        self.in_btn = QPushButton("Set IN [I]")
        self.in_btn.setFixedWidth(72)
        self.in_btn.setStyleSheet(
            "QPushButton { background: #1e3a2e; color: #a6e3a1; border: 1px solid #a6e3a1; "
            "border-radius: 3px; padding: 3px; font-size: 10px; } "
            "QPushButton:hover { background: #2e5a3e; }"
        )
        self.in_btn.clicked.connect(self._set_in_point)
        controls.addWidget(self.in_btn)

        self.out_btn = QPushButton("Set OUT [O]")
        self.out_btn.setFixedWidth(72)
        self.out_btn.setStyleSheet(
            "QPushButton { background: #3a1e2e; color: #f38ba8; border: 1px solid #f38ba8; "
            "border-radius: 3px; padding: 3px; font-size: 10px; } "
            "QPushButton:hover { background: #5a2e3e; }"
        )
        self.out_btn.clicked.connect(self._set_out_point)
        controls.addWidget(self.out_btn)

        self.clear_io_btn = QPushButton("Clear")
        self.clear_io_btn.setFixedWidth(48)
        self.clear_io_btn.setStyleSheet(
            "QPushButton { background: #313244; color: #6c7086; border: 1px solid #45475a; "
            "border-radius: 3px; padding: 3px; font-size: 10px; } "
            "QPushButton:hover { background: #45475a; }"
        )
        self.clear_io_btn.clicked.connect(self._clear_in_out)
        controls.addWidget(self.clear_io_btn)

        controls.addStretch()

        # Crop info label (shown only when crop is active)
        self.crop_label = QLabel()
        self.crop_label.setStyleSheet("color: #fab387; font-family: Courier; font-size: 10px;")
        self.crop_label.setVisible(False)
        controls.addWidget(self.crop_label)

        # Frame info
        self.frame_label = QLabel("Frame: 0 / 0  |  0.000s")
        self.frame_label.setStyleSheet("color: #a6adc8; font-family: Courier; font-size: 11px;")
        controls.addWidget(self.frame_label)

        layout.addLayout(controls)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame_count(self, count: int) -> None:
        self.frame_count = count
        self.timeline_bar.set_frame_count(count)
        self.gripper_widget.set_frame_count(count)
        self._update_label()
        self._update_crop_label()

    def set_fps(self, fps: float) -> None:
        self.fps = fps
        self.timeline_bar.set_fps(fps)
        self.gripper_widget.set_fps(fps)
        self._update_label()

    def set_current_frame(self, frame: int) -> None:
        if frame < 0 or frame >= self.frame_count:
            return
        self.current_frame = frame
        self.timeline_bar.set_current_frame(frame)
        self.gripper_widget.set_current_time(frame / self.fps if self.fps > 0 else 0.0)
        self._update_label()
        self.frame_changed.emit(frame)

    def set_current_frame_silent(self, frame: int) -> None:
        """Move the playhead without emitting frame_changed (used by playback loop)."""
        if frame < 0 or frame >= self.frame_count:
            return
        self.current_frame = frame
        self.timeline_bar.set_current_frame(frame)
        self.gripper_widget.set_current_time(frame / self.fps if self.fps > 0 else 0.0)
        self._update_label()

    def set_gripper_data(self, gripper_data: dict) -> None:
        """Charge les données gripper dans le panneau dédié."""
        self.gripper_widget.set_data(gripper_data)

    def refresh_annotations(self) -> None:
        """Refresh annotation display from label manager."""
        if self.label_manager is None:
            return
        label_colors = {lid: l.color for lid, l in self.label_manager.labels.items()}
        self.timeline_bar.set_annotations(self.label_manager.annotations, label_colors)

    def toggle_playback(self) -> None:
        self.is_playing = not self.is_playing
        self.play_btn.setText("Pause" if self.is_playing else "Play")
        self.play_toggled.emit(self.is_playing)

    def get_in_out(self) -> Tuple[Optional[int], Optional[int]]:
        return self.timeline_bar.in_point, self.timeline_bar.out_point

    def get_crop(self) -> Optional[Tuple[int, int]]:
        """Return (start_frame, end_frame) if crop is active, else None."""
        if self.timeline_bar.is_crop_active():
            return self.timeline_bar.crop_start, self.timeline_bar.crop_end
        return None

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_bar_frame_selected(self, frame: int) -> None:
        self.set_current_frame(frame)

    def _on_crop_changed(self, start: int, end: int) -> None:
        self._update_crop_label()
        self.crop_changed.emit(start, end)

    def _step_frame(self, delta: int) -> None:
        new_frame = max(0, min(self.current_frame + delta, self.frame_count - 1))
        self.set_current_frame(new_frame)

    def _set_in_point(self) -> None:
        self.timeline_bar.set_in_point(self.current_frame)
        self.in_point_set.emit(self.current_frame)

    def _set_out_point(self) -> None:
        self.timeline_bar.set_out_point(self.current_frame)
        self.out_point_set.emit(self.current_frame)

    def _clear_in_out(self) -> None:
        self.timeline_bar.set_in_point(None)
        self.timeline_bar.set_out_point(None)

    def _update_label(self) -> None:
        t = self.current_frame / self.fps if self.fps > 0 else 0.0
        self.frame_label.setText(
            f"Frame: {self.current_frame} / {self.frame_count}  |  {t:.3f}s"
        )

    def _update_crop_label(self) -> None:
        crop = self.get_crop()
        if crop is None:
            self.crop_label.setVisible(False)
        else:
            cs, ce = crop
            t_start = cs / self.fps if self.fps > 0 else cs
            t_end = ce / self.fps if self.fps > 0 else ce
            self.crop_label.setText(f"✂ {t_start:.2f}s → {t_end:.2f}s")
            self.crop_label.setVisible(True)
