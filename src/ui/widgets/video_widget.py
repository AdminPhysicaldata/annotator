"""Video display widget."""

import logging

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class VideoWidget(QWidget):
    """Widget for displaying video frames with overlays."""

    frame_clicked = pyqtSignal(int, int)  # x, y coordinates

    def __init__(self, parent=None):
        """Initialize video widget."""
        super().__init__(parent)

        self.current_frame: np.ndarray = None
        self.overlay_enabled = True
        self.overlay_data = {}

        # Setup UI
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        self.label.setMinimumSize(640, 480)
        self.label.mousePressEvent = self._on_mouse_press

        layout.addWidget(self.label)

    def set_frame(self, frame: np.ndarray) -> None:
        """Set current video frame.

        Args:
            frame: Video frame as numpy array (BGR format from OpenCV).
                   None is silently ignored.
        """
        if frame is None:
            return
        try:
            self.current_frame = frame.copy()
            self._update_display()
        except Exception as exc:
            logger.error("set_frame failed: %s", exc)

    def set_overlay_data(self, data: dict) -> None:
        """Set overlay data to display on video."""
        try:
            self.overlay_data = data if isinstance(data, dict) else {}
            self._update_display()
        except Exception as exc:
            logger.error("set_overlay_data failed: %s", exc)

    def enable_overlay(self, enabled: bool) -> None:
        """Enable or disable overlay display."""
        try:
            self.overlay_enabled = bool(enabled)
            self._update_display()
        except Exception as exc:
            logger.error("enable_overlay failed: %s", exc)

    def _update_display(self) -> None:
        """Update display with current frame and overlay."""
        if self.current_frame is None:
            return

        try:
            frame = self.current_frame

            # Normalise frame format to uint8 BGR (H, W, 3)
            if frame.ndim == 2:
                # Grayscale → convert to BGR so the rest of the pipeline is uniform
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 4:
                # BGRA — drop alpha channel
                frame = frame[:, :, :3]

            if frame.ndim != 3 or frame.shape[2] != 3:
                logger.warning("Unexpected frame shape %s — skipping display", frame.shape)
                return

            if frame.dtype != np.uint8:
                # Clamp and convert (handles float32 [0,1] or float32 [0,255])
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.overlay_enabled and self.overlay_data:
                frame_rgb = self._apply_overlay(frame_rgb)

            height, width, channel = frame_rgb.shape
            bytes_per_line = channel * width

            q_image = QImage(
                frame_rgb.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(q_image)

            label_size = self.label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = pixmap.scaled(
                    label_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.label.setPixmap(scaled_pixmap)

        except Exception as exc:
            logger.error("VideoWidget._update_display error: %s", exc)

    def _apply_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Apply overlay information to frame.

        Returns the original frame on any error.
        """
        try:
            frame = frame.copy()
            h, w = frame.shape[:2]

            # Draw position indicators
            if "position_2d" in self.overlay_data:
                try:
                    x, y = self.overlay_data["position_2d"]
                    x, y = int(x), int(y)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                except Exception as exc:
                    logger.debug("Overlay position_2d draw error: %s", exc)

            # Draw bounding boxes
            if "bboxes" in self.overlay_data:
                for bbox in self.overlay_data.get("bboxes", []):
                    try:
                        x1, y1, x2, y2 = bbox["coords"]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        color = bbox.get("color", (0, 255, 0))
                        label_text = str(bbox.get("label", ""))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        if label_text:
                            cv2.putText(
                                frame, label_text,
                                (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2,
                            )
                    except Exception as exc:
                        logger.debug("Overlay bbox draw error: %s", exc)

            # Draw text overlays
            if "text" in self.overlay_data:
                y_offset = 30
                for text_item in self.overlay_data.get("text", []):
                    try:
                        text = str(text_item.get("text", ""))
                        color = text_item.get("color", (255, 255, 255))
                        cv2.putText(
                            frame, text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2,
                        )
                        y_offset += 30
                    except Exception as exc:
                        logger.debug("Overlay text draw error: %s", exc)

        except Exception as exc:
            logger.error("_apply_overlay unexpected error: %s", exc)

        return frame

    def _on_mouse_press(self, event) -> None:
        """Handle mouse press on video."""
        if self.current_frame is None:
            return

        try:
            pos = event.pos()

            pixmap = self.label.pixmap()
            if pixmap is None or pixmap.isNull():
                return

            label_size = self.label.size()
            pixmap_size = pixmap.size()

            if pixmap_size.width() <= 0 or pixmap_size.height() <= 0:
                return

            x_offset = (label_size.width() - pixmap_size.width()) // 2
            y_offset = (label_size.height() - pixmap_size.height()) // 2

            if (
                pos.x() < x_offset
                or pos.x() > x_offset + pixmap_size.width()
                or pos.y() < y_offset
                or pos.y() > y_offset + pixmap_size.height()
            ):
                return

            frame_h, frame_w = self.current_frame.shape[:2]
            frame_x = int((pos.x() - x_offset) * frame_w / pixmap_size.width())
            frame_y = int((pos.y() - y_offset) * frame_h / pixmap_size.height())

            # Clamp to valid frame coordinates
            frame_x = max(0, min(frame_x, frame_w - 1))
            frame_y = max(0, min(frame_y, frame_h - 1))

            self.frame_clicked.emit(frame_x, frame_y)

        except Exception as exc:
            logger.error("VideoWidget mouse press error: %s", exc)

    def resizeEvent(self, event) -> None:
        """Handle widget resize."""
        try:
            super().resizeEvent(event)
            self._update_display()
        except Exception as exc:
            logger.error("VideoWidget resizeEvent error: %s", exc)
