"""Timeline widget for video navigation and annotation display."""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush
from typing import List


class TimelineWidget(QWidget):
    """Timeline widget with playback controls and annotation display."""

    frame_changed = pyqtSignal(int)  # Frame index
    play_toggled = pyqtSignal(bool)  # Is playing

    def __init__(self, parent=None):
        """Initialize timeline widget."""
        super().__init__(parent)

        self.frame_count = 0
        self.current_frame = 0
        self.fps = 30.0
        self.is_playing = False
        self.annotations_at_frame = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup UI layout."""
        layout = QVBoxLayout(self)

        # Timeline bar (custom painted)
        self.timeline_bar = TimelineBar()
        self.timeline_bar.setMinimumHeight(60)
        self.timeline_bar.frame_selected.connect(self._on_timeline_click)
        layout.addWidget(self.timeline_bar)

        # Controls
        controls_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        controls_layout.addWidget(self.frame_slider, stretch=1)

        # Frame info label
        self.frame_label = QLabel("Frame: 0 / 0 (0.00s)")
        controls_layout.addWidget(self.frame_label)

        layout.addLayout(controls_layout)

    def set_frame_count(self, frame_count: int) -> None:
        """Set total number of frames.

        Args:
            frame_count: Total frame count
        """
        self.frame_count = frame_count
        self.frame_slider.setMaximum(max(0, frame_count - 1))
        self.timeline_bar.set_frame_count(frame_count)
        self._update_frame_label()

    def set_fps(self, fps: float) -> None:
        """Set video framerate.

        Args:
            fps: Frames per second
        """
        self.fps = fps
        self._update_frame_label()

    def set_current_frame(self, frame_index: int) -> None:
        """Set current frame.

        Args:
            frame_index: Frame index
        """
        if frame_index < 0 or frame_index >= self.frame_count:
            return

        self.current_frame = frame_index
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_index)
        self.frame_slider.blockSignals(False)

        self.timeline_bar.set_current_frame(frame_index)
        self._update_frame_label()

        self.frame_changed.emit(frame_index)

    def set_annotations_at_frame(self, annotations: List) -> None:
        """Set annotations active at current frame.

        Args:
            annotations: List of annotations
        """
        self.annotations_at_frame = annotations

    def add_annotation_marker(self, frame_index: int, label_name: str, color: str) -> None:
        """Add annotation marker to timeline.

        Args:
            frame_index: Frame index
            label_name: Label name
            color: Color in hex format
        """
        self.timeline_bar.add_marker(frame_index, label_name, color)

    def clear_markers(self) -> None:
        """Clear all annotation markers."""
        self.timeline_bar.clear_markers()

    def toggle_playback(self) -> None:
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        self.play_button.setText("Pause" if self.is_playing else "Play")
        self.play_toggled.emit(self.is_playing)

    def _on_slider_changed(self, value: int) -> None:
        """Handle slider value change."""
        self.set_current_frame(value)

    def _on_timeline_click(self, frame_index: int) -> None:
        """Handle click on timeline bar."""
        self.set_current_frame(frame_index)

    def _update_frame_label(self) -> None:
        """Update frame info label."""
        timestamp = self.current_frame / self.fps if self.fps > 0 else 0.0
        self.frame_label.setText(
            f"Frame: {self.current_frame} / {self.frame_count} ({timestamp:.2f}s)"
        )


class TimelineBar(QWidget):
    """Custom painted timeline bar with annotation markers."""

    frame_selected = pyqtSignal(int)  # Frame index

    def __init__(self, parent=None):
        """Initialize timeline bar."""
        super().__init__(parent)

        self.frame_count = 0
        self.current_frame = 0
        self.markers = {}  # frame_index -> (label_name, color)

        self.setMinimumHeight(60)
        self.setMouseTracking(True)

    def set_frame_count(self, frame_count: int) -> None:
        """Set total frame count."""
        self.frame_count = frame_count
        self.update()

    def set_current_frame(self, frame_index: int) -> None:
        """Set current frame."""
        self.current_frame = frame_index
        self.update()

    def add_marker(self, frame_index: int, label_name: str, color: str) -> None:
        """Add annotation marker."""
        self.markers[frame_index] = (label_name, color)
        self.update()

    def clear_markers(self) -> None:
        """Clear all markers."""
        self.markers.clear()
        self.update()

    def paintEvent(self, event) -> None:
        """Paint timeline bar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QColor(40, 40, 40))

        if self.frame_count == 0:
            return

        # Margins: space at start and end of timeline
        margin = 20  # pixels on each side
        usable_width = width - 2 * margin

        # Timeline track
        track_y = height // 2
        track_height = 4
        painter.fillRect(margin, track_y - track_height // 2, usable_width, track_height, QColor(80, 80, 80))

        # Draw markers
        for frame_idx, (label_name, color_hex) in self.markers.items():
            # Map frame to position: frame 0 -> margin, frame (count-1) -> width-margin
            x = margin + int(frame_idx * usable_width / max(1, self.frame_count - 1))
            color = QColor(color_hex)
            painter.setPen(QPen(color, 2))
            painter.drawLine(x, 0, x, height)

        # Current frame indicator
        current_x = margin + int(self.current_frame * usable_width / max(1, self.frame_count - 1))
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawLine(current_x, 0, current_x, height)

        # Current frame handle
        handle_size = 10
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.drawEllipse(
            current_x - handle_size // 2,
            track_y - handle_size // 2,
            handle_size,
            handle_size
        )

    def mousePressEvent(self, event) -> None:
        """Handle mouse press."""
        if self.frame_count == 0:
            return

        margin = 20  # Must match paintEvent margin
        usable_width = self.width() - 2 * margin

        x = event.pos().x()
        # Clamp x to usable area
        x_clamped = max(margin, min(x, self.width() - margin))

        # Map position to frame: margin -> frame 0, width-margin -> frame (count-1)
        frame_index = int((x_clamped - margin) * max(1, self.frame_count - 1) / usable_width)
        frame_index = max(0, min(frame_index, self.frame_count - 1))

        self.frame_selected.emit(frame_index)
