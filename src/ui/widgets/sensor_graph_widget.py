"""Sensor data graph widget.

Displays synchronized tracker positions and gripper angles as time-series
plots with a vertical cursor line tracking the current playback position.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt
import pyqtgraph as pg


# Colors for different trackers/signals
TRACKER_COLORS = {
    "1": [(255, 80, 80), (80, 255, 80), (80, 80, 255)],     # R G B for X Y Z
    "2": [(255, 160, 80), (80, 255, 160), (160, 80, 255)],
    "3": [(255, 80, 160), (160, 255, 80), (80, 160, 255)],
}
GRIPPER_COLORS = {
    "1": (255, 200, 50),
    "2": (50, 200, 255),
}


class SensorGraphWidget(QWidget):
    """Displays tracker and gripper sensor data as time-series graphs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_time = 0.0
        self._cursor_lines = []
        self._plots = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # pyqtgraph styling
        pg.setConfigOptions(antialias=True, background="#1e1e2e", foreground="#cdd6f4")

        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.graphics_layout)

    def set_data(
        self,
        tracker_timestamps: np.ndarray = None,
        tracker_positions: dict = None,
        gripper_data: dict = None,
    ) -> None:
        """Load all sensor data for display.

        Args:
            tracker_timestamps: (N,) array of tracker sample times
            tracker_positions: dict tracker_id -> (N, 3) position array
            gripper_data: dict gripper_id -> (timestamps, angles)
        """
        self.graphics_layout.clear()
        self._cursor_lines = []
        self._plots = []

        row = 0

        # --- Tracker position plots (one row per axis: X, Y, Z) ---
        if tracker_timestamps is not None and tracker_positions:
            axis_names = ["X", "Y", "Z"]
            for axis_idx, axis_name in enumerate(axis_names):
                plot = self.graphics_layout.addPlot(row=row, col=0)
                plot.setLabel("left", f"Pos {axis_name}", units="m")
                plot.showGrid(x=True, y=True, alpha=0.15)
                plot.setXLink(self._plots[0] if self._plots else None)
                plot.getAxis("left").setWidth(50)
                plot.getAxis("left").setStyle(tickFont=pg.QtGui.QFont("Courier", 8))
                plot.getAxis("bottom").setStyle(tickFont=pg.QtGui.QFont("Courier", 8))

                for tid, positions in tracker_positions.items():
                    colors = TRACKER_COLORS.get(tid, [(200, 200, 200)] * 3)
                    color = colors[axis_idx]
                    plot.plot(
                        tracker_timestamps,
                        positions[:, axis_idx],
                        pen=pg.mkPen(color=color, width=1.5),
                        name=f"T{tid}",
                    )

                # Cursor line
                cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("r", width=1.5, style=Qt.PenStyle.DashLine))
                plot.addItem(cursor)
                self._cursor_lines.append(cursor)
                self._plots.append(plot)

                # Hide x-axis labels for all but last plot
                if axis_idx < 2:
                    plot.getAxis("bottom").setStyle(showValues=False)

                row += 1

        # --- Gripper angle plots ---
        if gripper_data:
            plot = self.graphics_layout.addPlot(row=row, col=0)
            plot.setLabel("left", "Grip", units="deg")
            plot.setLabel("bottom", "Time", units="s")
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").setWidth(50)
            plot.getAxis("left").setStyle(tickFont=pg.QtGui.QFont("Courier", 8))
            plot.getAxis("bottom").setStyle(tickFont=pg.QtGui.QFont("Courier", 8))

            if self._plots:
                plot.setXLink(self._plots[0])

            for gid, (timestamps, angles) in gripper_data.items():
                if timestamps is not None and angles is not None:
                    color = GRIPPER_COLORS.get(gid, (200, 200, 200))
                    plot.plot(
                        timestamps,
                        angles,
                        pen=pg.mkPen(color=color, width=1.5),
                        name=f"Grip {gid}",
                    )

            cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("r", width=1.5, style=Qt.PenStyle.DashLine))
            plot.addItem(cursor)
            self._cursor_lines.append(cursor)
            self._plots.append(plot)
            row += 1

        # If we have no data at all, show placeholder
        if row == 0:
            plot = self.graphics_layout.addPlot(row=0, col=0)
            plot.setTitle("No sensor data")
            self._plots.append(plot)

    def set_current_time(self, t: float) -> None:
        """Move the cursor line to the given time.

        Args:
            t: Time in seconds since session start
        """
        self._current_time = t
        for cursor in self._cursor_lines:
            cursor.setValue(t)
