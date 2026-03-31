"""Gripper angle graph widget.

Displays pince1 and pince2 angles as time-series plots with a vertical
cursor tracking the current playback position. Designed to sit in a
sidebar panel.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QLabel
from PyQt6.QtCore import Qt
import pyqtgraph as pg


GRIPPER_COLORS = {
    "left":  (255, 200, 50),   # yellow-gold
    "right": (50, 200, 255),   # cyan-blue
    # Legacy numeric keys kept for backwards compatibility
    "1": (255, 200, 50),
    "2": (50, 200, 255),
}


class GripperGraphWidget(QWidget):
    """Displays gripper (pince) angle data as time-series graphs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cursor_lines = []
        self._plots = []
        self._plots_by_gid: dict = {}   # gid -> PlotItem
        self._current_time = 0.0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        pg.setConfigOptions(antialias=True, background="#1e1e2e", foreground="#cdd6f4")

        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.graphics_layout)

    def set_data(self, gripper_data: dict) -> None:
        """Load gripper data for display.

        Args:
            gripper_data: dict gripper_id -> (timestamps, angles)
        """
        self.graphics_layout.clear()
        self._cursor_lines = []
        self._plots = []
        self._plots_by_gid = {}

        if not gripper_data:
            plot = self.graphics_layout.addPlot(row=0, col=0)
            plot.setTitle("No gripper data", color="#585b70")
            self._plots.append(plot)
            return

        row = 0
        for gid, (timestamps, angles) in gripper_data.items():
            if timestamps is None or angles is None:
                continue

            plot = self.graphics_layout.addPlot(row=row, col=0)
            plot.setLabel("left", f"Gripper {gid}", units="mm")
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").setWidth(45)
            plot.getAxis("left").setStyle(tickFont=pg.QtGui.QFont("Courier", 8))
            plot.getAxis("bottom").setStyle(tickFont=pg.QtGui.QFont("Courier", 8))

            if self._plots:
                plot.setXLink(self._plots[0])

            color = GRIPPER_COLORS.get(gid, (200, 200, 200))
            plot.plot(
                timestamps,
                angles,
                pen=pg.mkPen(color=color, width=1.5),
                name=f"Gripper {gid}",
            )

            cursor = pg.InfiniteLine(
                pos=0, angle=90,
                pen=pg.mkPen("r", width=1.5, style=Qt.PenStyle.DashLine),
            )
            plot.addItem(cursor)
            self._cursor_lines.append(cursor)
            self._plots.append(plot)
            self._plots_by_gid[gid] = plot

            # Only show x-axis label on last plot
            if row == 0 and len(gripper_data) > 1:
                plot.getAxis("bottom").setStyle(showValues=False)

            row += 1

        # Add time label to last plot
        if self._plots:
            self._plots[-1].setLabel("bottom", "Time", units="s")

    def set_current_time(self, t: float) -> None:
        """Move cursor to given time."""
        self._current_time = t
        for cursor in self._cursor_lines:
            cursor.setValue(t)

    def show_nan_warnings(self, warnings: dict) -> None:
        """Affiche un label d'avertissement NaN sur les plots concernés.

        Args:
            warnings: dict gid -> message (ex: {"1": "Pince 1 : 42 NaN (3.1%)"})
        """
        for gid, msg in warnings.items():
            plot = self._plots_by_gid.get(gid)
            if plot is None:
                continue
            # Titre en orange pour signaler le problème
            label = gid.capitalize()
            plot.setTitle(f"⚠ Gripper {label} — {msg}", color="#fab387", size="8pt")
