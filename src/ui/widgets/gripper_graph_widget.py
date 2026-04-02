"""Gripper angle graph widget — style Rerun/ultra-précis.

Affiche les angles left/right des grippers comme des séries temporelles
avec :
  - Courbe épaisse + remplissage gradient translucide sous la courbe
  - Curseur vertical rouge proéminent avec label de valeur instantanée
  - Overlay min / max / current en coin supérieur droit
  - Grille fine, axes typographiés, background sombre cohérent
"""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
import pyqtgraph as pg
from pyqtgraph import mkPen, mkBrush


# ---------------------------------------------------------------------------
# Palette (Catppuccin Mocha)
# ---------------------------------------------------------------------------

GRIPPER_COLORS = {
    "left":  (255, 200,  50),   # gold
    "right": ( 50, 200, 255),   # cyan
    # Legacy keys
    "1":     (255, 200,  50),
    "2":     ( 50, 200, 255),
}

_BG       = "#1e1e2e"
_FG       = "#cdd6f4"
_GRID_CLR = "#313244"
_AXIS_CLR = "#585b70"


# ---------------------------------------------------------------------------
# Per-gripper plot panel
# ---------------------------------------------------------------------------

class _GripperPlot:
    """Un panneau pyqtgraph pour un seul gripper."""

    def __init__(self, layout: pg.GraphicsLayout, row: int, gid: str,
                 timestamps: np.ndarray, angles: np.ndarray,
                 link_to: "pg.PlotItem | None" = None,
                 x_range: "tuple | None" = None):
        self.gid = gid
        self.timestamps = timestamps
        self.angles = angles

        color = GRIPPER_COLORS.get(gid, (200, 200, 200))
        r, g, b = color

        # ── Plot ──────────────────────────────────────────────────────────
        self.plot: pg.PlotItem = layout.addPlot(row=row, col=0)

        # Axes style
        for axis_name in ("left", "bottom", "top", "right"):
            ax = self.plot.getAxis(axis_name)
            ax.setPen(pg.mkPen(_GRID_CLR))
            ax.setTextPen(pg.mkPen(_FG))
            ax.setStyle(tickFont=pg.QtGui.QFont("Menlo", 8))

        self.plot.getAxis("left").setWidth(52)
        self.plot.showGrid(x=True, y=True, alpha=0.18)

        # Lien X entre tous les plots
        if link_to is not None:
            self.plot.setXLink(link_to)

        # ── Courbe principale ─────────────────────────────────────────────
        pen_main = mkPen(color=(r, g, b), width=2.0)
        self._curve = self.plot.plot(
            timestamps, angles,
            pen=pen_main,
            antialias=True,
            name=f"Gripper {gid}",
        )

        # ── Remplissage sous la courbe ────────────────────────────────────
        fill_color = (r, g, b, 35)
        self._fill = pg.FillBetweenItem(
            self._curve,
            pg.PlotDataItem([timestamps[0], timestamps[-1]], [0.0, 0.0],
                            pen=mkPen(None)),
            brush=mkBrush(*fill_color),
        )
        self.plot.addItem(self._fill)

        # ── Label côté gauche ─────────────────────────────────────────────
        side = "Left" if gid in ("left", "1") else "Right"
        label_html = (
            f'<span style="color:rgb({r},{g},{b});font-size:9pt;'
            f'font-family:Menlo">'
            f'<b>{side}</b></span>'
        )
        self.plot.setLabel("left", label_html)
        self.plot.getAxis("left").enableAutoSIPrefix(False)

        # Plage Y : petite marge
        valid = angles[np.isfinite(angles)]
        if valid.size:
            a_min, a_max = float(valid.min()), float(valid.max())
            margin = max((a_max - a_min) * 0.12, 0.5)
            self.plot.setYRange(a_min - margin, a_max + margin, padding=0)
        else:
            self.plot.setYRange(0, 1, padding=0)

        # Plage X : toute la durée de la session
        if x_range is not None:
            self.plot.setXRange(x_range[0], x_range[1], padding=0)
        self.plot.setLimits(xMin=x_range[0] if x_range else None,
                            xMax=x_range[1] if x_range else None)

        # ── Curseur vertical ─────────────────────────────────────────────
        self.cursor = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=mkPen("#f38ba8", width=1.5, style=Qt.PenStyle.SolidLine),
            movable=False,
        )
        self.plot.addItem(self.cursor)

        # ── Point mobile sur la courbe ────────────────────────────────────
        self._dot = pg.ScatterPlotItem(
            [0.0], [0.0],
            symbol="o",
            size=7,
            pen=mkPen("#f38ba8", width=1.5),
            brush=mkBrush(r, g, b, 220),
        )
        self.plot.addItem(self._dot)

        # ── Overlay texte : valeur courante + min/max ─────────────────────
        self._value_label = pg.TextItem(
            text="",
            anchor=(1.0, 0.0),
            color=(r, g, b),
        )
        self._value_label.setFont(pg.QtGui.QFont("Menlo", 9))
        self.plot.addItem(self._value_label)

        # Pré-calcul pour interpolation
        self._ts = timestamps
        self._angles = angles
        self._a_min = float(valid.min()) if valid.size else 0.0
        self._a_max = float(valid.max()) if valid.size else 1.0

        self._update_overlay_pos()

    # ------------------------------------------------------------------
    def _update_overlay_pos(self) -> None:
        """Place le label en haut à droite du plot."""
        vr = self.plot.viewRange()
        x_max = vr[0][1]
        y_max = vr[1][1]
        self._value_label.setPos(x_max, y_max)

    def set_time(self, t: float) -> None:
        self.cursor.setValue(t)

        # Interpoler la valeur
        if self._ts is not None and len(self._ts) > 1:
            val = float(np.interp(t, self._ts, self._angles))
            if np.isfinite(val):
                self._dot.setData([t], [val])
                self._dot.setVisible(True)

                self._update_overlay_pos()
                vr = self.plot.viewRange()
                x_max = vr[0][1]
                y_max = vr[1][1]
                self._value_label.setPos(x_max, y_max)

                unit = "mm"
                self._value_label.setText(
                    f" {val:+.2f} {unit}  \n"
                    f" min {self._a_min:.1f}  max {self._a_max:.1f}  "
                )
            else:
                self._dot.setVisible(False)
                self._value_label.setText("")


# ---------------------------------------------------------------------------
# Widget public
# ---------------------------------------------------------------------------

class GripperGraphWidget(QWidget):
    """Affiche les données gripper (angle/ouverture) sous forme de
    graphiques de séries temporelles haute fidélité (style Rerun)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._panels: list[_GripperPlot] = []
        self._current_time = 0.0
        self._setup_ui()

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        pg.setConfigOptions(antialias=True, background=_BG, foreground=_FG)

        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground(_BG)
        self.graphics_layout.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        # Espacement vertical entre les plots
        self.graphics_layout.ci.layout.setSpacing(4)
        layout.addWidget(self.graphics_layout)

    # ------------------------------------------------------------------
    def set_data(self, gripper_data: dict) -> None:
        """Charge les données gripper.

        Args:
            gripper_data: dict gid -> (timestamps np.ndarray, angles np.ndarray)
        """
        self.graphics_layout.clear()
        self._panels = []

        if not gripper_data:
            p = self.graphics_layout.addPlot(row=0, col=0)
            p.setTitle("No gripper data", color=_AXIS_CLR)
            return

        first_plot = None
        for row, (gid, (timestamps, angles)) in enumerate(gripper_data.items()):
            if timestamps is None or angles is None:
                continue
            ts = np.asarray(timestamps, dtype=float)
            ang = np.asarray(angles, dtype=float)
            if len(ts) < 2:
                continue

            panel = _GripperPlot(
                self.graphics_layout, row, gid, ts, ang,
                link_to=first_plot,
            )
            if first_plot is None:
                first_plot = panel.plot

            # Masquer l'axe X sur tous sauf le dernier
            if row < len(gripper_data) - 1:
                panel.plot.getAxis("bottom").setStyle(showValues=False)
                panel.plot.getAxis("bottom").setLabel("")
            else:
                panel.plot.setLabel("bottom", "Time", units="s")

            self._panels.append(panel)

        # Appliquer le temps courant immédiatement
        for panel in self._panels:
            panel.set_time(self._current_time)

    # ------------------------------------------------------------------
    def set_current_time(self, t: float) -> None:
        """Déplace le curseur au temps t (secondes)."""
        self._current_time = t
        for panel in self._panels:
            panel.set_time(t)

    # ------------------------------------------------------------------
    def show_nan_warnings(self, warnings: dict) -> None:
        """Affiche un titre d'avertissement NaN sur les plots concernés.

        Args:
            warnings: dict gid -> message
        """
        panels_by_gid = {p.gid: p for p in self._panels}
        for gid, msg in warnings.items():
            panel = panels_by_gid.get(gid)
            if panel is None:
                continue
            panel.plot.setTitle(
                f"⚠ Gripper {gid.capitalize()} — {msg}",
                color="#fab387", size="8pt",
            )
