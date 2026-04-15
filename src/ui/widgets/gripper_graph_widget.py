"""Gripper data graph widget.

Affiche les signaux d'ouverture gripper (mm) sous forme de panneaux
empilés, alignés pixel-parfait avec la timeline d'annotation (même
marge gauche/droite de 20 px).

Chaque panneau (_PANEL_H px) contient :
  - Bandeau titre avec nom du gripper et valeur courante
  - Zone de tracé avec fill gradient + ligne
  - Lignes de grille dotées aux valeurs min/mid/max
  - Curseur vertical rouge + dot flottant + label valeur
Dernier panneau suivi d'un ruler temporel (_RULER_H px).
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QLinearGradient,
    QPainterPath, QFont, QPixmap,
)
from typing import Optional, Dict, Tuple

# ---------------------------------------------------------------------------
# Palette (Catppuccin Mocha)
# ---------------------------------------------------------------------------

_BG      = QColor("#1e1e2e")
_BG_ALT  = QColor("#181825")
_GRID    = QColor("#313244")
_FG      = QColor("#cdd6f4")
_CURSOR  = QColor("#f38ba8")

_COLORS: Dict[str, QColor] = {
    "left":  QColor("#22d386"),
    "right": QColor("#f5c542"),
    "1":     QColor("#22d386"),
    "2":     QColor("#f5c542"),
}

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_PANEL_H  = 64   # px par panneau gripper
_TITLE_H  = 16   # bandeau titre en haut de chaque panneau
_RULER_H  = 18   # ruler temporel en bas
_MARGIN   = 20   # marge gauche/droite (aligné avec AnnotationTimelineBar)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nice_interval(duration: float, max_ticks: int = 10) -> float:
    """Retourne un intervalle de tick 'propre' pour un axe temporel."""
    if duration <= 0:
        return 1.0
    raw = duration / max_ticks
    for step in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120):
        if step >= raw:
            return float(step)
    return float(int(raw / 60 + 1) * 60)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class GripperGraphWidget(QWidget):
    """Panneaux empilés de séries temporelles d'ouverture gripper."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._current_time: float = 0.0
        self._fps: float = 30.0
        self._frame_count: int = 0
        self._cache: Optional[QPixmap] = None

        self.setMinimumHeight(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._sync_height()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_height(self) -> None:
        n = len(self._data)
        self.setFixedHeight(n * _PANEL_H + (_RULER_H if n > 0 else 0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, gripper_data: dict) -> None:
        """Charge les données gripper.

        Args:
            gripper_data: dict gid -> (timestamps_s np.ndarray, openings_mm np.ndarray)
        """
        self._data = {}
        if gripper_data:
            for gid, v in gripper_data.items():
                ts, ang = v
                if ts is not None and ang is not None and len(ts) > 1:
                    self._data[gid] = (
                        np.asarray(ts,  dtype=np.float64),
                        np.asarray(ang, dtype=np.float64),
                    )
        self._cache = None
        self._sync_height()
        self.update()

    def set_current_time(self, t: float) -> None:
        """Déplace le curseur au temps t (secondes depuis début de session)."""
        self._current_time = float(t)
        self.update()

    def set_fps(self, fps: float) -> None:
        self._fps = float(fps)
        self._cache = None
        self.update()

    def set_frame_count(self, count: int) -> None:
        self._frame_count = int(count)
        self._cache = None
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helper
    # ------------------------------------------------------------------

    def _t_to_x(self, t: float, w: int) -> float:
        """Convertit un temps (s) en pixel X, aligné avec la timeline."""
        if self._fps <= 0 or self._frame_count <= 1:
            return float(_MARGIN)
        usable = w - 2 * _MARGIN
        frac = min(max(t * self._fps / (self._frame_count - 1), 0.0), 1.0)
        return _MARGIN + frac * usable

    # ------------------------------------------------------------------
    # Static layer — waveforms + grilles (mise en cache)
    # ------------------------------------------------------------------

    def _build_static(self, w: int, h: int) -> QPixmap:
        pix = QPixmap(w, h)
        pix.fill(_BG)
        if not self._data:
            return pix

        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        usable_x = w - 2 * _MARGIN

        for row, (gid, (ts_arr, ang_arr)) in enumerate(self._data.items()):
            color = _COLORS.get(gid, QColor("#cdd6f4"))
            py0 = row * _PANEL_H

            # ── Fond du panneau ──────────────────────────────────────────
            bg = _BG_ALT if row % 2 == 1 else _BG
            p.fillRect(0, py0, w, _PANEL_H, bg)

            # Séparateur haut
            p.setPen(QPen(_GRID, 1))
            p.drawLine(0, py0, w, py0)

            # ── Bandeau titre ─────────────────────────────────────────────
            title_bg = QColor(color)
            title_bg.setAlpha(10)
            p.fillRect(0, py0, w, _TITLE_H, title_bg)

            side = "Left" if gid in ("left", "1") else "Right"
            tc = QColor(color)
            tc.setAlpha(210)
            p.setPen(tc)
            p.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
            p.drawText(_MARGIN, py0, usable_x, _TITLE_H,
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"{side} gripper")

            # ── Zone de tracé ─────────────────────────────────────────────
            plot_top    = py0 + _TITLE_H
            plot_bottom = py0 + _PANEL_H - 2
            plot_h      = plot_bottom - plot_top

            valid = np.isfinite(ang_arr)
            if not valid.any() or len(ts_arr) < 2:
                p.setPen(QColor(_FG).darker(150))
                p.setFont(QFont("Menlo", 8))
                p.drawText(_MARGIN, plot_top, usable_x, plot_h,
                           Qt.AlignmentFlag.AlignCenter, "no data")
                continue

            ts_v  = ts_arr[valid]
            ang_v = ang_arr[valid]
            a_min = float(ang_v.min())
            a_max = float(ang_v.max())
            a_rng = a_max - a_min if a_max > a_min else 1.0

            # ── Lignes de grille + ticks Y ────────────────────────────────
            for val in (a_max, (a_max + a_min) / 2.0, a_min):
                norm = (val - a_min) / a_rng
                gy = int(plot_top + (1.0 - norm) * plot_h)

                grid_c = QColor(_GRID)
                grid_c.setAlpha(180)
                p.setPen(QPen(grid_c, 1, Qt.PenStyle.DotLine))
                p.drawLine(_MARGIN, gy, w - _MARGIN, gy)

                tick_c = QColor(_FG)
                tick_c.setAlpha(80)
                p.setPen(tick_c)
                p.setFont(QFont("Menlo", 7))
                p.drawText(w - _MARGIN + 2, gy - 8, _MARGIN - 2, 16,
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           f"{val:.0f}")

            # ── Waveform — agrégation vectorisée par colonne de pixels ────
            if self._frame_count > 1 and self._fps > 0:
                px_xs = (_MARGIN
                         + np.clip(ts_v * self._fps / (self._frame_count - 1),
                                   0.0, 1.0) * usable_x)
            else:
                t_rng = float(ts_v[-1] - ts_v[0]) if ts_v[-1] > ts_v[0] else 1.0
                px_xs = _MARGIN + (ts_v - ts_v[0]) / t_rng * usable_x

            py_ys = plot_top + (1.0 - (ang_v - a_min) / a_rng) * plot_h

            px_int = np.round(px_xs).astype(np.int32)
            order  = np.argsort(px_int, kind="stable")
            xs_s   = px_int[order]
            ys_s   = py_ys[order].astype(np.float32)
            splits = np.where(np.diff(xs_s) != 0)[0] + 1
            grps   = np.split(ys_s, splits)
            uxs    = xs_s[np.concatenate([[0], splits])]
            ymids  = np.fromiter(
                (float(g.mean()) for g in grps),
                dtype=np.float32, count=len(uxs),
            )

            if len(uxs) < 2:
                continue

            # Gradient fill
            grad = QLinearGradient(0.0, float(plot_top), 0.0, float(plot_bottom))
            fill_top = QColor(color); fill_top.setAlpha(55)
            fill_bot = QColor(color); fill_bot.setAlpha(4)
            grad.setColorAt(0.0, fill_top)
            grad.setColorAt(1.0, fill_bot)

            fill_path = QPainterPath()
            fill_path.moveTo(float(uxs[0]),  float(plot_bottom))
            fill_path.lineTo(float(uxs[0]),  float(ymids[0]))
            for i in range(1, len(uxs)):
                fill_path.lineTo(float(uxs[i]), float(ymids[i]))
            fill_path.lineTo(float(uxs[-1]), float(plot_bottom))
            fill_path.closeSubpath()

            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(fill_path)

            # Ligne principale
            line_path = QPainterPath()
            line_path.moveTo(float(uxs[0]), float(ymids[0]))
            for i in range(1, len(uxs)):
                line_path.lineTo(float(uxs[i]), float(ymids[i]))

            lc = QColor(color); lc.setAlpha(220)
            p.setPen(QPen(lc, 1.8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(line_path)

        # ── Ruler temporel ────────────────────────────────────────────────
        ruler_y = len(self._data) * _PANEL_H
        p.fillRect(0, ruler_y, w, _RULER_H, QColor("#181825"))
        p.setPen(QPen(_GRID, 1))
        p.drawLine(0, ruler_y, w, ruler_y)

        if self._fps > 0 and self._frame_count > 1:
            dur  = (self._frame_count - 1) / self._fps
            step = _nice_interval(dur, max_ticks=10)
            p.setFont(QFont("Menlo", 7))
            tc = QColor(_FG); tc.setAlpha(130)
            t = 0.0
            while t <= dur + step * 0.5:
                tx = int(self._t_to_x(t, w))
                p.setPen(QPen(_GRID, 1))
                p.drawLine(tx, ruler_y, tx, ruler_y + 4)
                p.setPen(tc)
                p.drawText(tx - 20, ruler_y + 4, 40, _RULER_H - 4,
                           Qt.AlignmentFlag.AlignCenter,
                           f"{t:.1f}s")
                t += step

        p.end()
        return pix

    # ------------------------------------------------------------------
    # paintEvent — couche statique + couche dynamique (curseur)
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0 or not self._data:
            return

        if (self._cache is None
                or self._cache.width() != w
                or self._cache.height() != h):
            self._cache = self._build_static(w, h)

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.drawPixmap(0, 0, self._cache)

        cx = int(self._t_to_x(self._current_time, w))

        for row, (gid, (ts_arr, ang_arr)) in enumerate(self._data.items()):
            color = _COLORS.get(gid, QColor("#cdd6f4"))
            py0         = row * _PANEL_H
            plot_top    = py0 + _TITLE_H
            plot_bottom = py0 + _PANEL_H - 2
            plot_h      = plot_bottom - plot_top

            # Curseur vertical (toute la hauteur du panneau)
            p.setPen(QPen(_CURSOR, 1, Qt.PenStyle.SolidLine))
            p.drawLine(cx, py0 + 1, cx, py0 + _PANEL_H - 1)

            # Interpolation de la valeur courante
            valid = np.isfinite(ang_arr)
            if not valid.any():
                continue

            ts_v  = ts_arr[valid]
            ang_v = ang_arr[valid]
            a_min = float(ang_v.min())
            a_max = float(ang_v.max())
            a_rng = a_max - a_min if a_max > a_min else 1.0

            val = float(np.interp(self._current_time, ts_v, ang_v))
            if not np.isfinite(val):
                continue

            norm  = (val - a_min) / a_rng
            dot_y = int(plot_top + (1.0 - norm) * plot_h)

            # Dot sur la courbe
            p.setPen(QPen(_CURSOR, 1.5))
            p.setBrush(QColor(color))
            p.drawEllipse(cx - 4, dot_y - 4, 8, 8)

            # Label valeur (bascule côté si proche du bord droit)
            vc = QColor(color); vc.setAlpha(235)
            p.setPen(vc)
            p.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
            lx = cx + 8 if cx < w - 72 else cx - 70
            p.drawText(lx, dot_y - 7, 62, 14,
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"{val:.1f} mm")

            # Résumé valeur + min/max dans le bandeau titre (top-right)
            ic = QColor(color); ic.setAlpha(155)
            p.setPen(ic)
            p.setFont(QFont("Menlo", 7))
            side = "Left" if gid in ("left", "1") else "Right"
            usable_x = w - 2 * _MARGIN
            p.drawText(_MARGIN, py0, usable_x, _TITLE_H,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{val:.1f} mm  ·  min {a_min:.0f}  max {a_max:.0f}")

        p.end()
