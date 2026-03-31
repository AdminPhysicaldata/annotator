# """SeqensorWidget — panneau latéral affichant les segments auto-détectés par fluxseq."""
#
# from __future__ import annotations
#
# from typing import List, Optional
#
# from PyQt6.QtCore import Qt, pyqtSignal, QTimer
# from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush
# from PyQt6.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
#     QScrollArea, QFrame, QSizePolicy, QToolButton,
# )
#
#
# # ---------------------------------------------------------------------------
# # Badge coloré indiquant le cluster d'un segment
# # ---------------------------------------------------------------------------
#
# class _ColourBadge(QLabel):
#     """Petit label coloré affichant le numéro de cluster (ex. "L0", "L1"…)."""
#
#     def __init__(self, colour: str, text: str, parent=None):
#         super().__init__(text, parent)
#         self.setFixedSize(48, 20)
#         self.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.setStyleSheet(
#             f"background-color: {colour}; color: #1e1e2e; border-radius: 4px; "
#             f"font-size: 10px; font-weight: bold; font-family: Courier;"
#         )
#
#
# # ---------------------------------------------------------------------------
# # Ligne représentant un segment détecté
# # ---------------------------------------------------------------------------
#
# class _SegmentRow(QFrame):
#     """Widget représentant un seul segment dans la liste."""
#
#     jump_requested   = pyqtSignal(int)
#     apply_requested  = pyqtSignal(dict)
#
#     def __init__(self, seg: dict, index: int, parent=None):
#         super().__init__(parent)
#         self._seg = seg
#         self._index = index
#
#         self.setFrameShape(QFrame.Shape.StyledPanel)
#         self.setStyleSheet(
#             "QFrame { background: #181825; border: 1px solid #313244; border-radius: 4px; }"
#             "QFrame:hover { border: 1px solid #585b70; }"
#         )
#
#         layout = QHBoxLayout(self)
#         layout.setContentsMargins(6, 4, 6, 4)
#         layout.setSpacing(6)
#
#         colour  = seg.get("colour", "#89b4fa")
#         label   = seg.get("label", 0)
#         start_t = seg.get("start_t", 0.0)
#         end_t   = seg.get("end_t", 0.0)
#         dur     = seg.get("duration_s", end_t - start_t)
#
#         badge = _ColourBadge(colour, f"L{label}")
#         layout.addWidget(badge)
#
#         info_lbl = QLabel(
#             f"#{index + 1}  {start_t:.2f}s → {end_t:.2f}s  ({dur:.2f}s)"
#         )
#         info_lbl.setStyleSheet("color: #cdd6f4; font-family: Courier; font-size: 11px;")
#         layout.addWidget(info_lbl, stretch=1)
#
#         jump_btn = QToolButton()
#         jump_btn.setText("▶")
#         jump_btn.setFixedSize(22, 22)
#         jump_btn.setToolTip("Aller au début du segment")
#         jump_btn.setStyleSheet(
#             "QToolButton { background: #313244; color: #89b4fa; border: 1px solid #45475a; "
#             "border-radius: 3px; font-size: 11px; }"
#             "QToolButton:hover { background: #45475a; }"
#         )
#         jump_btn.clicked.connect(lambda: self.jump_requested.emit(seg.get("start_idx", 0)))
#         layout.addWidget(jump_btn)
#
#         apply_btn = QToolButton()
#         apply_btn.setText("+")
#         apply_btn.setFixedSize(22, 22)
#         apply_btn.setToolTip("Convertir en annotation d'intervalle")
#         apply_btn.setStyleSheet(
#             "QToolButton { background: #1e3a2e; color: #a6e3a1; border: 1px solid #a6e3a1; "
#             "border-radius: 3px; font-size: 14px; font-weight: bold; }"
#             "QToolButton:hover { background: #2e5a3e; }"
#         )
#         apply_btn.clicked.connect(lambda: self.apply_requested.emit(seg))
#         layout.addWidget(apply_btn)
#
#
# # ---------------------------------------------------------------------------
# # Panneau principal Seqensor
# # ---------------------------------------------------------------------------
#
# class SeqensorWidget(QWidget):
#     """Panneau latéral listant les segments auto-détectés par le pipeline fluxseq."""
#
#     jump_to_frame  = pyqtSignal(int)
#     apply_segment  = pyqtSignal(dict)
#
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._segments: List[dict] = []
#         self._setup_ui()
#
#     def _setup_ui(self) -> None:
#         """Construit la hiérarchie de widgets du panneau."""
#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(4, 4, 4, 4)
#         layout.setSpacing(4)
#
#         header = QHBoxLayout()
#         title = QLabel("Seqensor — Segments auto")
#         title.setStyleSheet(
#             "color: #89b4fa; font-family: Courier; font-size: 12px; font-weight: bold;"
#         )
#         header.addWidget(title)
#         header.addStretch()
#
#         self._status_lbl = QLabel("En attente…")
#         self._status_lbl.setStyleSheet(
#             "color: #6c7086; font-family: Courier; font-size: 10px;"
#         )
#         header.addWidget(self._status_lbl)
#
#         layout.addLayout(header)
#
#         self._running_lbl = QLabel("Analyse en cours…")
#         self._running_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self._running_lbl.setStyleSheet(
#             "color: #f9e2af; font-family: Courier; font-size: 11px;"
#         )
#         self._running_lbl.hide()
#         layout.addWidget(self._running_lbl)
#
#         self._scroll = QScrollArea()
#         self._scroll.setWidgetResizable(True)
#         self._scroll.setStyleSheet(
#             "QScrollArea { background: #11111b; border: 1px solid #313244; border-radius: 4px; }"
#             "QScrollBar:vertical { background: #1e1e2e; width: 8px; }"
#             "QScrollBar::handle:vertical { background: #45475a; border-radius: 3px; }"
#         )
#
#         self._list_widget = QWidget()
#         self._list_layout = QVBoxLayout(self._list_widget)
#         self._list_layout.setContentsMargins(4, 4, 4, 4)
#         self._list_layout.setSpacing(3)
#         self._list_layout.addStretch()
#
#         self._scroll.setWidget(self._list_widget)
#         layout.addWidget(self._scroll, stretch=1)
#
#         apply_all_btn = QPushButton("Appliquer tous comme annotations")
#         apply_all_btn.setStyleSheet(
#             "QPushButton { background: #313244; color: #89b4fa; border: 1px solid #45475a; "
#             "border-radius: 4px; padding: 4px; font-size: 11px; }"
#             "QPushButton:hover { background: #45475a; }"
#             "QPushButton:disabled { color: #585b70; border-color: #313244; }"
#         )
#         apply_all_btn.clicked.connect(self._apply_all)
#         apply_all_btn.setEnabled(False)
#         self._apply_all_btn = apply_all_btn
#         layout.addWidget(apply_all_btn)
#
#         self.setMinimumWidth(260)
#
#     def set_running(self, running: bool) -> None:
#         """Affiche ou masque l'indicateur d'analyse en cours."""
#         self._running_lbl.setVisible(running)
#         if running:
#             self._status_lbl.setText("Analyse en cours…")
#
#     def set_status(self, text: str) -> None:
#         """Met à jour le label de statut."""
#         self._status_lbl.setText(text)
#
#     def set_segments(self, segments: List[dict]) -> None:
#         """Peuple la liste avec les segments détectés par le worker."""
#         self._segments = segments
#         self._running_lbl.hide()
#
#         while self._list_layout.count() > 1:
#             item = self._list_layout.takeAt(0)
#             if item.widget():
#                 item.widget().deleteLater()
#
#         if not segments:
#             empty_lbl = QLabel("Aucun segment détecté.")
#             empty_lbl.setStyleSheet(
#                 "color: #6c7086; font-family: Courier; font-size: 11px;"
#             )
#             empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
#             self._list_layout.insertWidget(0, empty_lbl)
#             self._apply_all_btn.setEnabled(False)
#             return
#
#         for i, seg in enumerate(segments):
#             row = _SegmentRow(seg, i)
#             row.jump_requested.connect(self.jump_to_frame)
#             row.apply_requested.connect(self.apply_segment)
#             self._list_layout.insertWidget(i, row)
#
#         n_labels = len(set(s.get("label", 0) for s in segments))
#         self._status_lbl.setText(
#             f"{len(segments)} segments, {n_labels} classes"
#         )
#         self._apply_all_btn.setEnabled(True)
#
#     def clear(self) -> None:
#         """Réinitialise le panneau."""
#         self.set_segments([])
#         self._status_lbl.setText("En attente…")
#
#     def _apply_all(self) -> None:
#         """Émet apply_segment pour chaque segment de la liste."""
#         for seg in self._segments:
#             self.apply_segment.emit(seg)
