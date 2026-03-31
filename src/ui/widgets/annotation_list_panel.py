"""Annotation list panel — CRUD complet pour les annotations existantes.

- Clic simple  → pop-up de détail avec édition et suppression
- Double-clic  → naviguer au frame dans le player
- Filtre par type (Tout / Frame / Interval)
- Bouton "Tout effacer" en bas
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QMessageBox, QDialog, QFormLayout, QComboBox, QSpinBox,
    QDialogButtonBox, QSizePolicy, QFrame, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont

from ...labeling.label_manager import LabelManager, LabelType, Annotation


# ── Styles ────────────────────────────────────────────────────────────────────

_SS_BASE = "background:#1e1e2e; color:#cdd6f4;"

_SS_LIST = """
QListWidget {
    background: #181825;
    border: 1px solid #313244;
    border-radius: 4px;
    color: #cdd6f4;
    font-size: 11px;
}
QListWidget::item {
    padding: 6px 8px;
    border-bottom: 1px solid #11111b;
}
QListWidget::item:selected { background: #45475a; }
QListWidget::item:hover    { background: #2a2a3e; }
"""

_SS_BTN = lambda bg, fg, hov: (
    f"QPushButton {{ background:{bg}; color:{fg}; border:1px solid {fg}; "
    f"border-radius:4px; padding:4px 12px; font-size:11px; }}"
    f"QPushButton:hover {{ background:{hov}; }}"
    f"QPushButton:disabled {{ background:#1e1e2e; color:#45475a; border-color:#45475a; }}"
)

_SS_FILTER = (
    "QPushButton { background:#313244; color:#a6adc8; border:1px solid #45475a; "
    "border-radius:3px; padding:2px 8px; font-size:10px; }"
    "QPushButton:checked { background:#45475a; color:#cdd6f4; border-color:#cdd6f4; }"
    "QPushButton:hover { background:#3a3a5c; }"
)

_SS_DIALOG = """
QDialog          { background:#1e1e2e; }
QLabel           { color:#cdd6f4; font-size:11px; }
QComboBox, QSpinBox {
    background:#313244; color:#cdd6f4;
    border:1px solid #45475a; border-radius:3px;
    padding:8px 12px; font-size:11px;
    min-width: 200px;
}
QComboBox::drop-down { border:none; width: 24px; }
QComboBox QAbstractItemView {
    background:#313244; color:#cdd6f4;
    selection-background-color:#45475a;
    padding: 0px;
    outline: none;
    min-width: 200px;
    border: 1px solid #45475a;
}
QComboBox QAbstractItemView::item {
    padding: 8px 12px;
    min-height: 32px;
}
QComboBox QAbstractItemView::item:selected {
    background:#45475a;
}
QFrame[frameShape="4"] { color:#313244; }
"""


# ── Pop-up de détail / édition ────────────────────────────────────────────────

class AnnotationDetailDialog(QDialog):
    """Pop-up affichant les détails d'une annotation avec édition et suppression."""

    deleted = pyqtSignal()   # émis si l'utilisateur supprime
    saved   = pyqtSignal()   # émis si l'utilisateur sauvegarde

    def __init__(self, annotation: Annotation, label_manager: LabelManager, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.label_manager = label_manager

        self.setWindowTitle("Annotation")
        self.setMinimumWidth(400)
        self.setModal(True)
        self.setStyleSheet(_SS_DIALOG)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 16)

        ann = self.annotation

        # ── En-tête (type + id court) ──────────────────────────────────
        type_label = QLabel(
            "Annotation frame" if ann.annotation_type == LabelType.FRAME else "Annotation intervalle"
        )
        type_label.setFont(QFont("", 13, QFont.Weight.Bold))
        type_label.setStyleSheet("color:#89b4fa;")
        layout.addWidget(type_label)

        id_label = QLabel(f"ID : {ann.id[:8]}…")
        id_label.setStyleSheet("color:#585b70; font-size:9px;")
        layout.addWidget(id_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#313244;")
        layout.addWidget(sep)

        # ── Formulaire d'édition ───────────────────────────────────────
        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Label
        self.label_combo = QComboBox()
        self.label_combo.setEditable(False)
        self.label_combo.view().setAlternatingRowColors(False)
        for lbl in self.label_manager.labels.values():
            self.label_combo.addItem(lbl.name, userData=lbl.id)
            if lbl.id == ann.label_id:
                self.label_combo.setCurrentIndex(self.label_combo.count() - 1)
        form.addRow("Label :", self.label_combo)

        # Main (hand)
        self.hand_combo = QComboBox()
        self.hand_combo.setEditable(False)
        self.hand_combo.view().setAlternatingRowColors(False)
        self.hand_combo.addItems(["right", "left", ""])
        hand_val = ann.metadata.get("hand", "")
        idx = self.hand_combo.findText(hand_val)
        self.hand_combo.setCurrentIndex(idx if idx >= 0 else 2)
        form.addRow("Main :", self.hand_combo)

        # Fail flag
        self.fail_cb = QCheckBox("[fail]  — tentative échouée")
        self.fail_cb.setChecked(bool(ann.metadata.get("fail", False)))
        self.fail_cb.setStyleSheet("color: #f38ba8; font-weight: bold;")
        form.addRow("", self.fail_cb)

        # Frames
        if ann.annotation_type == LabelType.FRAME:
            self.frame_spin = QSpinBox()
            self.frame_spin.setRange(0, 999_999)
            self.frame_spin.setValue(ann.frame_index or 0)
            form.addRow("Frame :", self.frame_spin)
            self.start_spin = self.end_spin = None
        else:
            self.start_spin = QSpinBox()
            self.start_spin.setRange(0, 999_999)
            self.start_spin.setValue(ann.start_frame or 0)
            self.start_spin.valueChanged.connect(self._clamp_end)
            form.addRow("Frame début :", self.start_spin)

            self.end_spin = QSpinBox()
            self.end_spin.setRange(0, 999_999)
            self.end_spin.setValue(ann.end_frame or 0)
            form.addRow("Frame fin :", self.end_spin)
            self.frame_spin = None

        layout.addLayout(form)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#313244;")
        layout.addWidget(sep2)

        # ── Boutons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        del_btn = QPushButton("Supprimer")
        del_btn.setStyleSheet(_SS_BTN("#3a1e1e", "#f38ba8", "#5a2e2e"))
        del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(del_btn)

        btn_row.addStretch()

        cancel_btn = QPushButton("Annuler")
        cancel_btn.setStyleSheet(_SS_BTN("#313244", "#a6adc8", "#45475a"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Enregistrer")
        save_btn.setDefault(True)
        save_btn.setStyleSheet(_SS_BTN("#1e3a2e", "#a6e3a1", "#2e5a3e"))
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

    # ── Validation / actions ───────────────────────────────────────────

    def _clamp_end(self, start_val: int):
        if self.end_spin is not None and self.end_spin.value() < start_val:
            self.end_spin.setValue(start_val)

    def _on_save(self):
        ann = self.annotation
        lm = self.label_manager

        label_id = self.label_combo.currentData()
        new_label = lm.labels.get(label_id)
        if new_label is None:
            QMessageBox.warning(self, "Erreur", "Label invalide.")
            return

        if ann.annotation_type == LabelType.INTERVAL:
            start = self.start_spin.value()
            end = self.end_spin.value()
            if start > end:
                QMessageBox.warning(self, "Erreur", "La frame de début doit être ≤ à la frame de fin.")
                return

        # Retirer des index
        _remove_from_indices(ann, lm)

        # Modifier l'objet
        ann.label_id = label_id
        ann.label_name = new_label.name
        ann.metadata["hand"] = self.hand_combo.currentText()
        ann.metadata["fail"] = self.fail_cb.isChecked()

        if ann.annotation_type == LabelType.FRAME:
            ann.frame_index = self.frame_spin.value()
        else:
            ann.start_frame = self.start_spin.value()
            ann.end_frame = self.end_spin.value()

        # Réinsérer dans les index
        _insert_into_indices(ann, lm)

        self.saved.emit()
        self.accept()

    def _on_delete(self):
        reply = QMessageBox.question(
            self, "Supprimer",
            "Supprimer cette annotation ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self.label_manager.remove_annotation(self.annotation.id)
        except KeyError:
            pass
        self.deleted.emit()
        self.accept()


# ── Helpers d'index ──────────────────────────────────────────────────────────

def _remove_from_indices(ann: Annotation, lm: LabelManager):
    if ann.annotation_type == LabelType.FRAME:
        frames = [ann.frame_index]
    else:
        frames = range(ann.start_frame, ann.end_frame + 1)

    for fi in frames:
        lst = lm._annotations_by_frame.get(fi)
        if lst and ann in lst:
            lst.remove(ann)

    lst = lm._annotations_by_label.get(ann.label_id)
    if lst and ann in lst:
        lst.remove(ann)


def _insert_into_indices(ann: Annotation, lm: LabelManager):
    if ann.annotation_type == LabelType.FRAME:
        frames = [ann.frame_index]
    else:
        frames = range(ann.start_frame, ann.end_frame + 1)

    for fi in frames:
        lm._annotations_by_frame.setdefault(fi, []).append(ann)

    lm._annotations_by_label.setdefault(ann.label_id, []).append(ann)


# ── Panneau principal ─────────────────────────────────────────────────────────

class AnnotationListPanel(QWidget):
    """Panneau listant les annotations — clic → pop-up, double-clic → goto frame."""

    annotation_selected = pyqtSignal(int)   # frame à afficher dans le player
    annotations_changed = pyqtSignal()       # après tout C/U/D

    def __init__(self, label_manager: LabelManager, parent=None):
        super().__init__(parent)
        self.label_manager = label_manager
        self._active_filter = "all"
        # Timer pour distinguer clic simple du double-clic
        self._click_timer = QTimer()
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(250)
        self._click_timer.timeout.connect(self._fire_single_click)
        self._pending_click_item = None
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("Annotations")
        title.setFont(QFont("", 12, QFont.Weight.Bold))
        title.setStyleSheet("color:#cdd6f4;")
        hdr.addWidget(title)
        hdr.addStretch()
        self.count_label = QLabel("0")
        self.count_label.setStyleSheet("color:#6c7086; font-size:10px;")
        hdr.addWidget(self.count_label)
        layout.addLayout(hdr)

        # Filtres
        frow = QHBoxLayout()
        frow.setSpacing(4)
        self._filter_btns = {}
        for key, txt in (("all", "Tout"), ("frame", "Frame"), ("interval", "Interval")):
            btn = QPushButton(txt)
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setStyleSheet(_SS_FILTER)
            btn.clicked.connect(lambda _, k=key: self._set_filter(k))
            frow.addWidget(btn)
            self._filter_btns[key] = btn
        self._filter_btns["all"].setChecked(True)
        layout.addLayout(frow)

        # Liste
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(_SS_LIST)
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.itemClicked.connect(self._on_item_click_raw)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_click)
        layout.addWidget(self.list_widget, stretch=1)

        # Aide
        hint = QLabel("Clic → détails/édition   •   Double-clic → aller au frame")
        hint.setStyleSheet("color:#45475a; font-size:9px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#313244;")
        layout.addWidget(sep)

        # Tout effacer
        self.clear_btn = QPushButton("Tout effacer")
        self.clear_btn.setStyleSheet(_SS_BTN("#3a1e1e", "#f38ba8", "#5a2e2e"))
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._clear_all)
        layout.addWidget(self.clear_btn)

    # ── Filtre ────────────────────────────────────────────────────────

    def _set_filter(self, f: str):
        self._active_filter = f
        for key, btn in self._filter_btns.items():
            btn.setChecked(key == f)
        self.refresh()

    # ── Refresh (Read) ────────────────────────────────────────────────

    def refresh(self):
        self.list_widget.clear()

        anns = self.label_manager.annotations
        if self._active_filter == "frame":
            anns = [a for a in anns if a.annotation_type == LabelType.FRAME]
        elif self._active_filter == "interval":
            anns = [a for a in anns if a.annotation_type == LabelType.INTERVAL]

        for ann in anns:
            label = self.label_manager.labels.get(ann.label_id)
            color_hex = label.color if label else "#6c7086"
            hand = ann.metadata.get("hand", "")

            if ann.annotation_type == LabelType.FRAME:
                detail = f"frame {ann.frame_index}"
                icon_char = "◆"
            else:
                detail = f"frames {ann.start_frame} → {ann.end_frame}"
                icon_char = "▬"

            hand_tag = f"  [{hand}]" if hand else ""
            fail_tag = "  ⚠ [fail]" if ann.metadata.get("fail") else ""
            text = f"{icon_char}  {ann.label_name}{fail_tag}{hand_tag}\n    {detail}"

            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, ann.id)
            item.setFont(QFont("Courier", 10))

            # Couleur du label en fond léger
            bg = QColor(color_hex)
            bg.setAlpha(45)
            item.setBackground(bg)
            item.setForeground(QColor("#ffffff"))

            self.list_widget.addItem(item)

        total = len(self.label_manager.annotations)
        shown = self.list_widget.count()
        suffix = f" / {total}" if self._active_filter != "all" else ""
        self.count_label.setText(f"{shown}{suffix} annotation(s)")
        self.clear_btn.setEnabled(total > 0)

    # ── Interactions ──────────────────────────────────────────────────

    def _on_item_click_raw(self, item: QListWidgetItem):
        """Clic simple : attendre 250 ms pour ne pas confondre avec double-clic."""
        self._pending_click_item = item
        self._click_timer.start()

    def _fire_single_click(self):
        item = self._pending_click_item
        self._pending_click_item = None
        if item is None:
            return
        ann = self._ann_from_item(item)
        if ann is None:
            return
        dlg = AnnotationDetailDialog(ann, self.label_manager, self)
        dlg.saved.connect(self._on_changed)
        dlg.deleted.connect(self._on_changed)
        dlg.exec()

    def _on_item_double_click(self, item: QListWidgetItem):
        """Double-clic : annuler la pop-up et naviguer au frame."""
        self._click_timer.stop()
        self._pending_click_item = None
        ann = self._ann_from_item(item)
        if ann is None:
            return
        frame = ann.frame_index if ann.annotation_type == LabelType.FRAME else ann.start_frame
        self.annotation_selected.emit(frame)

    def _on_changed(self):
        self.refresh()
        self.annotations_changed.emit()

    def _clear_all(self):
        n = len(self.label_manager.annotations)
        if n == 0:
            return
        reply = QMessageBox.question(
            self, "Tout effacer",
            f"Effacer les {n} annotation(s) ? Les labels seront conservés.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.label_manager.clear_annotations()
        self._on_changed()

    # ── Helper ────────────────────────────────────────────────────────

    def _ann_from_item(self, item: QListWidgetItem):
        ann_id = item.data(Qt.ItemDataRole.UserRole)
        for ann in self.label_manager.annotations:
            if ann.id == ann_id:
                return ann
        return None
