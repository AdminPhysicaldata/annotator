"""Dialogue de gestion des labels d'un scénario (réservé aux chefs).

Permet à un chef de définir la liste des labels associés à un scénario
en base de données MongoDB. Les annotateurs voient ces labels en lecture seule.
"""

from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QLineEdit,
    QColorDialog, QInputDialog, QMessageBox, QComboBox,
    QDialogButtonBox, QFrame, QWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from ...storage.mongodb_client import MongoDBClient


# ---------------------------------------------------------------------------
# Dialogue principal
# ---------------------------------------------------------------------------

class ScenarioLabelDialog(QDialog):
    """Dialogue permettant à un chef de gérer les labels d'un scénario.

    Usage:
        dlg = ScenarioLabelDialog(mongo_client, scenario_name, parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Les labels ont été sauvegardés en BDD
    """

    def __init__(
        self,
        mongo_client: MongoDBClient,
        scenario_name: str,
        parent=None,
    ):
        super().__init__(parent)
        self.mongo_client = mongo_client
        self.scenario_name = scenario_name
        self._labels: List[Dict[str, Any]] = []  # [{name, color, description}, ...]

        self.setWindowTitle(f"Gérer les labels — {scenario_name}")
        self.setMinimumSize(520, 480)
        self.setStyleSheet("""
            QDialog { background: #1e1e2e; color: #cdd6f4; }
            QLabel  { color: #cdd6f4; }
        """)

        self._load_labels_from_db()
        self._setup_ui()
        self._refresh_list()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _load_labels_from_db(self) -> None:
        raw = self.mongo_client.get_scenario_labels(self.scenario_name)
        self._labels = []
        for lbl in raw:
            if isinstance(lbl, str):
                name = lbl.strip()
                if name:
                    self._labels.append({"name": name, "color": "#89b4fa", "description": ""})
            elif isinstance(lbl, dict):
                name = lbl.get("name", "").strip()
                if name:
                    self._labels.append({
                        "name": name,
                        "color": lbl.get("color", "#89b4fa"),
                        "description": lbl.get("description", ""),
                    })

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Header
        title = QLabel(f"Labels du scénario « {self.scenario_name} »")
        title.setFont(QFont("", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Définissez ici les labels que les annotateurs pourront utiliser pour ce scénario."
        )
        subtitle.setStyleSheet("color: #a6adc8; font-size: 11px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244;")
        layout.addWidget(sep)

        # Label list
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background: #181825; border: 1px solid #313244; "
            "border-radius: 4px; color: #cdd6f4; font-size: 12px; padding: 2px; } "
            "QListWidget::item { padding: 8px 12px; border-bottom: 1px solid #2a2a3e; } "
            "QListWidget::item:selected { background: rgba(137,180,250,0.3); "
            "border-left: 3px solid #89b4fa; } "
            "QListWidget::item:hover { background: #2a2a3e; }"
        )
        layout.addWidget(self.list_widget, stretch=1)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        add_btn = self._make_btn("+ Ajouter", "#1e3a2e", "#a6e3a1")
        add_btn.clicked.connect(self._add_label)
        btn_row.addWidget(add_btn)

        edit_btn = self._make_btn("Modifier", "#1e2a3a", "#89b4fa")
        edit_btn.clicked.connect(self._edit_label)
        btn_row.addWidget(edit_btn)

        del_btn = self._make_btn("Supprimer", "#3a1e1e", "#f38ba8")
        del_btn.clicked.connect(self._delete_label)
        btn_row.addWidget(del_btn)

        btn_row.addStretch()

        up_btn = self._make_btn("▲", "#2a2a3e", "#cdd6f4", fixed_width=36)
        up_btn.clicked.connect(self._move_up)
        btn_row.addWidget(up_btn)

        down_btn = self._make_btn("▼", "#2a2a3e", "#cdd6f4", fixed_width=36)
        down_btn.clicked.connect(self._move_down)
        btn_row.addWidget(down_btn)

        layout.addLayout(btn_row)

        # Dialog buttons
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #313244;")
        layout.addWidget(sep2)

        dialog_btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        dialog_btns.setStyleSheet(
            "QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; "
            "border-radius: 4px; padding: 6px 16px; } "
            "QPushButton:hover { background: #45475a; } "
            "QPushButton[text='Save'] { background: #1e3a2e; color: #a6e3a1; border-color: #a6e3a1; }"
        )
        dialog_btns.accepted.connect(self._save)
        dialog_btns.rejected.connect(self.reject)
        layout.addWidget(dialog_btns)

    @staticmethod
    def _make_btn(text: str, bg: str, fg: str, fixed_width: int = 0) -> QPushButton:
        btn = QPushButton(text)
        btn.setStyleSheet(
            f"QPushButton {{ background: {bg}; color: {fg}; border: 1px solid {fg}; "
            f"border-radius: 3px; padding: 5px 10px; font-size: 11px; }} "
            f"QPushButton:hover {{ background: {bg}; opacity: 0.8; }}"
        )
        if fixed_width:
            btn.setFixedWidth(fixed_width)
        return btn

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for lbl in self._labels:
            item = QListWidgetItem(f"  {lbl['name']}")
            item.setData(Qt.ItemDataRole.UserRole, lbl)
            color = QColor(lbl.get("color", "#89b4fa"))
            item.setBackground(color)
            brightness = (color.red() * 299 + color.green() * 587 + color.blue() * 114) / 1000
            item.setForeground(QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255))
            item.setFont(QFont("", 12, QFont.Weight.DemiBold))
            self.list_widget.addItem(item)

    def _selected_index(self) -> int:
        return self.list_widget.currentRow()

    def _add_label(self) -> None:
        name, ok = QInputDialog.getText(self, "Nouveau label", "Nom du label :")
        if not ok or not name.strip():
            return
        name = name.strip()
        if any(lbl["name"].lower() == name.lower() for lbl in self._labels):
            QMessageBox.warning(self, "Erreur", f"Le label « {name} » existe déjà.")
            return

        color = QColorDialog.getColor(QColor("#89b4fa"), self, "Couleur du label")
        if not color.isValid():
            return

        desc, _ = QInputDialog.getText(self, "Description", "Description (optionnel) :")
        self._labels.append({
            "name": name,
            "color": color.name(),
            "description": desc.strip(),
        })
        self._refresh_list()

    def _edit_label(self) -> None:
        idx = self._selected_index()
        if idx < 0:
            QMessageBox.information(self, "Sélection", "Sélectionnez un label à modifier.")
            return

        lbl = self._labels[idx]

        name, ok = QInputDialog.getText(self, "Modifier le label", "Nom :", text=lbl["name"])
        if not ok or not name.strip():
            return
        name = name.strip()

        if any(
            i != idx and self._labels[i]["name"].lower() == name.lower()
            for i in range(len(self._labels))
        ):
            QMessageBox.warning(self, "Erreur", f"Le label « {name} » existe déjà.")
            return

        color = QColorDialog.getColor(QColor(lbl["color"]), self, "Couleur du label")
        if not color.isValid():
            return

        desc, _ = QInputDialog.getText(
            self, "Description", "Description (optionnel) :", text=lbl.get("description", "")
        )
        self._labels[idx] = {
            "name": name,
            "color": color.name(),
            "description": desc.strip(),
        }
        self._refresh_list()

    def _delete_label(self) -> None:
        idx = self._selected_index()
        if idx < 0:
            QMessageBox.information(self, "Sélection", "Sélectionnez un label à supprimer.")
            return

        name = self._labels[idx]["name"]
        reply = QMessageBox.question(
            self, "Confirmer",
            f"Supprimer le label « {name} » du scénario ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._labels.pop(idx)
            self._refresh_list()

    def _move_up(self) -> None:
        idx = self._selected_index()
        if idx > 0:
            self._labels[idx - 1], self._labels[idx] = self._labels[idx], self._labels[idx - 1]
            self._refresh_list()
            self.list_widget.setCurrentRow(idx - 1)

    def _move_down(self) -> None:
        idx = self._selected_index()
        if 0 <= idx < len(self._labels) - 1:
            self._labels[idx], self._labels[idx + 1] = self._labels[idx + 1], self._labels[idx]
            self._refresh_list()
            self.list_widget.setCurrentRow(idx + 1)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self) -> None:
        ok = self.mongo_client.set_scenario_labels(self.scenario_name, self._labels)
        if ok:
            QMessageBox.information(
                self, "Sauvegardé",
                f"{len(self._labels)} label(s) sauvegardé(s) pour « {self.scenario_name} »."
            )
            self.accept()
        else:
            QMessageBox.critical(
                self, "Erreur",
                "Impossible de sauvegarder les labels en base de données.\n"
                "Vérifiez que le scénario existe et que vous êtes connecté."
            )

    # ------------------------------------------------------------------
    # Public accessors (for caller after dialog closed)
    # ------------------------------------------------------------------

    def get_labels(self) -> List[Dict[str, Any]]:
        """Return the current labels list."""
        return list(self._labels)
