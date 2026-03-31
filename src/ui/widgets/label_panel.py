"""Label management panel widget.

- Chef : voit les labels du scénario en BDD et peut les modifier via un dialogue dédié.
  Les boutons Add/Remove locaux sont masqués ; un bouton "Gérer les labels" s'affiche.
- Annotateur : voit les labels chargés depuis la BDD (lecture seule), sans possibilité
  d'ajouter ou supprimer des labels.
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QLineEdit,
    QColorDialog, QInputDialog, QMessageBox, QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from ...labeling.label_manager import LabelManager, LabelType, UNLABELED_LABEL_ID


class LabelPanel(QWidget):
    """Panel pour afficher et sélectionner des labels pour l'annotation.

    En mode chef, un bouton "Gérer les labels" ouvre le dialogue de gestion.
    En mode annotateur, la liste est en lecture seule.
    """

    label_selected = pyqtSignal(str)                   # label_id pour frame annotation
    interval_annotation_requested = pyqtSignal(str)    # label_id pour interval annotation
    manage_labels_requested = pyqtSignal()             # chef demande l'ouverture du dialogue

    def __init__(
        self,
        label_manager: LabelManager,
        is_chef: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.label_manager = label_manager
        self.is_chef = is_chef
        self._annotation_mode = "interval"
        self._setup_ui()
        self.refresh_labels()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- Header ---
        header_row = QHBoxLayout()
        header = QLabel("Labels")
        header.setFont(QFont("", 13, QFont.Weight.Bold))
        header.setStyleSheet("color: #cdd6f4;")
        header_row.addWidget(header)
        header_row.addStretch()

        if self.is_chef:
            role_badge = QLabel("CHEF")
            role_badge.setStyleSheet(
                "color: #fab387; background: #2a1e10; border: 1px solid #fab387; "
                "border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: bold;"
            )
            header_row.addWidget(role_badge)
        else:
            role_badge = QLabel("ANNOTATEUR")
            role_badge.setStyleSheet(
                "color: #89b4fa; background: #1e2a3a; border: 1px solid #89b4fa; "
                "border-radius: 3px; padding: 1px 6px; font-size: 10px; font-weight: bold;"
            )
            header_row.addWidget(role_badge)

        layout.addLayout(header_row)

        # --- Search bar ---
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Rechercher un label...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border: 1px solid #89b4fa;
            }
        """)
        self.search_input.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_input)

        # --- Label list ---
        self.label_list = QListWidget()
        self.label_list.setStyleSheet(
            "QListWidget { background: #1e1e2e; border: 1px solid #313244; border-radius: 4px; "
            "color: #cdd6f4; font-size: 12px; padding: 2px; } "
            "QListWidget::item { padding: 8px 12px; border-bottom: 1px solid #313244; "
            "text-align: left; } "
            "QListWidget::item:selected { background: rgba(137, 180, 250, 0.3); "
            "border-left: 3px solid #89b4fa; } "
            "QListWidget::item:hover { background: #2a2a3e; }"
        )
        self.label_list.itemClicked.connect(self._on_label_clicked)
        self.label_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.label_list)

        # --- Boutons selon le rôle ---
        if self.is_chef:
            # Chef : bouton de gestion des labels en BDD
            manage_btn = QPushButton("⚙ Gérer les labels du scénario")
            manage_btn.setStyleSheet(
                "QPushButton { background: #2a1e10; color: #fab387; border: 1px solid #fab387; "
                "border-radius: 3px; padding: 6px 12px; font-size: 11px; } "
                "QPushButton:hover { background: #3a2a10; }"
            )
            manage_btn.clicked.connect(self.manage_labels_requested.emit)
            layout.addWidget(manage_btn)
        # Annotateur : aucun bouton d'édition

        # --- Info ---
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # --- Statistics ---
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244;")
        layout.addWidget(sep)

        self.stats_label = QLabel("Annotations: 0")
        self.stats_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self.stats_label)

        self._update_info()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_is_chef(self, is_chef: bool) -> None:
        """Change le rôle dynamiquement (re-crée l'UI)."""
        self.is_chef = is_chef
        # Reconstruire l'UI
        for i in reversed(range(self.layout().count())):
            w = self.layout().itemAt(i).widget()
            if w:
                w.deleteLater()
        self._setup_ui()
        self.refresh_labels()

    def set_label_shortcuts(self, shortcuts: dict) -> None:
        """Reçoit un dict {label_id: numéro_touche (str)} depuis main_window."""
        self._label_shortcuts = shortcuts
        self.refresh_labels()

    def refresh_labels(self) -> None:
        """Rafraîchit la liste des labels."""
        self.label_list.clear()
        search_query = self.search_input.text().lower().strip() if hasattr(self, "search_input") else ""
        shortcuts = getattr(self, "_label_shortcuts", {})

        for label in self.label_manager.labels.values():
            # Ne jamais afficher le label interne "non labellisé"
            if label.id == UNLABELED_LABEL_ID:
                continue
            if search_query and search_query not in label.name.lower():
                continue

            key = shortcuts.get(label.id, "")
            display = f"[{key}]  {label.name}" if key else label.name
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, label.id)
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            color = QColor(label.color)
            item.setBackground(color)
            brightness = (color.red() * 299 + color.green() * 587 + color.blue() * 114) / 1000
            item.setForeground(QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255))
            item.setFont(QFont("", 12, QFont.Weight.DemiBold))
            self.label_list.addItem(item)

        self._update_statistics()

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_search_changed(self) -> None:
        self.refresh_labels()

    def _update_info(self) -> None:
        if self.is_chef:
            self.info_label.setText(
                "En tant que chef, vous pouvez définir les labels du scénario via le bouton ci-dessus, "
                "puis annoter normalement."
            )
        else:
            self.info_label.setText(
                "[A] coupe la track gauche · [Z] coupe la track droite · "
                "Cliquez un segment puis une touche numérique pour l'assigner."
            )

    def _on_label_clicked(self, item: QListWidgetItem) -> None:
        import logging
        logger = logging.getLogger(__name__)
        label_id = item.data(Qt.ItemDataRole.UserRole)
        logger.info("Label clicked: %s", label_id)
        self.search_input.clearFocus()
        self.label_list.setFocus()
        self.interval_annotation_requested.emit(label_id)

    def _update_statistics(self) -> None:
        stats = self.label_manager.get_statistics()
        self.stats_label.setText(
            f"Annotations: {stats['total_annotations']} "
            f"({stats['frame_annotations']} frame, {stats['interval_annotations']} interval)"
        )
