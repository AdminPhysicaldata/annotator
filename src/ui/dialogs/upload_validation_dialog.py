"""Upload validation dialog - rating and flags selection."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialogButtonBox, QGroupBox, QCheckBox, QSlider, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from typing import List, Tuple, Optional


class UploadValidationDialog(QDialog):
    """Dialog to validate upload with rating (0-4) and flags selection."""

    # Liste des flags disponibles
    AVAILABLE_FLAGS = [
        "Vidéo floue",
        "Mauvais éclairage",
        "Objet hors cadre",
        "Mouvement trop rapide",
        "Occlusion partielle",
        "Données incomplètes",
        "Problème de synchronisation",
        "Bruit dans les données",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Validation de l'upload")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self._rating = 2  # Default rating
        self._flags_checkboxes = {}

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Rating section ---
        rating_group = QGroupBox("Note de qualité")
        rating_group.setStyleSheet("""
            QGroupBox {
                color: #cdd6f4;
                border: 2px solid #313244;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        rating_layout = QVBoxLayout(rating_group)
        rating_layout.setSpacing(12)

        # Rating display (large number)
        self.rating_display = QLabel("2")
        self.rating_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rating_display.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        self.rating_display.setStyleSheet("color: #89b4fa; padding: 10px;")
        rating_layout.addWidget(self.rating_display)

        # Star buttons (0-4)
        stars_layout = QHBoxLayout()
        stars_layout.setSpacing(8)
        stars_layout.addStretch()

        self.star_buttons = []
        for i in range(5):
            btn = QPushButton(f"{i}")
            btn.setFixedSize(60, 60)
            btn.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            btn.setCheckable(True)
            btn.setStyleSheet(self._get_star_button_style(i == 2))
            btn.clicked.connect(lambda checked, rating=i: self._on_rating_selected(rating))
            self.star_buttons.append(btn)
            stars_layout.addWidget(btn)

        stars_layout.addStretch()
        rating_layout.addLayout(stars_layout)

        # Rating labels
        labels_layout = QHBoxLayout()
        labels_layout.addWidget(QLabel("Mauvais"), alignment=Qt.AlignmentFlag.AlignLeft)
        labels_layout.addStretch()
        labels_layout.addWidget(QLabel("Excellent"), alignment=Qt.AlignmentFlag.AlignRight)
        labels_layout.setContentsMargins(10, 0, 10, 0)
        rating_layout.addLayout(labels_layout)

        layout.addWidget(rating_group)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244;")
        layout.addWidget(sep)

        # --- Flags section ---
        flags_group = QGroupBox("Flags (optionnel)")
        flags_group.setStyleSheet("""
            QGroupBox {
                color: #cdd6f4;
                border: 2px solid #313244;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        flags_layout = QVBoxLayout(flags_group)
        flags_layout.setSpacing(8)

        info_label = QLabel("Sélectionnez les problèmes rencontrés (plusieurs choix possibles) :")
        info_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: normal;")
        flags_layout.addWidget(info_label)

        # Create checkboxes for each flag
        for flag in self.AVAILABLE_FLAGS:
            cb = QCheckBox(flag)
            cb.setStyleSheet("""
                QCheckBox {
                    color: #cdd6f4;
                    font-size: 12px;
                    font-weight: normal;
                    padding: 4px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 2px solid #585b70;
                    border-radius: 3px;
                    background: #1e1e2e;
                }
                QCheckBox::indicator:checked {
                    background: #89b4fa;
                    border-color: #89b4fa;
                }
                QCheckBox::indicator:hover {
                    border-color: #89b4fa;
                }
            """)
            self._flags_checkboxes[flag] = cb
            flags_layout.addWidget(cb)

        layout.addWidget(flags_group)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                font-size: 12px;
                border-radius: 4px;
            }
            QPushButton[text="OK"] {
                background: #40a02b;
                color: white;
                border: none;
            }
            QPushButton[text="OK"]:hover {
                background: #4cb832;
            }
            QPushButton[text="Cancel"] {
                background: #585b70;
                color: white;
                border: none;
            }
            QPushButton[text="Cancel"]:hover {
                background: #6c7086;
            }
        """)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Set initial rating
        self._on_rating_selected(2)

    def _get_star_button_style(self, selected: bool) -> str:
        """Get stylesheet for star button."""
        if selected:
            return """
                QPushButton {
                    background: #89b4fa;
                    color: white;
                    border: 2px solid #89b4fa;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background: #a6c9ff;
                    border-color: #a6c9ff;
                }
            """
        else:
            return """
                QPushButton {
                    background: #1e1e2e;
                    color: #cdd6f4;
                    border: 2px solid #585b70;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background: #313244;
                    border-color: #89b4fa;
                }
            """

    def _on_rating_selected(self, rating: int):
        """Handle rating selection."""
        self._rating = rating
        self.rating_display.setText(str(rating))

        # Update button styles
        for i, btn in enumerate(self.star_buttons):
            btn.setStyleSheet(self._get_star_button_style(i == rating))
            btn.setChecked(i == rating)

    def get_rating(self) -> int:
        """Get selected rating (0-4)."""
        return self._rating

    def get_flags(self) -> List[str]:
        """Get list of selected flags."""
        return [flag for flag, cb in self._flags_checkboxes.items() if cb.isChecked()]

    @staticmethod
    def get_validation(parent=None) -> Optional[Tuple[int, List[str]]]:
        """Show dialog and return (rating, flags) or None if cancelled.

        Returns:
            (rating, flags) if accepted, None if cancelled
        """
        dialog = UploadValidationDialog(parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_rating(), dialog.get_flags()
        return None
