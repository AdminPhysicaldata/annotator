"""Login dialog — authenticates an annotator against physical_data.annotators."""

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox, QFrame,
    QComboBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

if TYPE_CHECKING:
    from ...storage.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class MongoLoginDialog(QDialog):
    """Modal dialog that authenticates an annotator at startup.

    Expected MongoDB document structure:
        { username, password (plaintext), numero_poste, email, created_at }

    Args:
        mongo_client: Connected MongoDBClient used to fetch the list of available
                      scenarios so the user can select one before logging in.
    """

    def __init__(self, mongo_client: Optional["MongoDBClient"] = None, parent=None):
        super().__init__(parent)
        self._mongo_client = mongo_client
        self._accepted_credentials: Optional[Tuple[str, str]] = None
        self._mode: str = "annotation"   # "annotation" | "verification"
        self._selected_scenario: str = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("VIVE Labeler — Connexion")
        self.setFixedSize(420, 440)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; font-size: 13px; }
            QLabel#subtitle { color: #6c7086; font-size: 11px; }
            QLineEdit {
                background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 6px; padding: 8px; font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #89b4fa; }
            QComboBox {
                background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 6px; padding: 8px; font-size: 13px;
            }
            QComboBox:focus { border: 1px solid #89b4fa; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView {
                background: #313244; color: #cdd6f4;
                selection-background-color: #45475a;
                border: 1px solid #585b70;
            }
            QPushButton {
                background: #89b4fa; color: #1e1e2e; border: none;
                border-radius: 6px; padding: 10px 24px; font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background: #74c7ec; }
            QPushButton:pressed { background: #89dceb; }
            QPushButton#cancel_btn { background: #45475a; color: #cdd6f4; }
            QPushButton#cancel_btn:hover { background: #585b70; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(28, 24, 28, 24)

        # Title
        title = QLabel("Connexion annotateur")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #cdd6f4; margin-bottom: 2px;")
        layout.addWidget(title)

        subtitle = QLabel("Identifiez-vous avec votre nom d'annotateur")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        # Username
        layout.addWidget(QLabel("Nom d'annotateur"))
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("ex : Jawad")
        layout.addWidget(self.username_input)

        # Password
        layout.addWidget(QLabel("Mot de passe"))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("••••••••")
        layout.addWidget(self.password_input)

        # ── Scénario selector ─────────────────────────────────────────
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #313244; margin: 2px 0;")
        layout.addWidget(sep1)

        scenario_label = QLabel("Type de session")
        scenario_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")
        layout.addWidget(scenario_label)

        self._scenario_combo = QComboBox()
        self._scenario_combo.addItem("— Chargement des scénarios… —", "")
        self._scenario_combo.setEnabled(False)
        self._scenario_combo.currentIndexChanged.connect(self._on_scenario_changed)
        layout.addWidget(self._scenario_combo)

        self._load_scenarios()

        # ── Mode selector ─────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244; margin: 2px 0;")
        layout.addWidget(sep)

        mode_label = QLabel("Mode de travail")
        mode_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")
        layout.addWidget(mode_label)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)

        _btn_base = (
            "QPushButton {{ background: {bg}; color: {fg}; border: 2px solid {border}; "
            "border-radius: 6px; padding: 8px 0; font-size: 12px; font-weight: bold; }}"
            "QPushButton:hover {{ background: {hover}; }}"
        )

        self._annot_btn = QPushButton("✏  Annotation")
        self._annot_btn.setCheckable(True)
        self._annot_btn.setChecked(True)
        self._annot_btn.clicked.connect(lambda: self._set_mode("annotation"))
        mode_row.addWidget(self._annot_btn)

        self._verif_btn = QPushButton("🔍  Vérification")
        self._verif_btn.setCheckable(True)
        self._verif_btn.setChecked(False)
        self._verif_btn.clicked.connect(lambda: self._set_mode("verification"))
        mode_row.addWidget(self._verif_btn)

        layout.addLayout(mode_row)
        self._refresh_mode_buttons()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        cancel_btn = QPushButton("Annuler")
        cancel_btn.setObjectName("cancel_btn")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        connect_btn = QPushButton("Se connecter")
        connect_btn.setDefault(True)
        connect_btn.clicked.connect(self._on_connect)
        btn_layout.addWidget(connect_btn)

        layout.addLayout(btn_layout)

        self.username_input.returnPressed.connect(self._on_connect)
        self.password_input.returnPressed.connect(self._on_connect)

    # ------------------------------------------------------------------

    def _load_scenarios(self) -> None:
        """Charge la liste des scénarios depuis MongoDB et peuple le combo."""
        if self._mongo_client is None:
            self._scenario_combo.clear()
            self._scenario_combo.addItem("(aucun scénario — MongoDB non connecté)", "")
            self._scenario_combo.setEnabled(False)
            return

        try:
            scenarios = self._mongo_client.list_scenarios()
        except Exception as exc:
            logger.warning("Impossible de charger les scénarios : %s", exc)
            self._scenario_combo.clear()
            self._scenario_combo.addItem("(erreur de chargement des scénarios)", "")
            self._scenario_combo.setEnabled(False)
            return

        self._scenario_combo.clear()
        actifs = [s for s in scenarios if s.get("actif", True)]
        if not actifs:
            self._scenario_combo.addItem("(aucun scénario actif en BDD)", "")
            self._scenario_combo.setEnabled(False)
            return

        self._scenario_combo.addItem("— Choisir un scénario —", "")
        for s in sorted(actifs, key=lambda x: (x.get("nom") or x.get("description") or "").lower()):
            # Support both field names: "nom" (expected) or fallback to other string fields
            nom = s.get("nom") or s.get("name") or s.get("scenario") or s.get("description") or ""
            desc = s.get("description", "")
            label = f"{nom}  —  {desc}" if (nom and desc and nom != desc) else (nom or desc)
            self._scenario_combo.addItem(label, nom)

        self._scenario_combo.setEnabled(True)

    def _on_scenario_changed(self, index: int) -> None:
        data = self._scenario_combo.currentData()
        self._selected_scenario = data if isinstance(data, str) else ""

    # ------------------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._refresh_mode_buttons()

    def _refresh_mode_buttons(self) -> None:
        is_annot = (self._mode == "annotation")

        active_style = (
            "QPushButton { background: #89b4fa; color: #1e1e2e; border: 2px solid #89b4fa; "
            "border-radius: 6px; padding: 8px 0; font-size: 12px; font-weight: bold; }"
        )
        inactive_style = (
            "QPushButton { background: #313244; color: #a6adc8; border: 2px solid #45475a; "
            "border-radius: 6px; padding: 8px 0; font-size: 12px; font-weight: bold; }"
            "QPushButton:hover { background: #45475a; }"
        )

        self._annot_btn.setStyleSheet(active_style if is_annot else inactive_style)
        self._verif_btn.setStyleSheet(inactive_style if is_annot else active_style)
        self._annot_btn.setChecked(is_annot)
        self._verif_btn.setChecked(not is_annot)

    def _on_connect(self) -> None:
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username:
            QMessageBox.warning(self, "Erreur", "Veuillez saisir votre nom d'annotateur.")
            return
        if not password:
            QMessageBox.warning(self, "Erreur", "Veuillez saisir votre mot de passe.")
            return
        # Relire la valeur courante du combo au moment de valider
        current_data = self._scenario_combo.currentData()
        if isinstance(current_data, str) and current_data:
            self._selected_scenario = current_data
        if not self._selected_scenario:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner un type de session.")
            return

        self._accepted_credentials = (username, password)
        self.accept()

    def get_credentials(self) -> Optional[Tuple[str, str]]:
        """Return (username, password) if dialog was accepted."""
        return self._accepted_credentials

    def get_mode(self) -> str:
        """Return selected mode: 'annotation' or 'verification'."""
        return self._mode

    def get_scenario(self) -> str:
        """Return the selected scenario name (nom from MongoDB scenarios collection)."""
        return self._selected_scenario
