"""Dialog asking whether the session performs (do) or undoes (undo) the scenario."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from typing import Optional


class ScenarioActionDialog(QDialog):
    """Pop-up simple demandant si la session fait ou défait le scénario.

    Returns 'do' or 'undo' via get_action(), or None if the user cancelled.
    """

    def __init__(self, scenario_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Direction du scénario")
        self.setMinimumWidth(480)
        self.setModal(True)
        self._action: Optional[str] = None
        self._setup_ui(scenario_name)

    def _setup_ui(self, scenario_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(28, 24, 28, 24)

        # --- Title ---
        title = QLabel("Sens du scénario")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #cdd6f4;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # --- Scenario name ---
        scen_label = QLabel(f"Scénario : <b>{scenario_name}</b>")
        scen_label.setStyleSheet("color: #89b4fa; font-size: 12px;")
        scen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(scen_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244;")
        layout.addWidget(sep)

        # --- Schema / explanation ---
        schema = QLabel(
            "Cette session <b>effectue-t-elle</b> le scénario ou l'<b>annule-t-elle</b> ?\n\n"
            "  ✅  <b>DO</b>   — le robot <u>réalise</u> la tâche\n"
            "        → /mnt/storage/silver/<i>scénario</i>/<b>do</b>/\n\n"
            "  ↩️  <b>UNDO</b> — le robot <u>remet</u> la scène dans l'état initial\n"
            "        → /mnt/storage/silver/<i>scénario</i>/<b>undo</b>/"
        )
        schema.setTextFormat(Qt.TextFormat.RichText)
        schema.setWordWrap(True)
        schema.setStyleSheet(
            "color: #cdd6f4; font-size: 12px; "
            "background: #181825; border-radius: 8px; padding: 14px;"
        )
        layout.addWidget(schema)

        layout.addSpacing(4)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(16)

        btn_do = QPushButton("✅  DO  — réalise la tâche")
        btn_do.setFixedHeight(48)
        btn_do.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        btn_do.setStyleSheet("""
            QPushButton {
                background: #40a02b;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
            }
            QPushButton:hover { background: #4cb832; }
        """)
        btn_do.clicked.connect(lambda: self._select("do"))

        btn_undo = QPushButton("↩️  UNDO  — remet en place")
        btn_undo.setFixedHeight(48)
        btn_undo.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        btn_undo.setStyleSheet("""
            QPushButton {
                background: #e64553;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
            }
            QPushButton:hover { background: #f05068; }
        """)
        btn_undo.clicked.connect(lambda: self._select("undo"))

        btn_cancel = QPushButton("Annuler")
        btn_cancel.setFixedHeight(36)
        btn_cancel.setStyleSheet("""
            QPushButton {
                background: #313244;
                color: #a6adc8;
                border: none;
                border-radius: 6px;
                padding: 4px 14px;
                font-size: 11px;
            }
            QPushButton:hover { background: #45475a; }
        """)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_do)
        btn_layout.addWidget(btn_undo)
        layout.addLayout(btn_layout)

        cancel_row = QHBoxLayout()
        cancel_row.addStretch()
        cancel_row.addWidget(btn_cancel)
        layout.addLayout(cancel_row)

    def _select(self, action: str) -> None:
        self._action = action
        self.accept()

    def get_action(self) -> Optional[str]:
        """Returns 'do', 'undo', or None if cancelled."""
        return self._action

    @staticmethod
    def ask(scenario_name: str, parent=None) -> Optional[str]:
        """Show the dialog and return 'do', 'undo', or None if cancelled."""
        dlg = ScenarioActionDialog(scenario_name, parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.get_action()
        return None
