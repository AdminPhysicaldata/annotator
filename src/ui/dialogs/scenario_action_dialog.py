"""Dialog asking whether the session performs (do) or resets (reset) the scenario."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QComboBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from typing import Optional


class ScenarioActionDialog(QDialog):
    """Pop-up demandant si la session fait ou remet en place le scénario.

    Affiche une liste déroulante de tous les scénarios disponibles pour
    permettre de corriger le scénario avant validation.

    Returns ('do'|'reset', scenario_name) via get_result(), or (None, None)
    if the user cancelled.
    """

    def __init__(self, scenario_name: str, scenarios: list[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Direction du scénario")
        self.setMinimumWidth(520)
        self.setModal(True)
        self._action: Optional[str] = None
        self._scenarios = scenarios or []
        self._setup_ui(scenario_name)

    def _setup_ui(self, scenario_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 24, 28, 24)

        # --- Title ---
        title = QLabel("Sens du scénario")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #cdd6f4;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        sep_top = QFrame()
        sep_top.setFrameShape(QFrame.Shape.HLine)
        sep_top.setStyleSheet("color: #313244;")
        layout.addWidget(sep_top)

        # --- Scenario dropdown ---
        scen_row = QHBoxLayout()
        scen_row.setSpacing(10)
        scen_lbl = QLabel("Scénario :")
        scen_lbl.setStyleSheet("color: #a6adc8; font-size: 12px; font-weight: bold;")
        scen_lbl.setFixedWidth(80)
        scen_row.addWidget(scen_lbl)

        self._combo = QComboBox()
        self._combo.setStyleSheet("""
            QComboBox {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 12px;
            }
            QComboBox:focus { border-color: #89b4fa; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #1e1e2e;
                color: #cdd6f4;
                selection-background-color: #313244;
            }
        """)

        if self._scenarios:
            for s in self._scenarios:
                self._combo.addItem(s, s)
            # Pré-sélectionner le scénario courant
            idx = self._combo.findData(scenario_name)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
            elif scenario_name:
                self._combo.insertItem(0, scenario_name, scenario_name)
                self._combo.setCurrentIndex(0)
        else:
            # Pas de liste MongoDB — afficher le scénario courant en lecture seule
            self._combo.addItem(scenario_name or "(inconnu)", scenario_name or "")
            self._combo.setEnabled(False)

        scen_row.addWidget(self._combo, stretch=1)
        layout.addLayout(scen_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244;")
        layout.addWidget(sep)

        # --- Schema / explanation ---
        schema = QLabel(
            "Cette session <b>effectue-t-elle</b> le scénario ou le <b>remet-elle</b> en place ?\n\n"
            "  ✅  <b>DO</b>    — le robot <u>réalise</u> la tâche\n"
            "        → /mnt/storage/silver/<i>scénario</i>/<b>do</b>/\n\n"
            "  ↩️  <b>RESET</b> — le robot <u>remet</u> la scène dans l'état initial\n"
            "        → /mnt/storage/silver/<i>scénario</i>/<b>reset</b>/"
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

        btn_undo = QPushButton("↩️  RESET  — remet en place")
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
        btn_undo.clicked.connect(lambda: self._select("reset"))

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

    def get_result(self) -> tuple[Optional[str], str]:
        """Returns (action, scenario_name) where action is 'do', 'reset', or None if cancelled."""
        scenario = self._combo.currentData() or self._combo.currentText() or ""
        return self._action, scenario

    def get_action(self) -> Optional[str]:
        """Returns 'do', 'reset', or None if cancelled. (compat)"""
        return self._action

    @staticmethod
    def ask(scenario_name: str, parent=None) -> Optional[str]:
        """Compat: Show dialog without scenario list, return action only."""
        dlg = ScenarioActionDialog(scenario_name, scenarios=None, parent=parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.get_action()
        return None

    @staticmethod
    def ask_with_scenarios(
        scenario_name: str,
        scenarios: list[str],
        parent=None,
    ) -> tuple[Optional[str], str]:
        """Show dialog with scenario dropdown.

        Returns (action, selected_scenario) where action is 'do'|'reset'|None.
        """
        dlg = ScenarioActionDialog(scenario_name, scenarios=scenarios, parent=parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.get_result()
        return None, scenario_name
