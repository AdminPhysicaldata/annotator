"""Stats widget — affiche le nombre de sessions et les heures par scénario
pour chaque étape du pipeline d'ingestion (inbox, bronze, silver, gold).

Les données sont récupérées depuis le serveur HDD via SFTP dans un thread
de fond ; un bouton Rafraîchir relance le scan à la demande.
"""

import json
import logging
import stat as stat_mod
from typing import Dict, List, Optional, Tuple

import paramiko
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QScrollArea,
    QSizePolicy, QProgressBar,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zone definitions
# ---------------------------------------------------------------------------

ZONES = [
    ("inbox",  "Inbox"),
    ("bronze", "Bronze"),
    ("silver", "Silver"),
    ("gold",   "Gold"),
]

# Couleurs par zone (teinte Catppuccin Mocha)
ZONE_COLORS = {
    "inbox":  "#cba6f7",   # mauve
    "bronze": "#fab387",   # peach
    "silver": "#89dceb",   # sky
    "gold":   "#f9e2af",   # yellow
}

# ---------------------------------------------------------------------------
# SFTP helpers (synchronous, run inside the worker thread)
# ---------------------------------------------------------------------------

def _sftp_listdirs(sftp, path: str) -> List[str]:
    """Retourne les noms de sous-dossiers directs de *path*."""
    try:
        entries = sftp.listdir_attr(path)
        return sorted(e.filename for e in entries if stat_mod.S_ISDIR(e.st_mode))
    except Exception:
        return []


def _sftp_read_json(sftp, path: str) -> Optional[dict]:
    """Lit et décode un fichier JSON distant. Retourne None en cas d'erreur."""
    try:
        with sftp.open(path, "r") as f:
            return json.loads(f.read())
    except Exception:
        return None


def _guess_scenario(session_id: str, metadata: Optional[dict]) -> str:
    """Devine le nom du scénario depuis les métadonnées ou l'id de session."""
    if metadata:
        name = metadata.get("scenario_name") or metadata.get("scenario") or metadata.get("task")
        if name:
            return str(name)
    # Dernier recours : pas de scénario identifiable
    return "Inconnu"


def _duration_seconds(metadata: Optional[dict]) -> float:
    """Extrait la durée en secondes depuis les métadonnées."""
    if not metadata:
        return 0.0
    # Champ direct
    dur = metadata.get("duration_seconds")
    if dur is not None:
        try:
            return float(dur)
        except (TypeError, ValueError):
            pass
    # Calcul depuis start/end
    try:
        from datetime import datetime
        start = metadata.get("start_time", "")
        end = metadata.get("end_time", "")
        if start and end:
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
            t0 = datetime.fromisoformat(start)
            t1 = datetime.fromisoformat(end)
            return (t1 - t0).total_seconds()
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class StatsWorkerThread(QThread):
    """Scanne le serveur HDD SFTP et retourne les stats par zone et scénario.

    Émissions :
        progress(str)              — message de progression
        zone_ready(str, dict)      — (zone_key, {scenario: (count, hours)})
        finished()
        error(str)
    """

    progress = pyqtSignal(str)
    zone_ready = pyqtSignal(str, dict)   # zone_key → {scenario: (count, seconds)}
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        inbox_base: str,
        bronze_base: str,
        silver_base: str,
        gold_base: str,
        parent=None,
    ):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._bases = {
            "inbox":  inbox_base,
            "bronze": bronze_base,
            "silver": silver_base,
            "gold":   gold_base,
        }

    # ------------------------------------------------------------------
    def run(self) -> None:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(
                hostname=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                timeout=15,
                look_for_keys=False,
                allow_agent=False,
            )
            sftp = ssh.open_sftp()
        except Exception as exc:
            self.error.emit(f"Connexion HDD échouée : {exc}")
            return

        try:
            for zone_key, _label in ZONES:
                base = self._bases.get(zone_key, "")
                if not base:
                    self.zone_ready.emit(zone_key, {})
                    continue

                self.progress.emit(f"Scan {zone_key}…")
                stats: Dict[str, List[float]] = {}  # scenario → [total_seconds, count]

                sessions = _sftp_listdirs(sftp, base)
                for session_id in sessions:
                    meta_path = f"{base.rstrip('/')}/{session_id}/metadata.json"
                    metadata = _sftp_read_json(sftp, meta_path)
                    scenario = _guess_scenario(session_id, metadata)
                    dur = _duration_seconds(metadata)
                    if scenario not in stats:
                        stats[scenario] = [0.0, 0]
                    stats[scenario][0] += dur
                    stats[scenario][1] += 1

                # Convertir en (count, hours)
                result = {
                    sc: (int(v[1]), v[0] / 3600.0)
                    for sc, v in stats.items()
                }
                self.zone_ready.emit(zone_key, result)

        except Exception as exc:
            self.error.emit(f"Erreur scan : {exc}")
        finally:
            try:
                sftp.close()
                ssh.close()
            except Exception:
                pass
            self.finished.emit()


# ---------------------------------------------------------------------------
# Stats table for one zone
# ---------------------------------------------------------------------------

class _ZoneTable(QWidget):
    """Tableau compact pour une zone du pipeline."""

    _HEADER_STYLE = """
        QHeaderView::section {{
            background: {bg};
            color: #1e1e2e;
            border: none;
            padding: 5px 8px;
            font-size: 11px;
            font-weight: bold;
        }}
    """
    _TABLE_STYLE = """
        QTableWidget {
            background: #181825;
            color: #cdd6f4;
            border: none;
            gridline-color: #313244;
            font-size: 11px;
            outline: none;
        }
        QTableWidget::item { padding: 4px 8px; border: none; }
        QTableWidget::item:selected { background: #313244; }
        QScrollBar:vertical { width: 6px; background: #181825; }
        QScrollBar::handle:vertical { background: #45475a; border-radius: 3px; }
    """

    def __init__(self, zone_key: str, zone_label: str, parent=None):
        super().__init__(parent)
        self._zone_key = zone_key
        self._zone_label = zone_label
        self._color = ZONE_COLORS.get(zone_key, "#cdd6f4")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # En-tête zone
        header = QFrame()
        header.setFixedHeight(32)
        header.setStyleSheet(f"background: {self._color}; border-radius: 4px 4px 0 0;")
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(10, 0, 10, 0)

        title = QLabel(self._zone_label.upper())
        title.setFont(QFont("", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: #1e1e2e; background: transparent;")
        hlay.addWidget(title)

        self._summary = QLabel("–")
        self._summary.setFont(QFont("", 10))
        self._summary.setStyleSheet("color: #1e1e2e; background: transparent;")
        self._summary.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hlay.addWidget(self._summary, stretch=1)

        layout.addWidget(header)

        # Tableau
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Scénario", "Sessions", "Heures"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(self._TABLE_STYLE)
        self._table.horizontalHeader().setStyleSheet(
            self._HEADER_STYLE.format(bg=self._color)
        )
        self._table.setMinimumHeight(100)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._table)

        # Zone chargement
        self._loading_label = QLabel("Chargement…")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(self._loading_label)
        self._loading_label.hide()

    def set_loading(self, loading: bool) -> None:
        if loading:
            self._table.hide()
            self._loading_label.show()
        else:
            self._loading_label.hide()
            self._table.show()

    def update_data(self, data: Dict[str, Tuple[int, float]]) -> None:
        """Met à jour le tableau avec data = {scenario: (count, hours)}."""
        self._table.setRowCount(0)

        if not data:
            self._summary.setText("0 session — 0 h")
            self.set_loading(False)
            return

        total_sessions = 0
        total_hours = 0.0

        # Trier par nombre de sessions décroissant
        for scenario, (count, hours) in sorted(data.items(), key=lambda x: -x[1][0]):
            row = self._table.rowCount()
            self._table.insertRow(row)

            item_sc = QTableWidgetItem(scenario)
            item_sc.setForeground(QColor(self._color))
            self._table.setItem(row, 0, item_sc)

            item_count = QTableWidgetItem(str(count))
            item_count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 1, item_count)

            item_hours = QTableWidgetItem(f"{hours:.2f} h")
            item_hours.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 2, item_hours)

            total_sessions += count
            total_hours += hours

        self._summary.setText(f"{total_sessions} session(s) — {total_hours:.2f} h")
        self.set_loading(False)

    def set_error(self, msg: str) -> None:
        self._loading_label.setText(f"Erreur : {msg}")
        self._table.hide()
        self._loading_label.show()
        self._summary.setText("–")


# ---------------------------------------------------------------------------
# Main stats widget
# ---------------------------------------------------------------------------

class StatsWidget(QWidget):
    """Onglet Stats : vue d'ensemble du pipeline par scénario."""

    _STYLE = """
        StatsWidget { background: #1e1e2e; }
        QPushButton {
            background: #313244; color: #cdd6f4;
            border: 1px solid #45475a; border-radius: 6px;
            padding: 6px 16px; font-size: 12px;
        }
        QPushButton:hover { background: #45475a; }
        QPushButton:disabled { background: #1e1e2e; color: #45475a; border-color: #313244; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = None   # HDDConfig — set via set_config()
        self._worker: Optional[StatsWorkerThread] = None
        self._build_ui()

    # ------------------------------------------------------------------
    def set_config(self, hdd_config) -> None:
        """Reçoit la HDDConfig depuis le MainWindow."""
        self._config = hdd_config

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setStyleSheet(self._STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Barre du haut : titre + bouton
        top = QHBoxLayout()
        title = QLabel("Pipeline — Statistiques")
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #cdd6f4;")
        top.addWidget(title)
        top.addStretch()

        self._refresh_btn = QPushButton("Rafraîchir")
        self._refresh_btn.setFixedWidth(110)
        self._refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self._refresh_btn)
        root.addLayout(top)

        # Barre de statut
        self._status_label = QLabel("Cliquez sur Rafraîchir pour charger les statistiques.")
        self._status_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        root.addWidget(self._status_label)

        # Scroll area contenant les 4 tableaux
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #1e1e2e; }")

        container = QWidget()
        container.setStyleSheet("background: #1e1e2e;")
        grid = QHBoxLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)

        self._zone_tables: Dict[str, _ZoneTable] = {}
        for zone_key, zone_label in ZONES:
            tbl = _ZoneTable(zone_key, zone_label)
            self._zone_tables[zone_key] = tbl
            grid.addWidget(tbl)

        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Lance le scan des 4 zones dans un thread de fond."""
        if self._config is None:
            self._status_label.setText("Configuration HDD non disponible.")
            return

        if self._worker and self._worker.isRunning():
            return  # déjà en cours

        # Réinitialiser les tableaux
        for tbl in self._zone_tables.values():
            tbl.set_loading(True)

        self._refresh_btn.setEnabled(False)
        self._status_label.setText("Scan en cours…")

        self._worker = StatsWorkerThread(
            host=self._config.host,
            port=self._config.port,
            username=self._config.username,
            password=self._config.password,
            inbox_base=self._config.inbox_base,
            bronze_base=self._config.bronze_base,
            silver_base=self._config.silver_base,
            gold_base=self._config.gold_base,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.zone_ready.connect(self._on_zone_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    # ------------------------------------------------------------------
    def _on_progress(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _on_zone_ready(self, zone_key: str, data: dict) -> None:
        tbl = self._zone_tables.get(zone_key)
        if tbl:
            tbl.update_data(data)

    def _on_finished(self) -> None:
        self._refresh_btn.setEnabled(True)
        self._status_label.setText("Dernière mise à jour : maintenant")
        self._worker = None

    def _on_error(self, msg: str) -> None:
        self._status_label.setText(f"Erreur : {msg}")
        for tbl in self._zone_tables.values():
            tbl.set_error(msg)
        self._refresh_btn.setEnabled(True)
        self._worker = None
