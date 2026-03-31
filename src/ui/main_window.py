"""Main application window — multi-camera annotation workspace.

Layout:
┌──────────────────────────────────────────────────────────────────┐
│ Menu Bar                                                         │
├──────────────────────────────────────────┬──────────────────────┤
│  cam0        │  cam1        │  cam2      │  Label Panel          │
│  [video]     │  [video]     │  [video]   │                      │
│  [3D tracker]│  [3D tracker]│  [3D track]│  Gripper Graph       │
│              │              │            │  (pince1 + pince2)   │
├──────────────────────────────────────────┴──────────────────────┤
│   Annotation Timeline                                            │
├──────────────────────────────────────────────────────────────────┤
│ Status Bar                                                       │
└──────────────────────────────────────────────────────────────────┘
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QFileDialog, QMessageBox,
    QStatusBar, QStackedWidget, QApplication, QPushButton,
    QDialog, QLabel, QDialogButtonBox, QScrollArea,
    QListWidget, QListWidgetItem, QLineEdit, QFrame, QCheckBox,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QSlider, QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt6.QtGui import QAction, QKeySequence, QShortcut, QColor, QFont

from ..core.session_loader import SessionDataLoader
from ..core.session_swap import swap_cameras_on_disk, swap_trackers_on_disk
from ..core.transforms import Transform3D
# from ..core.seqensor_worker import SeqensorWorker
from ..labeling.label_manager import LabelManager, Annotation
from ..labeling.export import AnnotationExporter
from ..storage.mongodb_client import MongoDBClient
from ..storage.nas_client import NASClient, LocalJobFiles, upload_directory_sftp_background, silver_dest_path
from ..storage.spool_client import SpoolListWorker, SpoolDownloadWorker, SpoolBrowseWorker, HddVerificationWorker, HddUploadWorker
from ..queue.rabbitmq_consumer import (
    RabbitMQConsumer, RabbitMQPollerThread, AnnotationJob,
    ScenarioPrefetcher, PrefetchedScenario,
)
from ..utils.config import AppConfig
from ..core.csv_validator import validate_job_csvs
from .widgets.multi_video_widget import MultiVideoWidget
from .widgets.annotation_timeline import AnnotationTimeline
from .widgets.label_panel import LabelPanel
from .widgets.waiting_widget import WaitingWidget
from .widgets.annotation_list_panel import AnnotationListPanel, AnnotationDetailDialog
from .widgets.verification_widget import VerificationWidget
# from .widgets.seqensor_widget import SeqensorWidget
from .dialogs.session_browser_dialog import SessionBrowserDialog
from .dialogs.upload_validation_dialog import UploadValidationDialog
from .dialogs.scenario_label_dialog import ScenarioLabelDialog

logger = logging.getLogger(__name__)


class NASDownloadThread(QThread):
    """Worker thread that downloads all job files from the NAS (SFTP) in parallel."""

    download_finished = pyqtSignal(object)          # LocalJobFiles
    download_progress = pyqtSignal(str)             # legacy status text (unused)
    download_error = pyqtSignal(str)                # error message
    # (label, bytes_done, bytes_total) — emitted from worker threads via Qt queued connection
    file_progress = pyqtSignal(str, int, int)

    def __init__(self, nas_client: NASClient, job: AnnotationJob, parent=None):
        super().__init__(parent)
        self._nas = nas_client
        self._job = job

    def run(self):
        try:
            def _on_file_progress(label: str, done: int, total: int) -> None:
                self.file_progress.emit(label, done, total)

            local_files = self._nas.download_job_parallel(
                self._job, progress_cb=_on_file_progress
            )
            self.download_finished.emit(local_files)
        except Exception as exc:
            logger.error("NAS download failed: %s", exc, exc_info=True)
            self.download_error.emit(str(exc))


class FrameLoaderThread(QThread):
    """Background thread that decodes video frames without blocking the UI.

    Two modes:
    - Scrub mode (default): seek-based decode using jsonl offsets for frame
      accuracy. Only the most recently requested frame is decoded — older
      pending requests are dropped on new arrivals.
    - Playback mode: sequential cap.read() — no seek overhead. Started via
      start_playback(frame_index) and stopped via stop_playback().
    """

    frame_ready = pyqtSignal(int, dict)   # (frame_index, {camera_id: np.ndarray})
    playback_frame = pyqtSignal(int, dict)  # (frame_index, frames) during playback

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QMutex()
        self._pending_frame: Optional[int] = None
        self._session = None
        self._running = True

        # Playback mode state
        self._playback_active = False
        self._playback_start_frame: Optional[int] = None
        self._playback_frame_interval_ms: int = 33  # ~30 FPS default

    def set_session(self, session) -> None:
        with QMutexLocker(self._mutex):
            self._session = session

    def request_frame(self, frame_index: int) -> None:
        """Schedule a scrub-mode frame decode. Replaces any pending request."""
        with QMutexLocker(self._mutex):
            self._playback_active = False
            self._pending_frame = frame_index
        if not self.isRunning():
            self.start()

    def start_playback(self, frame_index: int, fps: float = 30.0) -> None:
        """Switch to playback mode starting at frame_index."""
        interval_ms = max(1, int(1000.0 / fps)) if fps > 0 else 33
        with QMutexLocker(self._mutex):
            self._pending_frame = None
            self._playback_start_frame = frame_index
            self._playback_frame_interval_ms = interval_ms
            self._playback_active = True
        if not self.isRunning():
            self.start()

    def stop_playback(self) -> None:
        """Stop playback mode (return to idle scrub mode)."""
        with QMutexLocker(self._mutex):
            self._playback_active = False
            self._playback_start_frame = None

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
            self._playback_active = False
            self._pending_frame = None

    def run(self) -> None:
        self._running = True
        while self._running:
            # --- Check playback mode first ---
            with QMutexLocker(self._mutex):
                playback = self._playback_active
                start_frame = self._playback_start_frame
                interval_ms = self._playback_frame_interval_ms
                session = self._session

            if playback and session is not None and start_frame is not None:
                # Seek all cameras to the start position once
                try:
                    session.seek_for_playback(start_frame)
                except Exception as exc:
                    logger.error("seek_for_playback failed at frame %d: %s", start_frame, exc)

                current_frame = start_frame
                max_frames = session.frame_count

                # Sequential playback loop — no seek, pure cap.read()
                while self._running:
                    with QMutexLocker(self._mutex):
                        if not self._playback_active:
                            break
                        session = self._session

                    if current_frame >= max_frames:
                        with QMutexLocker(self._mutex):
                            self._playback_active = False
                        break

                    try:
                        frames = session.get_all_frames_sequential()
                        self.playback_frame.emit(current_frame, frames)
                    except Exception as exc:
                        logger.error("Playback decode error frame %d: %s", current_frame, exc)

                    current_frame += 1
                    self.msleep(interval_ms)

                continue  # restart outer loop after playback ends

            # --- Scrub mode ---
            with QMutexLocker(self._mutex):
                frame_index = self._pending_frame
                self._pending_frame = None
                session = self._session

            if frame_index is None or session is None:
                self.msleep(5)
                continue

            try:
                frames = session.get_all_frames(frame_index)
                # Only emit if no newer request has arrived in the meantime
                with QMutexLocker(self._mutex):
                    still_current = (self._pending_frame is None)
                if still_current:
                    self.frame_ready.emit(frame_index, frames)
            except Exception as exc:
                logger.error("FrameLoaderThread error frame %d: %s", frame_index, exc)


class _SpoolScenarioPicker(QDialog):
    """Explorateur de fichiers SFTP pour sélectionner un scénario SPOOL.

    - Navigation par double-clic sur les dossiers
    - Sélection d'un scénario (dossier reconnu comme session) par simple clic
    - Barre de chemin éditable + bouton retour
    - Colonne Nom / Type / Taille
    - Filtre texte sur les scénarios
    """

    _STYLE = """
        QDialog { background: #1e1e2e; }
        QLabel { color: #cdd6f4; font-size: 12px; }
        QTreeWidget {
            background: #181825; color: #cdd6f4;
            border: 1px solid #313244; border-radius: 4px;
            font-size: 12px; outline: none;
        }
        QTreeWidget::item { padding: 3px 4px; }
        QTreeWidget::item:selected { background: #45475a; color: #cdd6f4; }
        QTreeWidget::item:hover { background: #2a2a3e; }
        QHeaderView::section {
            background: #1e1e2e; color: #a6adc8;
            border: none; border-bottom: 1px solid #313244;
            padding: 4px 6px; font-size: 11px;
        }
        QLineEdit {
            background: #313244; color: #cdd6f4;
            border: 1px solid #45475a; border-radius: 4px;
            padding: 5px 8px; font-size: 12px;
        }
        QPushButton {
            background: #89b4fa; color: #1e1e2e; border: none;
            border-radius: 4px; padding: 6px 16px; font-weight: bold;
            font-size: 12px;
        }
        QPushButton:hover { background: #74c7ec; }
        QPushButton:disabled { background: #313244; color: #585b70; }
        QPushButton#secondary {
            background: #313244; color: #cdd6f4; font-weight: normal;
        }
        QPushButton#secondary:hover { background: #45475a; }
        QPushButton#secondary:disabled { background: #1e1e2e; color: #45475a; }
    """

    # Couleurs des types d'entrées
    _COLOR_SCENARIO = QColor("#a6e3a1")   # vert — scénario sélectionnable
    _COLOR_DIR      = QColor("#cdd6f4")   # blanc — dossier de navigation
    _COLOR_FILE     = QColor("#6c7086")   # gris  — fichier (non cliquable)

    def __init__(self, host: str, port: int, username: str, password: str,
                 inbox_base: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SPOOL — Sélection de scénario")
        self.setMinimumSize(720, 500)
        self.resize(860, 580)
        self.setModal(True)
        self.setStyleSheet(self._STYLE)

        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._history: list[str] = []
        self._current_path: str = inbox_base.rstrip("/") or "/"
        self._selected_session_id: Optional[str] = None
        self._selected_inbox_base: str = self._current_path
        self._browse_worker: Optional[SpoolBrowseWorker] = None

        self._build_ui()
        self._navigate_to(self._current_path, push_history=False)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Barre de navigation ──────────────────────────────────────
        nav = QHBoxLayout()
        nav.setSpacing(4)

        self._back_btn = QPushButton("←")
        self._back_btn.setObjectName("secondary")
        self._back_btn.setFixedWidth(34)
        self._back_btn.setToolTip("Dossier précédent")
        self._back_btn.setEnabled(False)
        self._back_btn.clicked.connect(self._go_back)
        nav.addWidget(self._back_btn)

        self._up_btn = QPushButton("↑")
        self._up_btn.setObjectName("secondary")
        self._up_btn.setFixedWidth(34)
        self._up_btn.setToolTip("Dossier parent")
        self._up_btn.setEnabled(False)
        self._up_btn.clicked.connect(self._go_up)
        nav.addWidget(self._up_btn)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Chemin SFTP…")
        self._path_edit.returnPressed.connect(self._on_path_entered)
        nav.addWidget(self._path_edit, stretch=1)

        go_btn = QPushButton("Aller")
        go_btn.setObjectName("secondary")
        go_btn.clicked.connect(self._on_path_entered)
        nav.addWidget(go_btn)

        root.addLayout(nav)

        # ── Filtre ───────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        filter_row.addWidget(QLabel("Filtrer :"))
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Nom du scénario…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter, stretch=1)
        root.addLayout(filter_row)

        # ── Arborescence ─────────────────────────────────────────────
        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Nom", "Type", "Taille"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.setRootIsDecorated(False)
        self._tree.setSortingEnabled(False)
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.itemClicked.connect(self._on_item_clicked)
        root.addWidget(self._tree, stretch=1)

        # ── Status bar ───────────────────────────────────────────────
        self._status = QLabel("Connexion…")
        self._status.setStyleSheet("color: #6c7086; font-size: 11px;")
        root.addWidget(self._status)

        # ── Boutons ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Annuler")
        cancel_btn.setObjectName("secondary")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        self._ok_btn = QPushButton("Télécharger ce scénario")
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._ok_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate_to(self, path: str, push_history: bool = True) -> None:
        path = path.rstrip("/") or "/"

        if push_history and path != self._current_path:
            self._history.append(self._current_path)

        self._current_path = path
        self._path_edit.setText(path)
        self._back_btn.setEnabled(bool(self._history))
        self._up_btn.setEnabled(path != "/")
        self._ok_btn.setEnabled(False)
        self._selected_session_id = None
        self._status.setText("Chargement…")
        self._tree.clear()

        # Placeholder pendant le chargement
        loading = QTreeWidgetItem(["Chargement…", "", ""])
        loading.setForeground(0, QColor("#6c7086"))
        self._tree.addTopLevelItem(loading)

        # Annuler le worker précédent
        if self._browse_worker and self._browse_worker.isRunning():
            try:
                self._browse_worker.listing_ready.disconnect()
                self._browse_worker.error_occurred.disconnect()
            except Exception:
                pass
            self._browse_worker.quit()
            self._browse_worker.wait(300)

        self._browse_worker = SpoolBrowseWorker(
            host=self._host, port=self._port,
            username=self._username, password=self._password,
            path=path, parent=self,
        )
        self._browse_worker.listing_ready.connect(self._on_listing_ready)
        self._browse_worker.error_occurred.connect(self._on_error)
        self._browse_worker.finished.connect(self._browse_worker.deleteLater)
        self._browse_worker.start()

    def _go_back(self) -> None:
        if self._history:
            prev = self._history.pop()
            self._navigate_to(prev, push_history=False)

    def _go_up(self) -> None:
        parent = str(Path(self._current_path).parent)
        self._navigate_to(parent)

    def _on_path_entered(self) -> None:
        self._navigate_to(self._path_edit.text().strip())

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_listing_ready(self, path: str, items: list) -> None:
        if path != self._current_path:
            return

        self._tree.clear()
        self._filter.clear()

        n_dirs = n_scenarios = n_files = 0
        bold = QFont()
        bold.setBold(True)

        for item in items:
            name = item["name"]
            is_dir = item["is_dir"]
            is_scenario = item["is_scenario"]
            size = item["size"]

            if is_dir:
                if is_scenario:
                    icon = "📋"
                    type_label = "Scénario"
                    color = self._COLOR_SCENARIO
                    n_scenarios += 1
                else:
                    icon = "📁"
                    type_label = "Dossier"
                    color = self._COLOR_DIR
                    n_dirs += 1
                size_label = ""
            else:
                icon = "📄"
                type_label = "Fichier"
                color = self._COLOR_FILE
                size_label = self._fmt_size(size)
                n_files += 1

            row = QTreeWidgetItem([f"{icon}  {name}", type_label, size_label])
            row.setData(0, Qt.ItemDataRole.UserRole, {
                "name": name,
                "is_dir": is_dir,
                "is_scenario": is_scenario,
            })
            row.setForeground(0, color)
            row.setForeground(1, color)
            if is_scenario:
                row.setFont(0, bold)
            self._tree.addTopLevelItem(row)

        parts = []
        if n_dirs:
            parts.append(f"{n_dirs} dossier(s)")
        if n_scenarios:
            parts.append(f"{n_scenarios} scénario(s)")
        if n_files:
            parts.append(f"{n_files} fichier(s)")
        self._status.setText("  •  ".join(parts) + f"   —   {path}" if parts else f"Dossier vide  —  {path}")

    def _on_error(self, error: str) -> None:
        self._tree.clear()
        err_item = QTreeWidgetItem([f"⚠  {error}", "", ""])
        err_item.setForeground(0, QColor("#f38ba8"))
        self._tree.addTopLevelItem(err_item)
        self._status.setText(f"Erreur : {error}")

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data["is_scenario"]:
            # Double-clic sur scénario = sélectionner et confirmer
            self._selected_session_id = data["name"]
            self._selected_inbox_base = self._current_path
            self.accept()
        elif data["is_dir"]:
            child = f"{self._current_path.rstrip('/')}/{data['name']}"
            self._navigate_to(child)

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data["is_scenario"]:
            self._selected_session_id = data["name"]
            self._selected_inbox_base = self._current_path
            self._ok_btn.setEnabled(True)
        else:
            self._selected_session_id = None
            self._ok_btn.setEnabled(False)

    def _apply_filter(self, text: str) -> None:
        """Masque les entrées qui ne correspondent pas au filtre."""
        text = text.lower()
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if not data:
                item.setHidden(False)
                continue
            if text:
                match = text in data["name"].lower()
                item.setHidden(not match)
            else:
                item.setHidden(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_size(n: int) -> str:
        if n < 1024:
            return f"{n} B"
        if n < 1024 ** 2:
            return f"{n/1024:.1f} KB"
        if n < 1024 ** 3:
            return f"{n/1024**2:.1f} MB"
        return f"{n/1024**3:.2f} GB"

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def selected_session_id(self) -> Optional[str]:
        return self._selected_session_id

    def selected_inbox_base(self) -> str:
        return self._selected_inbox_base


class _NasPathDialog(QDialog):
    """Explorateur de chemin SFTP pour choisir où déposer un fichier sur le NAS.

    Affiche l'arborescence du NAS, permet de naviguer, et retourne
    le chemin complet du fichier de destination.
    """

    _STYLE = """
        QDialog { background: #1e1e2e; }
        QLabel  { color: #cdd6f4; font-size: 12px; }
        QTreeWidget {
            background: #181825; color: #cdd6f4;
            border: 1px solid #313244; border-radius: 4px; font-size: 12px;
        }
        QTreeWidget::item { padding: 3px 4px; }
        QTreeWidget::item:selected { background: #45475a; }
        QTreeWidget::item:hover    { background: #2a2a3e; }
        QHeaderView::section {
            background: #1e1e2e; color: #a6adc8;
            border: none; border-bottom: 1px solid #313244; padding: 4px 6px; font-size: 11px;
        }
        QLineEdit {
            background: #313244; color: #cdd6f4;
            border: 1px solid #45475a; border-radius: 4px; padding: 5px 8px; font-size: 12px;
        }
        QPushButton {
            background: #89b4fa; color: #1e1e2e; border: none;
            border-radius: 4px; padding: 6px 14px; font-weight: bold; font-size: 12px;
        }
        QPushButton:hover { background: #74c7ec; }
        QPushButton:disabled { background: #313244; color: #585b70; }
        QPushButton#secondary {
            background: #313244; color: #cdd6f4; font-weight: normal;
        }
        QPushButton#secondary:hover { background: #45475a; }
    """

    def __init__(self, host: str, port: int, username: str, password: str,
                 key_path: str, default_path: str, title: str, filename: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(640, 460)
        self.resize(760, 520)
        self.setStyleSheet(self._STYLE)

        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._key_path = key_path
        self._filename = filename
        self._current_path = str(Path(default_path).parent)
        self._history: list[str] = []
        self._browse_worker: Optional[SpoolBrowseWorker] = None

        self._build_ui()
        # Mettre le chemin complet dans la barre de destination
        self._dest_edit.setText(default_path)
        self._navigate_to(self._current_path, push_history=False)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Barre de navigation
        nav = QHBoxLayout()
        nav.setSpacing(4)
        self._back_btn = QPushButton("←")
        self._back_btn.setObjectName("secondary")
        self._back_btn.setFixedWidth(32)
        self._back_btn.setEnabled(False)
        self._back_btn.clicked.connect(self._go_back)
        nav.addWidget(self._back_btn)
        self._up_btn = QPushButton("↑")
        self._up_btn.setObjectName("secondary")
        self._up_btn.setFixedWidth(32)
        self._up_btn.setEnabled(False)
        self._up_btn.clicked.connect(self._go_up)
        nav.addWidget(self._up_btn)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Chemin courant…")
        self._path_edit.returnPressed.connect(self._on_path_entered)
        nav.addWidget(self._path_edit, stretch=1)
        go_btn = QPushButton("Aller")
        go_btn.setObjectName("secondary")
        go_btn.clicked.connect(self._on_path_entered)
        nav.addWidget(go_btn)
        root.addLayout(nav)

        # Arborescence
        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Nom", "Type"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.setRootIsDecorated(False)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.itemClicked.connect(self._on_item_clicked)
        root.addWidget(self._tree, stretch=1)

        # Status
        self._status = QLabel("Connexion…")
        self._status.setStyleSheet("color: #6c7086; font-size: 11px;")
        root.addWidget(self._status)

        # Chemin de destination final
        dest_row = QHBoxLayout()
        dest_row.setSpacing(6)
        dest_row.addWidget(QLabel("Fichier de destination :"))
        self._dest_edit = QLineEdit()
        self._dest_edit.setPlaceholderText(f"/chemin/sur/le/nas/{self._filename}")
        dest_row.addWidget(self._dest_edit, stretch=1)
        root.addLayout(dest_row)

        # Boutons
        btn_row = QHBoxLayout()
        cancel = QPushButton("Annuler")
        cancel.setObjectName("secondary")
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)
        btn_row.addStretch()
        self._ok_btn = QPushButton("Sauvegarder ici")
        self._ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._ok_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _navigate_to(self, path: str, push_history: bool = True) -> None:
        path = path.rstrip("/") or "/"
        if push_history and path != self._current_path:
            self._history.append(self._current_path)
        self._current_path = path
        self._path_edit.setText(path)
        self._back_btn.setEnabled(bool(self._history))
        self._up_btn.setEnabled(path != "/")
        self._status.setText("Chargement…")
        self._tree.clear()

        loading = QTreeWidgetItem(["Chargement…", ""])
        loading.setForeground(0, QColor("#6c7086"))
        self._tree.addTopLevelItem(loading)

        if self._browse_worker and self._browse_worker.isRunning():
            try:
                self._browse_worker.listing_ready.disconnect()
                self._browse_worker.error_occurred.disconnect()
            except Exception:
                pass
            self._browse_worker.quit()
            self._browse_worker.wait(300)

        self._browse_worker = SpoolBrowseWorker(
            host=self._host, port=self._port,
            username=self._username, password=self._password,
            path=path, parent=self,
        )
        self._browse_worker.listing_ready.connect(self._on_listing_ready)
        self._browse_worker.error_occurred.connect(self._on_error)
        self._browse_worker.finished.connect(self._browse_worker.deleteLater)
        self._browse_worker.start()

    def _go_back(self) -> None:
        if self._history:
            self._navigate_to(self._history.pop(), push_history=False)

    def _go_up(self) -> None:
        self._navigate_to(str(Path(self._current_path).parent))

    def _on_path_entered(self) -> None:
        self._navigate_to(self._path_edit.text().strip())

    def _on_listing_ready(self, path: str, items: list) -> None:
        if path != self._current_path:
            return
        self._tree.clear()
        dir_color  = QColor("#cdd6f4")
        file_color = QColor("#6c7086")
        for item in items:
            icon = "📁" if item["is_dir"] else "📄"
            row = QTreeWidgetItem([f"{icon}  {item['name']}", "Dossier" if item["is_dir"] else "Fichier"])
            row.setData(0, Qt.ItemDataRole.UserRole, item)
            row.setForeground(0, dir_color if item["is_dir"] else file_color)
            row.setForeground(1, dir_color if item["is_dir"] else file_color)
            self._tree.addTopLevelItem(row)
        self._status.setText(f"{len(items)} élément(s)  —  {path}")

    def _on_error(self, error: str) -> None:
        self._tree.clear()
        err = QTreeWidgetItem([f"⚠  {error}", ""])
        err.setForeground(0, QColor("#f38ba8"))
        self._tree.addTopLevelItem(err)
        self._status.setText(f"Erreur : {error}")

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data["is_dir"]:
            child = f"{self._current_path.rstrip('/')}/{data['name']}"
            self._navigate_to(child)

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data["is_dir"]:
            # Mettre à jour le chemin de destination quand on navigue
            dest = f"{self._current_path.rstrip('/')}/{data['name']}/{self._filename}"
            self._dest_edit.setText(dest)

    def destination_path(self) -> str:
        return self._dest_edit.text().strip()

    @classmethod
    def get_path(cls, parent, host, port, username, password, key_path,
                 default_path, title, filename) -> tuple[str, bool]:
        dlg = cls(host=host, port=port, username=username, password=password,
                  key_path=key_path, default_path=default_path,
                  title=title, filename=filename, parent=parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.destination_path(), True
        return "", False


class _SaveDestinationDialog(QDialog):
    """Dialogue de choix de destination pour la sauvegarde des annotations.

    Propose deux options :
      - Enregistrement local  → QFileDialog natif
      - Envoi vers le NAS     → désactivé si le NAS est inaccessible
    """

    # Choix retournés
    LOCAL = "local"
    NAS   = "nas"

    _STYLE = """
        QDialog { background: #1e1e2e; }
        QLabel  { color: #cdd6f4; font-size: 13px; }
        QLabel#hint { color: #a6adc8; font-size: 11px; }
        QPushButton {
            background: #313244; color: #cdd6f4; border: none;
            border-radius: 6px; padding: 14px 20px;
            font-size: 13px; text-align: left;
        }
        QPushButton:hover:enabled  { background: #45475a; }
        QPushButton:disabled       { background: #1e1e2e; color: #45475a; }
        QPushButton#cancel_btn {
            background: transparent; color: #6c7086;
            padding: 6px 12px; font-size: 12px;
        }
        QPushButton#cancel_btn:hover { color: #cdd6f4; }
    """

    def __init__(self, nas_host: str, nas_port: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sauvegarder les annotations")
        self.setModal(True)
        self.setFixedWidth(440)
        self.setStyleSheet(self._STYLE)

        self._nas_host = nas_host
        self._nas_port = nas_port
        self._choice: Optional[str] = None
        self._nas_ok: Optional[bool] = None   # None = vérification en cours

        self._build_ui()
        self._start_nas_check()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 16)
        root.setSpacing(12)

        title = QLabel("Où sauvegarder les annotations ?")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #cdd6f4;")
        root.addWidget(title)

        # ── Option Local ────────────────────────────────────────────
        self._local_btn = QPushButton()
        self._local_btn.clicked.connect(self._pick_local)
        root.addWidget(self._local_btn)
        self._refresh_local_btn()

        # ── Option NAS ──────────────────────────────────────────────
        self._nas_btn = QPushButton()
        self._nas_btn.setEnabled(False)   # activé après vérification
        self._nas_btn.clicked.connect(self._pick_nas)
        root.addWidget(self._nas_btn)
        self._refresh_nas_btn()

        # ── Annuler ─────────────────────────────────────────────────
        cancel_btn = QPushButton("Annuler")
        cancel_btn.setObjectName("cancel_btn")
        cancel_btn.clicked.connect(self.reject)
        root.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def _refresh_local_btn(self) -> None:
        self._local_btn.setText(
            "💾  Enregistrer en local\n"
            "     Choisir l'emplacement sur cet ordinateur"
        )

    def _refresh_nas_btn(self) -> None:
        if self._nas_ok is None:
            status = "⏳  Vérification de la connexion…"
            color = "#6c7086"
        elif self._nas_ok:
            status = f"✅  {self._nas_host}  —  disponible"
            color = "#a6e3a1"
        else:
            status = f"❌  {self._nas_host}  —  inaccessible"
            color = "#f38ba8"

        self._nas_btn.setText(
            f"🗄  Envoyer vers le NAS\n"
            f"     {status}"
        )
        self._nas_btn.setStyleSheet(
            f"QPushButton {{ background: #1e1e2e; color: #cdd6f4; border: 1px solid #313244; "
            f"border-radius: 6px; padding: 14px 20px; font-size: 13px; text-align: left; }}"
            f"QPushButton:hover:enabled {{ background: #313244; }}"
            f"QPushButton:disabled {{ background: #1a1a27; color: #45475a; }}"
        )

    # ------------------------------------------------------------------
    # NAS check  (worker thread léger)
    # ------------------------------------------------------------------

    def _start_nas_check(self) -> None:
        import socket

        class _NasChecker(QThread):
            result = pyqtSignal(bool)
            def __init__(self, host, port):
                super().__init__()
                self._h, self._p = host, port
            def run(self):
                try:
                    with socket.create_connection((self._h, self._p), timeout=4):
                        self.result.emit(True)
                except Exception:
                    self.result.emit(False)

        self._checker = _NasChecker(self._nas_host, self._nas_port)
        self._checker.result.connect(self._on_nas_check_done)
        self._checker.finished.connect(self._checker.deleteLater)
        self._checker.start()

    def _on_nas_check_done(self, ok: bool) -> None:
        self._nas_ok = ok
        self._nas_btn.setEnabled(ok)
        self._refresh_nas_btn()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pick_local(self) -> None:
        self._choice = self.LOCAL
        self.accept()

    def _pick_nas(self) -> None:
        self._choice = self.NAS
        self.accept()

    def choice(self) -> Optional[str]:
        return self._choice


class MainWindow(QMainWindow):
    """Main application window for multi-camera annotation."""

    def __init__(self, config: AppConfig, session_dir: str = None, mongo_client: MongoDBClient = None, annotator_name: str = "", initial_mode: str = "annotation"):
        super().__init__()
        self.config = config
        self.mongo_client = mongo_client
        self._initial_mode = initial_mode  # "annotation" | "verification"
        # Pre-fill annotator from login if not already set in config
        if annotator_name and not self.config.annotator:
            self.config.annotator = annotator_name
        self.session: Optional[SessionDataLoader] = None
        self.label_manager = LabelManager()

        # Playback state
        self.current_frame_index = 0
        self.is_playing = False

        # camera_id -> tracker_name mapping, populated at session load
        self._tracker_map: dict = {}

        # Currently selected segment (for the new split-and-label workflow)
        self._selected_segment: Optional["Annotation"] = None

        # Temporal flow: variable step size for arrow key navigation
        # Levels: 1, 5, 10, 30, 60, 150, 300 frames per step
        self._step_levels = [1, 5, 10, 30, 60, 150, 300]
        self._step_level_index = 0  # starts at 1 frame

        # Worker threads (kept as attributes so they aren't GC'd)
        self._poller_thread: Optional[RabbitMQPollerThread] = None
        self._download_thread: Optional[NASDownloadThread] = None
        self._spool_list_worker: Optional[SpoolListWorker] = None
        self._spool_download_worker: Optional[SpoolDownloadWorker] = None
        self._annotation_hdd_worker: Optional[HddVerificationWorker] = None
        self._upload_thread = None
        self._prefetcher: Optional[ScenarioPrefetcher] = None
        self._hdd_verification_worker: Optional[HddVerificationWorker] = None
        self._hdd_upload_worker: Optional[HddUploadWorker] = None
        self._hdd_current_session_id: Optional[str] = None  # session_id de la session en cours de vérification

        # Prefetch : téléchargement de la session suivante pendant qu'on vérifie la courante
        self._hdd_prefetch_worker: Optional[HddVerificationWorker] = None
        self._hdd_prefetched: Optional[tuple] = None          # (session_id, LocalJobFiles) prêt à charger
        self._hdd_waiting_for_prefetch: bool = False          # validate/reject en attente du prefetch
        self._hdd_session_decided: bool = False               # True si la session courante a été validée/rejetée

        # Async frame loader — decodes video frames off the main thread
        self._frame_loader = FrameLoaderThread(self)
        self._frame_loader.frame_ready.connect(self._on_frame_ready)
        self._frame_loader.playback_frame.connect(self._on_playback_frame)

        # Debounce timer: collapses rapid scrubbing into a single decode call.
        # 30ms = max one decode per visual frame at 30fps, prevents queue buildup.
        self._frame_debounce = QTimer(self)
        self._frame_debounce.setSingleShot(True)
        self._frame_debounce.setInterval(30)
        self._frame_debounce.timeout.connect(self._flush_frame_request)
        self._pending_frame_request: Optional[int] = None
        # # Thread de segmentation automatique (fluxseq). Un seul worker actif à la fois ;
        # # le précédent est arrêté proprement avant de lancer le suivant.
        # self._seqensor_worker: Optional[SeqensorWorker] = None

        # Job courant (pour pouvoir déplacer la source sur HDFS lors d'un rejet)
        self._current_job: Optional[AnnotationJob] = None

        # Background upload subprocesses (fire-and-forget, tracked for logging)
        self._bg_upload_procs: list[subprocess.Popen] = []
        self._bg_upload_timer = QTimer(self)
        self._bg_upload_timer.setInterval(2000)
        self._bg_upload_timer.timeout.connect(self._poll_background_uploads)

        # # Cache des segments auto-détectés par le SeqensorWorker pour la session courante.
        # # Réinitialisé à chaque nouvelle session dans _launch_seqensor().
        # self._seqensor_segments: list = []

        # Setup UI
        self._setup_ui()
        self._create_menus()
        self._create_statusbar()
        self._setup_shortcuts()
        self._initialize_default_labels()

        # Apply initial mode chosen at login (before any session is loaded)
        if self._initial_mode == "verification":
            self._mode_btn.setChecked(True)

        # Auto-load session if provided, otherwise show waiting screen (no auto-polling)
        if session_dir:
            QTimer.singleShot(100, lambda: self._load_session(session_dir))
        else:
            QTimer.singleShot(100, self._show_waiting_screen)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle("VIVE Labeler — Multi-Camera Annotation")
        self.resize(1920, 1100)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background: #1e1e2e; }
            QSplitter::handle { background: #313244; width: 3px; height: 3px; }
            QMenuBar { background: #181825; color: #cdd6f4; border-bottom: 1px solid #313244; }
            QMenuBar::item:selected { background: #313244; }
            QMenu { background: #1e1e2e; color: #cdd6f4; border: 1px solid #313244; }
            QMenu::item:selected { background: #313244; }
            QStatusBar { background: #181825; color: #a6adc8; border-top: 1px solid #313244; font-size: 11px; }
        """)

        # Stacked widget: page 0 = waiting, page 1 = annotation workspace
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # --- Page 0: Waiting widget ---
        self.waiting_widget = WaitingWidget()
        self.waiting_widget.skip_requested.connect(self._on_skip_job)
        self.waiting_widget.load_from_nas_requested.connect(self._on_load_from_nas)
        self.waiting_widget.load_from_spool_requested.connect(self._on_load_from_spool)
        self.stack.addWidget(self.waiting_widget)

        # --- Page 1: Annotation workspace ---
        workspace = QWidget()
        main_layout = QVBoxLayout(workspace)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Top: content splitter (cameras+trackers | right panel) - full height
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: multi-video with per-camera 3D trackers + timeline below
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.multi_video = MultiVideoWidget()
        self.multi_video.swap_requested.connect(self._on_camera_swap_requested)
        self.multi_video.rotate_requested.connect(self._on_camera_rotate_requested)
        self.multi_video.tracker_swap_requested.connect(self._on_tracker_swap_requested)
        left_layout.addWidget(self.multi_video, stretch=1)

        # Timeline below cameras (height ajustée dynamiquement selon les grippers)
        self.timeline = AnnotationTimeline(label_manager=self.label_manager)
        self.timeline.frame_changed.connect(self._on_frame_changed)
        self.timeline.play_toggled.connect(self._on_play_toggled)
        self.timeline.annotation_clicked.connect(self._on_timeline_annotation_clicked)
        self.timeline.segment_selected.connect(self._on_segment_selected)
        self.timeline.setFixedHeight(110)
        left_layout.addWidget(self.timeline)

        content_splitter.addWidget(left_panel)

        # Right: label panel + annotations + buttons stacked vertically
        right_panel = QSplitter(Qt.Orientation.Vertical)

        _is_chef = self.mongo_client.is_chef if self.mongo_client else False
        self.label_panel = LabelPanel(self.label_manager, is_chef=_is_chef)
        self.label_panel.label_selected.connect(self._on_frame_annotation)
        self.label_panel.interval_annotation_requested.connect(self._on_interval_annotation)
        self.label_panel.manage_labels_requested.connect(self._open_scenario_label_manager)
        right_panel.addWidget(self.label_panel)

        # Annotation list panel (now in right panel)
        self.annotation_list_panel = AnnotationListPanel(self.label_manager)
        self.annotation_list_panel.annotation_selected.connect(self._goto_frame)
        self.annotation_list_panel.annotations_changed.connect(self._on_annotations_changed)
        right_panel.addWidget(self.annotation_list_panel)

        # Scenario name input + buttons container at bottom of right panel
        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout(buttons_widget)  # Vertical layout for buttons
        buttons_layout.setContentsMargins(4, 4, 4, 4)
        buttons_layout.setSpacing(4)

        _input_style = """
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
        """

        # Annotator name input
        annotator_container = QWidget()
        annotator_layout = QVBoxLayout(annotator_container)
        annotator_layout.setContentsMargins(0, 0, 0, 0)
        annotator_layout.setSpacing(2)

        annotator_label = QLabel("Annotateur:")
        annotator_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")
        annotator_layout.addWidget(annotator_label)

        self.annotator_input = QLineEdit()
        self.annotator_input.setPlaceholderText("Votre prénom…")
        self.annotator_input.setStyleSheet(_input_style)
        self.annotator_input.setText(self.config.annotator)
        self.annotator_input.textChanged.connect(self._on_annotator_changed)
        annotator_layout.addWidget(self.annotator_input)
        buttons_layout.addWidget(annotator_container)

        # Scenario name input
        scenario_container = QWidget()
        scenario_layout = QVBoxLayout(scenario_container)
        scenario_layout.setContentsMargins(0, 0, 0, 0)
        scenario_layout.setSpacing(2)

        scenario_label = QLabel("Nom du scénario:")
        scenario_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")
        scenario_layout.addWidget(scenario_label)

        self.scenario_input = QLineEdit()
        self.scenario_input.setPlaceholderText("Ex: Goblets, Jean, Serviettes, Balles...")
        self.scenario_input.setStyleSheet(_input_style)
        scenario_layout.addWidget(self.scenario_input)
        buttons_layout.addWidget(scenario_container)

        self.upload_btn = QPushButton("⬆ Upload")
        self.upload_btn.setMinimumHeight(40)
        self.upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #40a02b;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #4cb832; }
            QPushButton:pressed { background-color: #389926; }
            QPushButton:disabled { background-color: #585b70; color: #a6adc8; }
        """)
        self.upload_btn.clicked.connect(self._upload_to_hdfs)
        buttons_layout.addWidget(self.upload_btn)

        self.reject_btn = QPushButton("✕ Rejeter")
        self.reject_btn.setMinimumHeight(40)
        self.reject_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reject_btn.setStyleSheet("""
            QPushButton {
                background-color: #e64553;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #f04a5a; }
            QPushButton:pressed { background-color: #c73545; }
            QPushButton:disabled { background-color: #585b70; color: #a6adc8; }
        """)
        self.reject_btn.clicked.connect(self._reject_session)
        buttons_layout.addWidget(self.reject_btn)

        self.delete_session_btn = QPushButton("🗑 Supprimer la session")
        self.delete_session_btn.setMinimumHeight(40)
        self.delete_session_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_session_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #f38ba8;
                border: 1px solid #f38ba8;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3d2030; border-color: #f38ba8; }
            QPushButton:pressed { background-color: #2a1520; }
            QPushButton:disabled { background-color: #313244; color: #585b70; border-color: #585b70; }
        """)
        self.delete_session_btn.clicked.connect(self._delete_session)
        buttons_layout.addWidget(self.delete_session_btn)

        buttons_widget.setFixedHeight(260)  # annotator + scenario + upload + reject + delete + margins
        right_panel.addWidget(buttons_widget)

        # # Panneau Seqensor : liste des segments auto-détectés.
        # self.seqensor_widget = SeqensorWidget()
        # self.seqensor_widget.jump_to_frame.connect(self._goto_frame)
        # self.seqensor_widget.apply_segment.connect(self._on_seqensor_apply_segment)
        # right_panel.addWidget(self.seqensor_widget)

        right_panel.setSizes([250, 250, 200, 160])  # Labels, Gripper, Annotations, Scenario+Buttons
        right_panel.setMinimumWidth(280)
        content_splitter.addWidget(right_panel)

        content_splitter.setSizes([1560, 360])
        main_layout.addWidget(content_splitter, stretch=1)

        self.stack.addWidget(workspace)

        # --- Page 2: Verification workspace ---
        self.verification_widget = VerificationWidget()
        self.verification_widget.validated.connect(self._on_verification_validate)
        self.verification_widget.rejected.connect(self._on_verification_reject)
        self.verification_widget.swap_requested.connect(self._on_camera_swap_requested)
        self.verification_widget.tracker_swap_requested.connect(self._on_tracker_swap_requested)
        self.stack.addWidget(self.verification_widget)

        # Start on waiting page
        self.stack.setCurrentIndex(0)

    def _create_menus(self) -> None:
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        open_session = QAction("&Open Session Directory...", self)
        open_session.setShortcut(QKeySequence("Ctrl+O"))
        open_session.triggered.connect(self._open_session_dialog)
        file_menu.addAction(open_session)

        file_menu.addSeparator()

        save_action = QAction("&Save Annotations...", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save_annotations)
        file_menu.addAction(save_action)

        load_action = QAction("&Load Annotations...", self)
        load_action.triggered.connect(self._load_annotations)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("&Export Annotations")
        for fmt, label in [("json", "As &JSON..."), ("csv", "As &CSV..."), ("lerobot", "As &LeRobot Format...")]:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, f=fmt: self._export_annotations(f))
            export_menu.addAction(action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View
        view_menu = menubar.addMenu("&View")


        # Options
        options_menu = menubar.addMenu("&Options")

        search_session_action = QAction("&Rechercher une session…", self)
        search_session_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        search_session_action.triggered.connect(self._open_session_browser)
        options_menu.addAction(search_session_action)

        # Help
        help_menu = menubar.addMenu("&Help")
        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_statusbar(self) -> None:
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready — Open a session directory to begin")

        # Mode switch button (Annotation ↔ Vérification)
        self._mode_btn = QPushButton("🔍 Mode Vérification")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setChecked(False)
        self._mode_btn.setFixedHeight(22)
        self._mode_btn.setStyleSheet(
            "QPushButton { background: #313244; color: #cdd6f4; border: none; "
            "border-radius: 4px; padding: 0 10px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background: #45475a; }"
            "QPushButton:checked { background: #89b4fa; color: #1e1e2e; }"
            "QPushButton:checked:hover { background: #74c7ec; }"
        )
        self._mode_btn.toggled.connect(self._on_mode_toggled)
        self.statusbar.addPermanentWidget(self._mode_btn)

        # Permanent right-side label showing current step size
        self._step_label = QLabel("Pas : 1 frame")
        self._step_label.setStyleSheet(
            "color: #89b4fa; font-family: Courier; font-size: 11px; padding: 0 8px;"
        )
        self.statusbar.addPermanentWidget(self._step_label)
        self._refresh_step_label()

    def _setup_shortcuts(self) -> None:
        # Playback
        self._sc_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_playback)

        # Arrow Left/Right: navigate by current step size
        self._sc_left  = QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-self._current_step()))
        self._sc_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(self._current_step()))

        # Arrow Up/Down: change temporal flow (step size)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, self._increase_step)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, self._decrease_step)

        # In/Out
        QShortcut(QKeySequence(Qt.Key.Key_I), self, self._shortcut_set_in)
        QShortcut(QKeySequence(Qt.Key.Key_O), self, self._shortcut_set_out)

        # Home/End
        self._sc_home = QShortcut(QKeySequence(Qt.Key.Key_Home), self, lambda: self._goto_frame(0))
        self._sc_end  = QShortcut(QKeySequence(Qt.Key.Key_End), self, self._goto_last_frame)

        # A: coupe le segment main gauche sous le playhead
        # Z: coupe le segment main droite sous le playhead
        QShortcut(QKeySequence(Qt.Key.Key_A), self, lambda: self._shortcut_split_hand("left"))
        QShortcut(QKeySequence(Qt.Key.Key_Z), self, lambda: self._shortcut_split_hand("right"))

        # Raccourcis numériques pour assigner un label (initialisés vides, remplis après chargement)
        self._label_shortcuts: list[QShortcut] = []

    def _register_label_shortcuts(self) -> None:
        """Enregistre les touches 1-9 pour assigner rapidement un label au segment sélectionné."""
        from ..labeling.label_manager import UNLABELED_LABEL_ID

        # Supprimer les anciens raccourcis
        for sc in self._label_shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._label_shortcuts.clear()

        # Labels visibles (hors __unlabeled__), dans l'ordre d'affichage
        labels = [l for l in self.label_manager.labels.values() if l.id != UNLABELED_LABEL_ID]

        keys = [
            Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5,
            Qt.Key.Key_6, Qt.Key.Key_7, Qt.Key.Key_8, Qt.Key.Key_9, Qt.Key.Key_0,
        ]
        key_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        shortcut_map: dict[str, str] = {}  # label_id -> touche affichée
        for i, label in enumerate(labels[:len(keys)]):
            label_id = label.id
            sc = QShortcut(QKeySequence(keys[i]), self, lambda lid=label_id: self._on_interval_annotation(lid))
            self._label_shortcuts.append(sc)
            shortcut_map[label_id] = key_names[i]

        self.label_panel.set_label_shortcuts(shortcut_map)


    def _initialize_default_labels(self) -> None:
        """Charge les labels par défaut depuis la config (fallback uniquement)."""
        for label_name in self.config.labeling.default_labels:
            self.label_manager.add_label(name=label_name)
        self.label_panel.refresh_labels()
        self._register_label_shortcuts()

    def _load_labels_from_scenario(self, scenario_name: str) -> bool:
        """Charge les labels du scénario depuis MongoDB dans le LabelManager.

        Retourne True si des labels ont été chargés, False sinon (fallback sur defaults).
        """
        if not self.mongo_client or not scenario_name:
            return False
        try:
            db_labels = self.mongo_client.get_scenario_labels(scenario_name)
        except Exception as exc:
            logger.warning("Impossible de charger les labels du scénario '%s': %s", scenario_name, exc)
            return False

        if not db_labels:
            logger.info("Aucun label défini en BDD pour le scénario '%s'", scenario_name)
            return False

        # Réinitialise le LabelManager avec les labels de la BDD
        self.label_manager.labels.clear()
        self.label_manager.annotations.clear()
        self.label_manager._annotations_by_frame.clear()
        self.label_manager._annotations_by_label.clear()

        for lbl in db_labels:
            if isinstance(lbl, str):
                name, color, description = lbl.strip(), "#89b4fa", ""
            else:
                name = lbl.get("name", "").strip()
                color = lbl.get("color", "#89b4fa")
                description = lbl.get("description", "")
            if not name:
                continue
            self.label_manager.add_label(name=name, color=color, description=description)

        self.label_panel.refresh_labels()
        self._register_label_shortcuts()
        logger.info(
            "Labels chargés depuis la BDD pour '%s': %s",
            scenario_name,
            [lbl if isinstance(lbl, str) else lbl.get("name") for lbl in db_labels],
        )
        return True

    def _open_scenario_label_manager(self) -> None:
        """Ouvre le dialogue de gestion des labels (réservé aux chefs)."""
        if not self.mongo_client or not self.mongo_client.is_chef:
            QMessageBox.warning(self, "Accès refusé", "Seuls les chefs peuvent gérer les labels.")
            return

        scenario_name = self.scenario_input.text().strip() if hasattr(self, "scenario_input") else ""
        if not scenario_name:
            QMessageBox.warning(
                self,
                "Scénario requis",
                "Veuillez d'abord indiquer le nom du scénario dans le champ prévu.",
            )
            return

        dlg = ScenarioLabelDialog(self.mongo_client, scenario_name, self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            # Recharge les labels en local depuis la BDD
            self._load_labels_from_scenario(scenario_name)
            self.timeline.refresh_annotations()
            self.statusbar.showMessage(
                f"Labels du scénario « {scenario_name} » rechargés depuis la base de données."
            )

    # ------------------------------------------------------------------
    # Job Polling (RabbitMQ) + Prefetch
    # ------------------------------------------------------------------

    def _make_rabbitmq_consumer(self) -> RabbitMQConsumer:
        return RabbitMQConsumer(
            host=self.config.rabbitmq.host,
            port=self.config.rabbitmq.port,
            username=self.config.rabbitmq.username,
            password=self.config.rabbitmq.password,
            queue_name=self.config.rabbitmq.queue_name,
            virtual_host=self.config.rabbitmq.virtual_host,
        )

    def _make_nas_client(self) -> NASClient:
        return NASClient(
            host=self.config.nas.host,
            port=self.config.nas.port,
            username=self.config.nas.username,
            password=self.config.nas.password or None,
            key_path=self.config.nas.key_path,
            local_dir=Path(self.config.nas.local_dir) if self.config.nas.local_dir else None,
        )


    def _start_job_polling(self) -> None:
        """Start polling RabbitMQ for an annotation job.

        If the prefetcher has already pre-downloaded the next scenario this
        transitions instantly (no waiting page).  Otherwise falls back to the
        normal polling + download flow.
        """
        # Fast path: prefetched scenario already sitting in the buffer
        if self._prefetcher is not None and self._prefetcher.ready_event.is_set():
            scenario = self._prefetcher.consume()
            self._stop_prefetcher()
            if scenario is not None:
                logger.info("Using prefetched scenario: %s", scenario.session_dir)
                self._activate_prefetched_scenario(scenario)
                return

        # Slow path: nothing ready yet — show waiting screen
        self.stack.setCurrentIndex(0)
        self.waiting_widget.start_animation()
        self.waiting_widget.set_status("Connexion à RabbitMQ...")
        self.statusbar.showMessage("En attente d'un job d'annotation...")

        # If a prefetcher is already running (but not ready), hook into it
        # instead of launching a new polling loop.
        if self._prefetcher is not None and self._prefetcher.isRunning():
            self._prefetcher.prefetch_status.connect(self._on_prefetch_status_waiting)
            self._prefetcher.prefetch_ready.connect(self._on_prefetch_ready_from_waiting)
            self._prefetcher.prefetch_error.connect(self._on_download_error)
            return

        # Brand-new poll: launch a poller thread
        consumer = self._make_rabbitmq_consumer()
        self._poller_thread = RabbitMQPollerThread(consumer, poll_interval_s=5.0, parent=self)
        self._poller_thread.job_received.connect(self._on_job_received)
        self._poller_thread.poll_status.connect(self._on_poll_status)
        self._poller_thread.error_occurred.connect(self._on_poll_error)
        self._poller_thread.start()

    # -- Signals from the plain poller (waiting screen) --

    def _on_poll_status(self, status: str, detail: str) -> None:
        self.waiting_widget.set_status(status)
        self.waiting_widget.set_queue_info(detail)
        self.statusbar.showMessage(status)

    def _on_poll_error(self, error: str) -> None:
        self.waiting_widget.set_status(f"Erreur : {error}")
        self.statusbar.showMessage(f"Erreur RabbitMQ : {error}")

    def _on_job_received(self, job: AnnotationJob) -> None:
        """A job was fetched from RabbitMQ — start downloading from HDFS."""
        self._current_job = job
        logger.info("Job received, starting HDFS download")
        self.waiting_widget.set_status("Téléchargement des données depuis HDFS...")
        self.waiting_widget.set_queue_info("")
        self.statusbar.showMessage("Téléchargement des fichiers du job...")
        self.waiting_widget.reset_file_progress()
        self.waiting_widget.show_skip_button()

        self._download_thread = NASDownloadThread(self._make_nas_client(), job, parent=self)
        self._download_thread.download_progress.connect(self._on_download_progress)
        self._download_thread.file_progress.connect(self.waiting_widget.update_file_progress)
        self._download_thread.download_finished.connect(self._on_download_finished)
        self._download_thread.download_error.connect(self._on_download_error)
        self._download_thread.start()

    def _on_download_progress(self, text: str) -> None:
        self.waiting_widget.set_status(text)
        self.statusbar.showMessage(text)

    def _on_download_finished(self, local_files: LocalJobFiles) -> None:
        """All files downloaded — validate CSVs, then load the session and switch to workspace."""
        logger.info("HDFS download complete, loading session from %s", local_files.tracker.parent)
        self.waiting_widget.stop_animation()

        # --- Scan d'intégrité des CSV ---
        self.waiting_widget.set_status("Vérification de l'intégrité des données...")
        self.statusbar.showMessage("Validation des fichiers CSV...")

        report = validate_job_csvs(local_files)

        # Erreurs fatales : impossible de charger la session
        if report.is_fatal:
            logger.error("CSV validation — erreurs fatales : %s", report.fatal_errors)
            self.waiting_widget.set_status("Données invalides — job rejeté")
            self.statusbar.showMessage("Erreur : données CSV invalides")
            QMessageBox.critical(
                self,
                "Données invalides",
                "Les fichiers CSV sont invalides et ne peuvent pas être chargés :\n\n"
                + "\n".join(f"  • {e}" for e in report.fatal_errors),
            )
            return

        # Avertissements : données incomplètes, laisser l'utilisateur choisir
        if report.warnings:
            logger.warning("CSV validation — avertissements : %s", report.warnings)
            warn_text = (
                "⚠ Données incomplètes détectées :\n"
                + "\n".join(f"  • {w}" for w in report.warnings)
                + "\n\nVous pouvez charger la session quand même ou la rejeter."
            )

            # Dialogue principal : charger ou rejeter ?
            choice = QMessageBox(self)
            choice.setWindowTitle("Données incomplètes")
            choice.setText(
                "<b>⚠ Des données incomplètes ont été détectées.</b><br>"
                "La session peut quand même être chargée et annotée."
            )
            choice.setInformativeText("\n".join(f"  • {w}" for w in report.warnings))
            load_btn   = choice.addButton("Charger quand même", QMessageBox.ButtonRole.AcceptRole)
            reject_btn = choice.addButton("Rejeter ce job…",    QMessageBox.ButtonRole.RejectRole)
            load_btn.setStyleSheet(
                "background:#40a02b; color:white; font-weight:bold; padding:6px 14px; border-radius:6px;"
            )
            reject_btn.setStyleSheet(
                "background:#e64553; color:white; font-weight:bold; padding:6px 14px; border-radius:6px;"
            )
            choice.exec()

            if choice.clickedButton() is reject_btn:
                reason = self._ask_reject_reason(extra_context=warn_text)
                if reason is None:
                    self._show_waiting_screen()
                    return
                logger.info("Job rejeté (données CSV) — raison : %s", reason)
                self.waiting_widget.set_status(f"Job rejeté : {reason[:60]}")
                self.statusbar.showMessage(f"Job rejeté ({reason[:60]}) — passage au suivant…")
                self._go_to_next_job()
                return

        self.statusbar.showMessage("CSV validés — chargement de la session...")
        self.waiting_widget.hide_skip_button()

        session_dir = str(local_files.tracker.parent)
        self._load_session(session_dir)
        self._show_workspace()

    def _on_skip_job(self) -> None:
        """L'utilisateur veut passer au job suivant depuis l'écran d'attente."""
        # Annuler le téléchargement en cours s'il y en a un
        if self._download_thread is not None and self._download_thread.isRunning():
            self._download_thread.terminate()
            self._download_thread.wait(2000)
            self._download_thread = None

        # Arrêter le poller si actif
        if self._poller_thread is not None and self._poller_thread.isRunning():
            self._poller_thread.stop()
            self._poller_thread.wait(2000)
            self._poller_thread = None

        self.waiting_widget.hide_skip_button()
        self.waiting_widget.reset_file_progress()
        self.waiting_widget.set_status("Job ignoré — passage au suivant...")
        self.statusbar.showMessage("Job ignoré — reprise du polling...")
        logger.info("Job skipped by user from waiting screen (job_id=%s)", self._current_job)
        self._current_job = None

        self._go_to_next_job()

    def _on_load_from_nas(self) -> None:
        """L'utilisateur veut charger directement une session depuis S3 ou HDD."""
        dlg = SessionBrowserDialog(s3_config=self.config.s3, hdd_config=self.config.hdd, parent=self)
        dlg.session_selected.connect(self._on_nas_session_selected)
        dlg.exec()

    def _on_nas_session_selected(self, session_dir: str) -> None:
        """Une session a été choisie manuellement depuis le NAS — stopper le polling et charger."""
        # Stop any ongoing download
        if self._download_thread is not None and self._download_thread.isRunning():
            self._download_thread.terminate()
            self._download_thread.wait(2000)
            self._download_thread = None

        # Stop poller
        if self._poller_thread is not None and self._poller_thread.isRunning():
            self._poller_thread.stop()
            self._poller_thread.wait(2000)
            self._poller_thread = None

        self._current_job = None
        self.waiting_widget.hide_skip_button()
        self.waiting_widget.reset_file_progress()
        self.waiting_widget.stop_animation()

        logger.info("Loading NAS session directly: %s", session_dir)
        self._load_session(session_dir)
        self._show_workspace()

    # ------------------------------------------------------------------
    # SPOOL — récupération de scénarios depuis le serveur SPOOL (SFTP)
    # ------------------------------------------------------------------

    def _on_load_from_spool(self) -> None:
        """Récupère automatiquement la prochaine session depuis le HDD inbox (même logique que la vérification)."""
        hdd = self.config.hdd

        # Stopper le poller RabbitMQ si actif
        if self._poller_thread is not None and self._poller_thread.isRunning():
            self._poller_thread.stop()
            self._poller_thread.wait(2000)
            self._poller_thread = None

        self.waiting_widget.set_status("Connexion au serveur HDD…")
        self.waiting_widget.reset_file_progress()
        self.statusbar.showMessage("HDD : recherche de la prochaine session…")

        self._annotation_hdd_worker = HddVerificationWorker(
            host=hdd.host, port=hdd.port,
            username=hdd.username, password=hdd.password,
            inbox_base=hdd.silver_base,
            parent=self,
        )
        self._annotation_hdd_worker.file_progress.connect(self.waiting_widget.update_file_progress)
        self._annotation_hdd_worker.download_finished.connect(self._on_annotation_hdd_download_finished)
        self._annotation_hdd_worker.no_session_available.connect(self._on_annotation_hdd_no_session)
        self._annotation_hdd_worker.error_occurred.connect(self._on_annotation_hdd_error)
        self._annotation_hdd_worker.start()

    def _on_annotation_hdd_download_finished(self, session_id: str, local_files: object) -> None:
        """Téléchargement HDD (annotation) terminé — charger la session."""
        logger.info("Annotation HDD download complete: %s", session_id)
        self.statusbar.showMessage(f"Session '{session_id}' prête.")
        self._load_session(str(local_files.tracker.parent))
        self._show_workspace()

    def _on_annotation_hdd_no_session(self) -> None:
        self.waiting_widget.set_status("Aucune session disponible dans le HDD inbox.")
        self.statusbar.showMessage("HDD inbox vide — aucune session à annoter.")

    def _on_annotation_hdd_error(self, error: str) -> None:
        self.waiting_widget.set_status(f"Erreur HDD : {error[:80]}")
        self.statusbar.showMessage(f"Erreur HDD annotation : {error}")
        QMessageBox.critical(
            self, "Erreur HDD",
            f"Impossible de télécharger la session depuis le serveur HDD :\n{error}",
        )

    def _on_spool_list_error(self, error: str) -> None:
        self.waiting_widget.set_status("Erreur SPOOL")
        self.statusbar.showMessage(f"Erreur SPOOL : {error}")
        QMessageBox.critical(
            self, "Erreur SPOOL",
            f"Impossible de lister les scénarios depuis le SPOOL :\n{error}",
        )

    def _on_spool_scenarios_listed(self, scenarios: list) -> None:
        """Non utilisé — conservé pour compatibilité."""
        pass

    def _on_spool_download_error(self, error: str) -> None:
        self.waiting_widget.set_status(f"Erreur SPOOL : {error[:80]}")
        self.statusbar.showMessage(f"Erreur téléchargement SPOOL : {error}")
        QMessageBox.critical(
            self, "Erreur SPOOL",
            f"Impossible de télécharger le scénario depuis le SPOOL :\n{error}",
        )

    def _on_download_error(self, error: str) -> None:
        self.waiting_widget.set_status(f"Erreur de téléchargement : {error[:80]}")
        self.statusbar.showMessage(f"Erreur HDFS : {error}")

        msg = QMessageBox(self)
        msg.setWindowTitle("Erreur de téléchargement")
        msg.setText("<b>Impossible de télécharger les fichiers depuis HDFS.</b>")
        msg.setInformativeText(error)
        msg.setIcon(QMessageBox.Icon.Critical)
        next_btn = msg.addButton("Passer au job suivant", QMessageBox.ButtonRole.AcceptRole)
        retry_btn = msg.addButton("Réessayer", QMessageBox.ButtonRole.ResetRole)
        next_btn.setStyleSheet(
            "background:#e64553; color:white; font-weight:bold; padding:6px 14px; border-radius:6px;"
        )
        retry_btn.setStyleSheet(
            "background:#45475a; color:#cdd6f4; font-weight:bold; padding:6px 14px; border-radius:6px;"
        )
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is retry_btn:
            # Relancer le téléchargement du même job
            if self._current_job is not None:
                self.waiting_widget.set_status("Nouvelle tentative de téléchargement...")
                self.waiting_widget.reset_file_progress()
                self.waiting_widget.show_skip_button()
                self._download_thread = NASDownloadThread(
                    self._make_nas_client(), self._current_job, parent=self
                )
                self._download_thread.download_progress.connect(self._on_download_progress)
                self._download_thread.file_progress.connect(self.waiting_widget.update_file_progress)
                self._download_thread.download_finished.connect(self._on_download_finished)
                self._download_thread.download_error.connect(self._on_download_error)
                self._download_thread.start()
            else:
                self.waiting_widget.hide_skip_button()
                self._go_to_next_job()
        else:
            # Passer au job suivant
            self.waiting_widget.hide_skip_button()
            self.waiting_widget.reset_file_progress()
            logger.info("Download error — skipping to next job: %s", error)
            self._current_job = None
            self._go_to_next_job()

    # -- Prefetcher management --

    def _launch_prefetcher(self) -> None:
        """Start a background prefetch of the next scenario."""
        self._stop_prefetcher()  # safety: cancel any lingering prefetcher

        consumer = self._make_rabbitmq_consumer()
        nas_client = self._make_nas_client()

        self._prefetcher = ScenarioPrefetcher(
            consumer=consumer,
            hdfs_client=nas_client,  # NASClient (paramètre hérité)
            poll_interval_s=5.0,
            parent=self,
        )
        self._prefetcher.prefetch_status.connect(
            lambda msg: self.statusbar.showMessage(msg, 3000)
        )
        self._prefetcher.prefetch_error.connect(self._on_prefetch_background_error)
        self._prefetcher.start()
        logger.info("Prefetcher launched in background")

    def _stop_prefetcher(self) -> None:
        """Stop and clean up the prefetcher thread (does NOT discard files)."""
        if self._prefetcher is not None:
            if self._prefetcher.isRunning():
                self._prefetcher.stop()
                self._prefetcher.wait(3000)
            self._prefetcher = None

    def _on_prefetch_background_error(self, error: str) -> None:
        """Prefetch failed silently in background — log only, don't disrupt user."""
        logger.warning("Background prefetch error (non-fatal): %s", error)
        self.statusbar.showMessage(f"Prefetch échoué (sera réessayé) : {error[:80]}", 4000)

    # -- Fast-path: prefetcher signals received while on waiting screen --

    def _on_prefetch_status_waiting(self, msg: str) -> None:
        self.waiting_widget.set_status(msg)
        self.statusbar.showMessage(msg)

    def _on_prefetch_ready_from_waiting(self) -> None:
        """The prefetcher finished while we were already on the waiting screen."""
        scenario = self._prefetcher.consume() if self._prefetcher else None
        self._stop_prefetcher()
        if scenario is None:
            return
        self.waiting_widget.stop_animation()
        self._activate_prefetched_scenario(scenario)

    def _activate_prefetched_scenario(self, scenario: PrefetchedScenario) -> None:
        """Load a prefetched scenario directly into the workspace."""
        logger.info("Activating prefetched scenario: %s", scenario.session_dir)
        self._current_job = scenario.job
        self._load_session(str(scenario.session_dir))
        self._show_workspace()

    # ------------------------------------------------------------------
    # Session Loading
    # ------------------------------------------------------------------

    def _open_session_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Open Session Directory",
            str(Path.home() / "Desktop"),
        )
        if path:
            self._load_session(path)
            self._show_workspace()

    def _open_session_browser(self) -> None:
        """Open the session browser dialog (local filesystem, S3 or HDD)."""
        dialog = SessionBrowserDialog(s3_config=self.config.s3, hdd_config=self.config.hdd, parent=self)
        dialog.session_selected.connect(self._on_session_browser_selected)
        dialog.exec()

    def _on_session_browser_selected(self, local_path: str) -> None:
        self._load_session(local_path)
        self._show_workspace()

    def _load_session(self, session_dir: str) -> None:
        self.statusbar.showMessage(f"Chargement de la session : {session_dir}")
        QApplication.processEvents()

        new_session = None
        try:
            new_session = SessionDataLoader(session_dir)

            # Setup multi-video with per-camera tracker views
            # Order: left, head, right (left on the left, head in center, right on the right)
            camera_keys = list(new_session.cameras.keys())
            # Reorder to: left, head, right
            desired_order = ["left", "head", "right"]
            camera_positions = [cam for cam in desired_order if cam in camera_keys]
            # Add any remaining cameras not in the desired order
            camera_positions.extend([cam for cam in camera_keys if cam not in desired_order])

            tracker_names = new_session.metadata.tracker_names

            # Build tracker_map: assign a tracker to each camera by matching name.
            # If tracker names match camera positions (head/left/right), use direct mapping.
            # Otherwise fall back to assigning trackers by index.
            tracker_map: dict = {}
            for i, cam_pos in enumerate(camera_positions):
                if cam_pos in tracker_names:
                    tracker_map[cam_pos] = cam_pos
                elif i < len(tracker_names):
                    tracker_map[cam_pos] = tracker_names[i]

            self.multi_video.set_cameras(camera_positions, tracker_map=tracker_map)

            # Setup timeline
            self.timeline.set_frame_count(new_session.frame_count)
            self.timeline.set_fps(new_session.fps)

            # Setup gripper graph
            gripper_data = {}
            for side in new_session.metadata.gripper_sides:
                try:
                    ts, openings = new_session.get_gripper_timeseries(side)
                    if ts is not None:
                        gripper_data[side] = (ts, openings)
                except Exception as exc:
                    logger.warning("Gripper '%s' timeseries unavailable: %s", side, exc)
            self.timeline.set_gripper_data(gripper_data)
            # Adjust overall timeline height: base + controls row + gripper tracks
            n_grippers = len(gripper_data)
            self.timeline.setFixedHeight(110 + n_grippers * 24)

            # Load 3D tracker trajectories into the viewers
            try:
                all_trajectories = new_session.get_all_tracker_positions()
                # Map trajectories to camera columns via tracker_map
                traj_by_cam = {}
                for cam_pos, tracker_name in tracker_map.items():
                    traj = all_trajectories.get(tracker_name)
                    if traj is not None and len(traj) >= 2:
                        traj_by_cam[cam_pos] = traj
                if traj_by_cam:
                    self.multi_video.set_tracker_trajectories(traj_by_cam)
            except Exception as exc:
                logger.warning("Could not load tracker trajectories: %s", exc)

            # Keep tracker_map for sensor updates
            self._tracker_map = tracker_map

            # All setup succeeded — commit the new session
            if self.session:
                try:
                    self.session.release()
                except Exception:
                    pass
            self.session = new_session
            new_session = None  # ownership transferred

            # Give the frame loader the new session
            self._frame_loader.set_session(self.session)

            # Populate scenario input from session metadata (or clear if absent)
            scenario_from_meta = self.session.metadata.scenario_name
            self.scenario_input.setText(scenario_from_meta)

            # Charger les labels depuis la BDD (scénario), sinon fallback sur les defaults
            self.label_manager.labels.clear()
            self.label_manager.annotations.clear()
            self.label_manager._annotations_by_frame.clear()
            self.label_manager._annotations_by_label.clear()
            loaded_from_db = self._load_labels_from_scenario(scenario_from_meta)
            if not loaded_from_db:
                self._initialize_default_labels()

            # Crée les segments initiaux "non labellisés" pour chaque main
            self.label_manager.initialize_full_segment(self.session.frame_count, hand="left")
            self.label_manager.initialize_full_segment(self.session.frame_count, hand="right")
            # Réinitialise la sélection de segment
            self._selected_segment = None
            self.timeline.timeline_bar.set_selected_segment(None)
            self.timeline.refresh_annotations()

            # Load first frame
            self._goto_frame(0)

            # Feed verification widget with new session data (trajectories)
            self.verification_widget.load_session(self.session)
            self.verification_widget.set_buttons_enabled(True)
            self.verification_widget.set_info(
                f"Session : {self.session.metadata.session_id}  |  "
                f"{self.session.frame_count} frames @ {self.session.fps:.0f} FPS"
            )

            # Détection et affichage des NaN par zone
            nan_report = self.session.get_nan_report()
            if nan_report:
                zones = ", ".join(nan_report.keys())
                logger.warning("NaN détectés dans la session : %s", zones)

            self.statusbar.showMessage(
                f"Session chargée : {self.session.metadata.session_id} — "
                f"{self.session.frame_count} frames, {len(camera_positions)} caméras"
                + (f" — ⚠ NaN détectés ({len(nan_report)} zone(s))" if nan_report else "")
            )

            # # Lance la segmentation automatique en arrière-plan dès que la session est prête.
            # self._launch_seqensor(session_dir)

        except Exception as e:
            logger.exception("Failed to load session from %s", session_dir)
            # Release the partially-initialised session so we don't leak video captures
            if new_session is not None:
                try:
                    new_session.release()
                except Exception:
                    pass
            QMessageBox.critical(
                self, "Erreur de chargement",
                f"Impossible de charger la session :\n{e}",
            )
            self.statusbar.showMessage("Échec du chargement de session")

    # ------------------------------------------------------------------
    # Camera swap permanent (disque)
    # ------------------------------------------------------------------

    def _on_camera_swap_requested(self, pos_a: str, pos_b: str) -> None:
        """Échange sur disque les fichiers de deux caméras, puis recharge la session."""
        if self.session is None:
            return

        reply = QMessageBox.question(
            self,
            "Échanger les caméras",
            f"Échanger définitivement les données de « {pos_a} » et « {pos_b} » sur le disque ?\n\n"
            "Les fichiers vidéo, timestamps et colonnes tracker seront renommés.\n"
            "Cette opération est permanente.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            # Revert the visual swap in the verification widget
            if self._mode_btn.isChecked():
                self.verification_widget.load_session(self.session)
                frames = self.session.get_all_frames(self.current_frame_index)
                self.verification_widget.set_frames(frames)
                self.verification_widget.set_current_frame(self.current_frame_index, self.session)
            return

        session_dir = str(self.session.session_dir)
        self.statusbar.showMessage(f"Échange des caméras {pos_a} ↔ {pos_b} en cours…")
        QApplication.processEvents()

        try:
            # Arrêter les threads de décodage d'abord (handles Windows)
            self.verification_widget.release_decoders()
            # Libérer les captures vidéo de la session
            self.session.release()
            self.session = None

            swap_cameras_on_disk(session_dir, pos_a, pos_b)

        except Exception as e:
            logger.exception("Swap caméras échoué")
            QMessageBox.critical(self, "Erreur", f"Échec du swap caméras :\n{e}")
            self.statusbar.showMessage("Swap échoué")
            # Recharger quand même pour remettre la session dans un état propre
            self._load_session(session_dir)
            return

        # Recharger la session avec les fichiers renommés
        self._load_session(session_dir)
        self.statusbar.showMessage(f"Caméras {pos_a} ↔ {pos_b} échangées — session rechargée")

    # ------------------------------------------------------------------
    # Tracker swap (disque — CSV seulement)
    # ------------------------------------------------------------------

    def _on_tracker_swap_requested(self, tracker_a: str, tracker_b: str = "") -> None:
        """Échange définitif de deux trackers dans le CSV.

        Appelé depuis :
        - verification_widget (deux noms fournis directement)
        - multi_video annotation (un seul nom → ouvre un QInputDialog)
        """
        if self.session is None:
            return

        if not tracker_b:
            # Mode annotation : choisir la cible via dialog
            tracker_names = list(self.session.metadata.tracker_names)
            others = [t for t in tracker_names if t != tracker_a]
            if not others:
                QMessageBox.information(self, "Swap tracker", "Aucun autre tracker disponible.")
                return
            from PyQt6.QtWidgets import QInputDialog
            tracker_b, ok = QInputDialog.getItem(
                self,
                "Échanger le tracker",
                f"Échanger « {tracker_a} » avec :",
                others, 0, False,
            )
            if not ok:
                return

        reply = QMessageBox.question(
            self,
            "Échanger les trackers",
            f"Échanger définitivement les colonnes de « {tracker_a} » et « {tracker_b} » dans le CSV ?\n\n"
            "Les colonnes x, y, z, qw, qx, qy, qz des deux trackers seront interverties.\n"
            "Cette opération est permanente.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        session_dir = str(self.session.session_dir)
        self.statusbar.showMessage(f"Échange trackers {tracker_a} ↔ {tracker_b} en cours…")
        QApplication.processEvents()

        try:
            # Le CSV ne touche pas aux vidéos, donc on n'arrête PAS les décodeurs —
            # les threads vidéo continuent de tourner pendant le swap du CSV.
            # On libère seulement la session Python (lectures pandas en mémoire).
            old_session = self.session
            self.session = None
            old_session.release()

            swap_trackers_on_disk(session_dir, tracker_a, tracker_b)

            # Recharger uniquement les données tracker (CSV) dans une nouvelle session,
            # sans relancer les décodeurs vidéo ni réinitialiser les colonnes.
            new_session = SessionDataLoader(session_dir)
            self.session = new_session
            self.verification_widget.reload_trackers_only(new_session)

        except Exception as e:
            logger.exception("Swap trackers échoué")
            QMessageBox.critical(self, "Erreur", f"Échec du swap trackers :\n{e}")
            self.statusbar.showMessage("Swap trackers échoué")
            self._load_session(session_dir)
            return

        self.statusbar.showMessage(f"Trackers {tracker_a} ↔ {tracker_b} échangés")

    # ------------------------------------------------------------------
    # Camera rotation (disque)
    # ------------------------------------------------------------------

    def _on_camera_rotate_requested(self, camera_id: str) -> None:
        """Applique une rotation 180° permanente sur la vidéo de la caméra."""
        if self.session is None:
            return

        reply = QMessageBox.question(
            self,
            "Pivoter la vidéo",
            f"Pivoter la vidéo de « {camera_id} » de 180° ?\n\n"
            "Cette opération est permanente et ré-encodera la vidéo avec ffmpeg.\n"
            "Cela peut prendre quelques secondes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        session_dir = self.session.session_dir
        video_path = session_dir / "videos" / f"{camera_id}.mp4"

        if not video_path.exists():
            # Essayer .avi
            video_path = session_dir / "videos" / f"{camera_id}.avi"
            if not video_path.exists():
                QMessageBox.warning(self, "Erreur", f"Vidéo {camera_id} introuvable.")
                return

        self.statusbar.showMessage(f"Rotation de la vidéo {camera_id} en cours…")
        QApplication.processEvents()

        try:
            # Libérer les captures vidéo
            self.session.release()
            self.session = None

            # Rotation avec ffmpeg
            self._rotate_video_180(video_path)

        except Exception as e:
            logger.exception("Rotation vidéo échouée")
            QMessageBox.critical(self, "Erreur", f"Échec de la rotation :\n{e}")
            self.statusbar.showMessage("Rotation échouée")
            # Recharger quand même
            self._load_session(str(session_dir))
            return

        # Recharger la session
        self._load_session(str(session_dir))
        self.statusbar.showMessage(f"Vidéo {camera_id} pivotée — session rechargée")

    def _rotate_video_180(self, video_path: Path) -> None:
        """Rotate video 180° using ffmpeg (hflip,vflip or transpose=2,transpose=2)."""
        import subprocess
        import tempfile
        import shutil

        # Create temporary output file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir=video_path.parent)
        import os
        os.close(temp_fd)
        temp_path = Path(temp_path)

        try:
            # 180° rotation: flip horizontally + vertically (more efficient than double transpose)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", "hflip,vflip",  # 180° rotation
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "copy",  # Copy audio if any
                str(temp_path),
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=300,  # 5 minutes timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg exited with code {result.returncode}:\n"
                    + result.stderr.decode(errors="replace")[-2000:]
                )

            # Replace original with rotated version
            shutil.move(str(temp_path), str(video_path))
            logger.info("Video rotated 180° successfully: %s", video_path)

        finally:
            # Cleanup temp file if it still exists
            if temp_path.exists():
                temp_path.unlink()

    # # ------------------------------------------------------------------
    # # Seqensor — segmentation automatique de la session
    # # ------------------------------------------------------------------

    # def _launch_seqensor(self, session_dir: str) -> None:
    #     if self._seqensor_worker is not None and self._seqensor_worker.isRunning():
    #         self._seqensor_worker.quit()
    #         self._seqensor_worker.wait(2000)
    #     self._seqensor_segments = []
    #     self.seqensor_widget.clear()
    #     self.seqensor_widget.set_running(True)
    #     ref_time = None
    #     if self.session is not None:
    #         ref_time = self.session._ref_time
    #     self._seqensor_worker = SeqensorWorker(
    #         session_dir=session_dir,
    #         ref_time=ref_time,
    #         fps=self.session.fps if self.session else 30.0,
    #         parent=self,
    #     )
    #     self._seqensor_worker.progress.connect(self._on_seqensor_progress)
    #     self._seqensor_worker.segments_ready.connect(self._on_seqensor_ready)
    #     self._seqensor_worker.error_occurred.connect(self._on_seqensor_error)
    #     self._seqensor_worker.start()
    #     logger.info("SeqensorWorker started for %s", session_dir)

    # def _on_seqensor_progress(self, text: str) -> None:
    #     self.seqensor_widget.set_status(text)
    #     self.statusbar.showMessage(text, 3000)

    # def _on_seqensor_ready(self, segments: list) -> None:
    #     self._seqensor_segments = segments
    #     self.seqensor_widget.set_running(False)
    #     self.seqensor_widget.set_segments(segments)
    #     self.timeline.set_seqensor_segments(segments)
    #     logger.info("SeqensorWorker finished: %d segments", len(segments))

    # def _on_seqensor_error(self, error: str) -> None:
    #     self.seqensor_widget.set_running(False)
    #     self.seqensor_widget.set_status(f"Erreur : {error[:80]}")
    #     logger.warning("SeqensorWorker error: %s", error)

    # def _on_seqensor_apply_segment(self, seg: dict) -> None:
    #     if self.session is None:
    #         return
    #     label_name = f"Cluster {seg.get('label', 0)}"
    #     colour     = seg.get("colour", "#89b4fa")
    #     label = self.label_manager.get_label_by_name(label_name)
    #     if label is None:
    #         label = self.label_manager.add_label(name=label_name, color=colour)
    #         self.label_panel.refresh_labels()
    #     start = seg.get("start_idx", 0)
    #     end   = max(seg.get("end_idx", start), start)
    #     try:
    #         self.label_manager.add_interval_annotation(
    #             start_frame=start,
    #             end_frame=end,
    #             label_id=label.id,
    #             metadata={
    #                 "source": "seqensor",
    #                 "start_t": seg.get("start_t"),
    #                 "end_t":   seg.get("end_t"),
    #                 "duration_s": seg.get("duration_s"),
    #             },
    #         )
    #         self.timeline.refresh_annotations()
    #         self.label_panel._update_statistics()
    #         self.statusbar.showMessage(
    #             f"Segment Seqensor appliqué : '{label_name}' frames {start}–{end}"
    #         )
    #     except Exception as e:
    #         logger.error("Failed to apply Seqensor segment: %s", e)

    # ------------------------------------------------------------------
    # Frame Navigation
    # ------------------------------------------------------------------

    def _request_frame(self, frame_index: int) -> None:
        """Schedule a frame decode via debounce — collapses rapid requests."""
        self._pending_frame_request = frame_index
        if not self._frame_debounce.isActive():
            self._frame_debounce.start()

    def _flush_frame_request(self) -> None:
        """Called by debounce timer — sends the most recent frame to the loader."""
        if self._pending_frame_request is not None and self.session is not None:
            self._frame_loader.request_frame(self._pending_frame_request)
            # Update sensor data immediately (cheap — no video decode)
            self._update_sensors(self._pending_frame_request)
            self._pending_frame_request = None

    def _update_sensors(self, frame_index: int) -> None:
        """Update 3D tracker viewers for the given frame index."""
        if self.session is None:
            return
        try:
            t = self.session.frame_timestamp(frame_index)
            tracker_states = self.session.get_tracker_state(t)
            tracker_map = getattr(self, "_tracker_map", {})
            for cam_pos, tracker_name in tracker_map.items():
                state = tracker_states.get(tracker_name)
                if state is None:
                    continue
                transform = Transform3D(
                    position=state["position"],
                    rotation=state["quaternion"],
                    rotation_format="quaternion",
                )
                self.multi_video.update_tracker(cam_pos, transform)
        except Exception as exc:
            logger.debug("_update_sensors error: %s", exc)

    def _on_frame_ready(self, frame_index: int, frames: dict) -> None:
        """Called on main thread when background decode finishes."""
        # Discard stale results if user has already moved further
        if frame_index != self.current_frame_index:
            return
        try:
            self.multi_video.set_frames(frames)
        except Exception as e:
            logger.error("Error displaying frame %d: %s", frame_index, e)
        # Also update verification widget if active
        if self._mode_btn.isChecked():
            try:
                self.verification_widget.set_frames(frames)
                self.verification_widget.set_current_frame(frame_index, self.session)
            except Exception as e:
                logger.error("Error updating verification frame %d: %s", frame_index, e)

    def _load_frame(self, frame_index: int) -> None:
        """Request async frame load (used during playback where no debounce needed)."""
        if self.session is None:
            return
        self._frame_loader.request_frame(frame_index)
        self._update_sensors(frame_index)

    def _on_frame_changed(self, frame_index: int) -> None:
        # Update index immediately — debounced decode follows
        self.current_frame_index = frame_index
        self._request_frame(frame_index)

    def _toggle_playback(self) -> None:
        self.timeline.toggle_playback()

    def _on_play_toggled(self, is_playing: bool) -> None:
        self.is_playing = is_playing
        if is_playing:
            if self.session is None:
                return
            fps = self.session.fps if self.session.fps > 0 else 30.0
            # Hand off to the thread's sequential playback loop (no seek per frame)
            self._frame_loader.start_playback(self.current_frame_index, fps=fps)
        else:
            # Stop the thread's playback loop
            self._frame_loader.stop_playback()

    def _on_timeline_annotation_clicked(self, annotation: Annotation) -> None:
        """Open edit dialog when user right-clicks an annotation on the timeline."""
        if annotation is None:
            return

        dlg = AnnotationDetailDialog(annotation, self.label_manager, self)
        # Track if the annotation was modified or deleted
        was_modified = [False]

        def on_saved():
            was_modified[0] = True

        def on_deleted():
            was_modified[0] = True

        dlg.saved.connect(on_saved)
        dlg.deleted.connect(on_deleted)
        dlg.exec()

        # Refresh UI AFTER dialog is closed
        if was_modified[0]:
            self._on_annotation_modified()

    def _on_annotation_modified(self) -> None:
        """Refresh UI after annotation is modified or deleted from timeline."""
        logger.info("Annotation modified - refreshing UI")
        try:
            # Force timeline to rebuild its cache and redraw
            self.timeline.refresh_annotations()
            # Refresh the annotation list panel
            self.annotation_list_panel.refresh()
            # Update label panel statistics
            self.label_panel._update_statistics()
            # Force immediate repaint
            self.timeline.repaint()
            # Restore focus to label list to ensure clicks work
            self.label_panel.label_list.setFocus()
            logger.info("UI refresh completed")
        except Exception as e:
            logger.error(f"Error refreshing UI after annotation modification: {e}", exc_info=True)

    def _on_playback_frame(self, frame_index: int, frames: dict) -> None:
        """Called from the main thread when the playback loop emits a new frame."""
        if not self.is_playing:
            return
        if self.session is not None and frame_index >= self.session.frame_count:
            # End of video — stop playback
            self.timeline.toggle_playback()
            return
        self.current_frame_index = frame_index
        self.timeline.set_current_frame_silent(frame_index)
        try:
            self.multi_video.set_frames(frames)
        except Exception as exc:
            logger.error("Error displaying playback frame %d: %s", frame_index, exc)
        self._update_sensors(frame_index)
        # Also update verification widget if active
        if self._mode_btn.isChecked():
            try:
                self.verification_widget.set_frames(frames)
                self.verification_widget.set_current_frame(frame_index, self.session)
            except Exception as exc:
                logger.error("Error updating verification playback frame %d: %s", frame_index, exc)

    def _advance_frame(self) -> None:
        """Legacy slot kept for QTimer compatibility — unused in playback mode."""
        pass

    def _step(self, delta: int) -> None:
        if self.session is None:
            return
        new_frame = max(0, min(self.current_frame_index + delta, self.session.frame_count - 1))
        self.timeline.set_current_frame(new_frame)

    def _goto_frame(self, frame: int) -> None:
        self.timeline.set_current_frame(frame)

    def _goto_last_frame(self) -> None:
        if self.session:
            self.timeline.set_current_frame(self.session.frame_count - 1)

    # ------------------------------------------------------------------
    # Temporal flow (step size)
    # ------------------------------------------------------------------

    def _current_step(self) -> int:
        return self._step_levels[self._step_level_index]

    def _increase_step(self) -> None:
        if self._step_level_index < len(self._step_levels) - 1:
            self._step_level_index += 1
        self._refresh_step_label()

    def _decrease_step(self) -> None:
        if self._step_level_index > 0:
            self._step_level_index -= 1
        self._refresh_step_label()

    def _refresh_step_label(self) -> None:
        """Update the permanent step-size label in the status bar."""
        try:
            step = self._current_step()
            fps = self.session.fps if self.session else 30.0
            fps = fps if fps > 0 else 30.0
            seconds = step / fps
            if step == 1:
                text = "Pas : 1 frame"
            elif seconds < 1:
                text = f"Pas : {step} fr ({seconds:.2f}s)"
            else:
                text = f"Pas : {step} fr ({seconds:.1f}s)"
            self._step_label.setText(text)
        except Exception as exc:
            logger.error("_refresh_step_label error: %s", exc)

    # Keep backward-compat alias (used nowhere but harmless)
    def _show_step_info(self) -> None:
        self._refresh_step_label()

    # ------------------------------------------------------------------
    # In/Out shortcuts
    # ------------------------------------------------------------------

    def _shortcut_set_in(self) -> None:
        self.timeline.timeline_bar.set_in_point(self.current_frame_index)

    def _shortcut_set_out(self) -> None:
        self.timeline.timeline_bar.set_out_point(self.current_frame_index)

    def _shortcut_split_hand(self, hand: str) -> None:
        """Coupe le segment de la main *hand* sous le playhead (A=gauche, Z=droite)."""
        if self.session is None:
            return
        frame = self.current_frame_index
        seg = self.label_manager.get_segment_at_frame(frame, hand=hand)
        if seg is None:
            hand_label = "gauche" if hand == "left" else "droite"
            self.statusbar.showMessage(f"Aucun segment [{hand_label}] à couper ici.", 2000)
            return
        if frame <= seg.start_frame or frame > seg.end_frame:
            self.statusbar.showMessage("Le playhead est au bord du segment, pas de coupe.", 2000)
            return
        right_part = self.label_manager.split_annotation_at_frame(seg.id, frame)
        if right_part is None:
            self.statusbar.showMessage("Impossible de couper le segment ici.", 2000)
            return

        # Pause la vidéo si elle est en lecture
        if self.is_playing:
            self.timeline.toggle_playback()

        # Sélectionne automatiquement la partie droite (frame … end) pour que
        # l'annotation commence exactement à la frame du curseur
        self._selected_segment = right_part
        self.timeline.timeline_bar.set_selected_segment(right_part)

        self.timeline.refresh_annotations()
        self.annotation_list_panel.refresh()
        self.label_panel._update_statistics()
        hand_label = "G" if hand == "left" else "D"
        self.statusbar.showMessage(
            f"[{hand_label}] Segment coupé à {frame} — {seg.start_frame}…{frame-1} | {frame}…{right_part.end_frame}"
            " — assignez un label avec 1/2/3…"
        )

    def _on_segment_selected(self, annotation: "Annotation") -> None:
        """Called when the user clicks a segment on the timeline."""
        self._selected_segment = annotation
        self.timeline.timeline_bar.set_selected_segment(annotation)
        from ..labeling.label_manager import UNLABELED_LABEL_ID
        if annotation.label_id == UNLABELED_LABEL_ID:
            self.statusbar.showMessage(
                f"Segment sélectionné : frames {annotation.start_frame}–{annotation.end_frame} "
                "— cliquez sur un label pour l'assigner"
            )
        else:
            self.statusbar.showMessage(
                f"Segment sélectionné : « {annotation.label_name} » "
                f"frames {annotation.start_frame}–{annotation.end_frame}"
            )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def _ask_hand(self) -> "Optional[tuple[str, bool]]":
        """Show a compact popup asking which hand and whether the action failed.

        Returns ``(hand, is_fail)`` where *hand* is ``'right'`` or ``'left'``,
        and *is_fail* is ``True`` when the [fail] toggle is active.
        Returns ``None`` if the user cancels (closes without clicking).
        """
        logger.debug("_ask_hand() called")
        from PyQt6.QtWidgets import QCheckBox as _QCheckBox
        from PyQt6.QtGui import QCursor

        dlg = QDialog(self)
        dlg.setWindowTitle("Main")
        dlg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        dlg.setModal(True)
        dlg.setFixedSize(260, 90)
        dlg.setStyleSheet(
            "QDialog { background: #1e1e2e; border: 1px solid #45475a; border-radius: 6px; }"
            "QPushButton { font-size: 13px; font-weight: bold; border-radius: 4px; padding: 6px 18px; }"
            "QCheckBox { color: #f38ba8; font-size: 12px; font-weight: bold; }"
            "QCheckBox::indicator { width: 15px; height: 15px; border: 2px solid #f38ba8;"
            "  border-radius: 3px; background: #1e1e2e; }"
            "QCheckBox::indicator:checked { background: #f38ba8; }"
        )
        root = QVBoxLayout(dlg)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # Row 1 : hand buttons
        hand_row = QHBoxLayout()
        hand_row.setSpacing(10)

        result = [None]   # will hold (hand, is_fail)

        btn_left = QPushButton("Gauche")
        btn_left.setStyleSheet("QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; }"
                               "QPushButton:hover { background: #45475a; }")
        btn_right = QPushButton("Droite")
        btn_right.setStyleSheet("QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; }"
                                "QPushButton:hover { background: #45475a; }")

        fail_cb = _QCheckBox("[fail]")

        def _pick(hand):
            result[0] = (hand, fail_cb.isChecked())
            logger.debug(f"Hand selected: {hand}, fail: {fail_cb.isChecked()}")
            dlg.accept()

        btn_left.clicked.connect(lambda: _pick("left"))
        btn_right.clicked.connect(lambda: _pick("right"))

        hand_row.addWidget(btn_left)
        hand_row.addWidget(btn_right)
        root.addLayout(hand_row)

        # Row 2 : fail checkbox
        fail_row = QHBoxLayout()
        fail_row.addStretch()
        fail_row.addWidget(fail_cb)
        fail_row.addStretch()
        root.addLayout(fail_row)

        pos = QCursor.pos()
        dlg.move(pos.x() - 130, pos.y() - 90)

        exec_result = dlg.exec()
        logger.debug(f"_ask_hand() dialog exec result: {exec_result}, result: {result[0]}")
        return result[0]

    def _on_frame_annotation(self, label_id: str) -> None:
        """Add frame annotation at current frame."""
        if self.session is None:
            return
        logger.debug(f"_on_frame_annotation called for label {label_id}")
        pick = self._ask_hand()
        logger.debug(f"Pick returned: {pick}")
        if pick is None:
            logger.warning("Hand selection cancelled - annotation not created")
            return
        hand, is_fail = pick
        try:
            self.label_manager.add_frame_annotation(
                frame_index=self.current_frame_index,
                label_id=label_id,
                metadata={"hand": hand, "fail": is_fail},
            )
            self.timeline.refresh_annotations()
            self.annotation_list_panel.refresh()
            label = self.label_manager.labels[label_id]
            fail_str = " [fail]" if is_fail else ""
            self.statusbar.showMessage(
                f"Frame annotation: '{label.name}'{fail_str} at frame {self.current_frame_index} [{hand}]"
            )
            self.label_panel._update_statistics()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to add annotation:\n{e}")

    def _on_interval_annotation(self, label_id: str) -> None:
        """Assign a label to the currently selected segment (new workflow).

        Falls back to the legacy IN/OUT workflow if no segment is selected.
        """
        logger.info(f"_on_interval_annotation called with label_id: {label_id}")
        if self.session is None:
            logger.warning("Session is None, cannot create annotation")
            return

        from ..labeling.label_manager import UNLABELED_LABEL_ID

        # ── New workflow: assign label to the selected segment ──────────
        if self._selected_segment is not None:
            seg = self._selected_segment
            # Verify the segment still exists in the manager
            seg_exists = any(a.id == seg.id for a in self.label_manager.annotations)
            if seg_exists:
                # La main est déjà définie par la touche A/Z qui a créé le segment
                hand = seg.metadata.get("hand", "right")
                try:
                    self.label_manager.assign_label_to_annotation(
                        annotation_id=seg.id,
                        label_id=label_id,
                        metadata={"hand": hand, "fail": False},
                    )
                    self.label_manager.convert_unlabeled_to_idle()
                    # Clear selection
                    self._selected_segment = None
                    self.timeline.timeline_bar.set_selected_segment(None)
                    self.timeline.refresh_annotations()
                    self.annotation_list_panel.refresh()
                    label = self.label_manager.labels[label_id]
                    self.statusbar.showMessage(
                        f"Label assigné : '{label.name}' "
                        f"frames {seg.start_frame}–{seg.end_frame} [{hand}]"
                    )
                    self.label_panel._update_statistics()
                except Exception as e:
                    QMessageBox.warning(self, "Erreur", f"Impossible d'assigner le label :\n{e}")
                return
            else:
                # The segment was deleted (e.g. split), clear stale reference
                self._selected_segment = None
                self.timeline.timeline_bar.set_selected_segment(None)

        # ── Legacy fallback: use IN/OUT points ──────────────────────────
        in_pt, out_pt = self.timeline.get_in_out()
        logger.info(f"IN/OUT points: {in_pt}, {out_pt}")
        if in_pt is None or out_pt is None:
            QMessageBox.information(
                self, "Sélectionner un segment",
                "Cliquez sur un segment de la timeline pour le sélectionner,\n"
                "puis cliquez sur un label pour l'assigner.\n\n"
                "Ou utilisez les points IN [I] / OUT [O] comme avant."
            )
            return

        pick = self._ask_hand()
        if pick is None:
            return
        hand, is_fail = pick

        start = min(in_pt, out_pt)
        end = max(in_pt, out_pt)

        try:
            self.label_manager.add_interval_annotation(
                start_frame=start,
                end_frame=end,
                label_id=label_id,
                metadata={"hand": hand, "fail": is_fail},
            )
            self.timeline.refresh_annotations()
            self.annotation_list_panel.refresh()
            label = self.label_manager.labels[label_id]
            fail_str = " [fail]" if is_fail else ""
            self.statusbar.showMessage(
                f"Interval annotation: '{label.name}'{fail_str} frames {start}-{end} [{hand}]"
            )
            self.label_panel._update_statistics()

            # Clear in/out after annotation
            self.timeline._clear_in_out()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to add annotation:\n{e}")

    def _on_annotations_changed(self) -> None:
        """Synchronise timeline et statistiques après un CRUD depuis l'AnnotationListPanel."""
        self.timeline.refresh_annotations()
        self.label_panel._update_statistics()

    # ------------------------------------------------------------------
    # Annotator identity
    # ------------------------------------------------------------------

    def _on_annotator_changed(self, text: str) -> None:
        """Persist annotator name to config whenever the field changes."""
        self.config.annotator = text.strip()

    # ------------------------------------------------------------------
    # Save / Load / Export
    # ------------------------------------------------------------------

    def _save_annotations(self) -> None:
        if not self.label_manager.annotations:
            QMessageBox.information(self, "Info", "No annotations to save")
            return

        default_name = "annotations.json"
        if self.session:
            default_name = f"annotations_{self.session.metadata.session_id}.json"

        # ── Dialogue de choix de destination ────────────────────────
        dlg = _SaveDestinationDialog(
            nas_host=self.config.nas.host or "",
            nas_port=self.config.nas.port,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        choice = dlg.choice()

        # ── Sauvegarde locale ────────────────────────────────────────
        if choice == _SaveDestinationDialog.LOCAL:
            path, _ = QFileDialog.getSaveFileName(
                self, "Enregistrer les annotations",
                str(Path.home() / default_name),
                "JSON Files (*.json)",
            )
            if not path:
                return
            try:
                self.label_manager.save_to_file(Path(path))
                self.statusbar.showMessage(f"Annotations sauvegardées : {path}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de sauvegarder :\n{e}")

        # ── Sauvegarde NAS ───────────────────────────────────────────
        elif choice == _SaveDestinationDialog.NAS:
            self._save_annotations_to_nas(default_name)

    def _save_annotations_to_nas(self, filename: str) -> None:
        """Sauvegarde le fichier d'annotations directement sur le NAS via SFTP."""
        import tempfile, paramiko

        cfg = self.config.nas

        # Choisir le chemin de destination sur le NAS
        # On propose le silver_root comme base par défaut
        default_remote = f"{cfg.silver_root.rstrip('/')}/{filename}" if cfg.silver_root else f"/{filename}"
        remote_path, ok = _NasPathDialog.get_path(
            parent=self,
            host=cfg.host,
            port=cfg.port,
            username=cfg.username,
            password=cfg.password or "",
            key_path=cfg.key_path,
            default_path=default_remote,
            title="Choisir l'emplacement sur le NAS",
            filename=filename,
        )
        if not ok or not remote_path:
            return

        # Export dans un fichier temporaire puis upload
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            self.label_manager.save_to_file(tmp_path)

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            kw = dict(hostname=cfg.host, port=cfg.port, username=cfg.username, timeout=15)
            if cfg.password:
                kw["password"] = cfg.password
            elif cfg.key_path:
                kw["key_filename"] = cfg.key_path
            ssh.connect(**kw)
            sftp = ssh.open_sftp()
            # Créer les dossiers parents si nécessaire
            remote_dir = str(Path(remote_path).parent)
            try:
                sftp.makedirs(remote_dir)
            except Exception:
                pass
            sftp.put(str(tmp_path), remote_path)
            sftp.close()
            ssh.close()
            tmp_path.unlink(missing_ok=True)

            self.statusbar.showMessage(f"Annotations sauvegardées sur le NAS : {remote_path}")
            QMessageBox.information(
                self, "Sauvegarde NAS",
                f"Annotations sauvegardées avec succès :\n{cfg.host}:{remote_path}",
            )
        except Exception as e:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            QMessageBox.critical(self, "Erreur NAS", f"Impossible de sauvegarder sur le NAS :\n{e}")

    def _load_annotations(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations", str(Path.home()), "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            self.label_manager.load_from_file(Path(path))
            self.label_panel.refresh_labels()
            self.timeline.refresh_annotations()
            self.annotation_list_panel.refresh()
            self.statusbar.showMessage(f"Loaded annotations from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")

    def _lerobot_export_dialog(self) -> tuple[str, bool]:
        """Show a dialog to choose output directory and vflip option.

        Returns:
            (path, vflip) — path is "" if the user cancelled.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Export LeRobot v3.0")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel("<b>Dossier de destination</b>"))

        dir_row = QHBoxLayout()
        dir_edit = QLineEdit(str(Path.home()))
        dir_edit.setReadOnly(True)
        dir_row.addWidget(dir_edit, stretch=1)

        browse_btn = QPushButton("Parcourir…")
        def _browse():
            p = QFileDialog.getExistingDirectory(dlg, "Choisir le dossier de destination", dir_edit.text())
            if p:
                dir_edit.setText(p)
        browse_btn.clicked.connect(_browse)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        layout.addWidget(QFrame())  # separator

        vflip_cb = QCheckBox("Retourner les vidéos verticalement (vflip)")
        vflip_cb.setToolTip(
            "Applique un flip haut/bas via ffmpeg sur chaque vidéo exportée.\n"
            "Requiert ffmpeg installé sur le système."
        )
        layout.addWidget(vflip_cb)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return "", False

        return dir_edit.text(), vflip_cb.isChecked()

    def _export_annotations(self, format_type: str) -> None:
        if not self.label_manager.annotations:
            QMessageBox.information(self, "Info", "No annotations to export")
            return

        # For LeRobot export, check scenario name
        scenario_name = ""
        if format_type == "lerobot":
            scenario_name = self.scenario_input.text().strip()
            if not scenario_name:
                QMessageBox.warning(
                    self,
                    "Nom du scénario requis",
                    "Veuillez saisir un nom de scénario avant d'exporter en format LeRobot."
                )
                return

        vflip = False
        if format_type == "lerobot":
            path, vflip = self._lerobot_export_dialog()
        else:
            extensions = {"json": "JSON Files (*.json)", "csv": "CSV Files (*.csv)"}
            suffixes = {"json": ".json", "csv": ".csv"}
            path, _ = QFileDialog.getSaveFileName(
                self, f"Export as {format_type.upper()}",
                str(Path.home() / f"annotations{suffixes[format_type]}"),
                extensions[format_type],
            )
        if not path:
            return

        try:
            dataset_info = {}
            if self.session:
                session_id = self.session.metadata.session_id
                # For LeRobot format, use SCENARIO_OriginalName pattern
                if format_type == "lerobot" and scenario_name:
                    export_session_id = f"{scenario_name}_{session_id}"
                else:
                    export_session_id = session_id

                dataset_info = {
                    "session_id": export_session_id,
                    "fps": self.session.fps,
                    "duration": self.session.duration,
                    "frame_count": self.session.frame_count,
                }

            exporter = AnnotationExporter(self.label_manager, dataset_info)
            if format_type == "json":
                exporter.export_to_json(Path(path))
            elif format_type == "csv":
                exporter.export_to_csv(Path(path), format_type="annotation_based")
            elif format_type == "lerobot":
                exporter.export_to_lerobot_format(Path(path), session=self.session, vflip=vflip)

            self.statusbar.showMessage(f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    # ------------------------------------------------------------------
    # Upload to HDFS (fire-and-forget en sous-process)
    # ------------------------------------------------------------------

    def _upload_to_hdfs(self) -> None:
        """Export les annotations en format LeRobot puis upload vers le NAS (silver)
        en arrière-plan dans un sous-process indépendant.

        L'utilisateur est immédiatement renvoyé vers l'enregistrement suivant
        sans attendre la fin du transfert réseau.
        """
        if not self.label_manager.annotations:
            QMessageBox.information(self, "Info", "Aucune annotation à uploader.")
            return
        if self.session is None:
            QMessageBox.warning(self, "Erreur", "Aucune session chargée.")
            return

        # Check annotator name
        annotator = self.config.annotator
        if not annotator:
            QMessageBox.warning(
                self,
                "Nom d'annotateur requis",
                "Veuillez saisir votre prénom dans le champ 'Annotateur' avant d'uploader."
            )
            return

        # Show validation dialog (rating + flags)
        validation_result = UploadValidationDialog.get_validation(self)
        if validation_result is None:
            # User cancelled
            return

        rating, flags = validation_result
        logger.info(f"Upload validation: rating={rating}, flags={flags}")

        session_id = self.session.metadata.session_id
        session_dir = self.session.session_dir

        # 1. Export des JSON directement dans le dossier de session
        try:
            dataset_info = {
                "session_id": session_id,
                "fps": self.session.fps,
                "duration": self.session.duration,
                "frame_count": self.session.frame_count,
            }
            exporter = AnnotationExporter(self.label_manager, dataset_info)
            exporter.export_to_chunk_format(
                output_dir=session_dir,
                episode_index=0,
                annotator=self.config.annotator,
                quality_rating=rating,
                quality_flags=flags,
            )
            logger.info(
                "JSON exportés dans le dossier de session : %s", session_dir
            )
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Export JSON échoué :\n{e}")
            return

        # 2. Upload du dossier de session vers HDD gold/{session_id}
        hdd_dest = f"{self.config.hdd.gold_base.rstrip('/')}/{session_id}"
        proc = None
        try:
            proc = upload_directory_sftp_background(
                local_dir=session_dir,
                nas_dest=hdd_dest,
                host=self.config.hdd.host,
                port=self.config.hdd.port,
                username=self.config.hdd.username,
                password=self.config.hdd.password or "",
                key_path=None,
                delete_after=False,        # on garde le dossier local
                delete_session_dir=None,
            )
            self._bg_upload_procs.append(proc)
            if not self._bg_upload_timer.isActive():
                self._bg_upload_timer.start()
            logger.info(
                "Upload HDD gold background démarré (PID %d) : %s -> sftp://%s%s",
                proc.pid, session_dir, self.config.hdd.host, hdd_dest,
            )
        except Exception as e:
            logger.error("Impossible de lancer le sous-process d'upload HDD : %s", e)
            fallback_dir = Path.cwd() / "chunk_exports" / session_id
            try:
                import shutil as _shutil
                _shutil.copytree(str(session_dir), str(fallback_dir), dirs_exist_ok=True)
                QMessageBox.warning(
                    self,
                    "Upload échoué — export local conservé",
                    f"Le transfert vers le HDD a échoué :\n{e}\n\n"
                    f"Les fichiers sont conservés localement dans :\n{fallback_dir}",
                )
            except Exception as copy_exc:
                logger.error("Fallback local échoué : %s", copy_exc)

        self.statusbar.showMessage(
            f"Upload HDD gold en cours (PID {proc.pid}) — passage au job suivant…"
            if proc else
            f"Upload HDD échoué — fichiers conservés dans {session_dir}"
        )

        # 3. Nettoyage côté application et passage immédiat au job suivant.
        #    On garde le session_dir sur disque — le subprocess d'upload
        #    le supprimera lui-même après transfert réussi.
        self._cleanup_session_state(keep_session_dir=True)
        self._go_to_next_job()

    def _poll_background_uploads(self) -> None:
        """Vérifie périodiquement l'état des sous-process d'upload en cours."""
        still_running = []
        for proc in self._bg_upload_procs:
            ret = proc.poll()
            if ret is None:
                still_running.append(proc)
            else:
                # Lire la sortie pour logger
                try:
                    output, _ = proc.communicate(timeout=1)
                    for line in (output or "").splitlines():
                        if line.strip():
                            logger.info("[bg-upload PID %d] %s", proc.pid, line)
                except Exception:
                    pass
                if ret == 0:
                    logger.info("Upload background terminé avec succès (PID %d)", proc.pid)
                else:
                    logger.warning("Upload background terminé avec erreur code %d (PID %d)", ret, proc.pid)

        self._bg_upload_procs = still_running
        if not still_running:
            self._bg_upload_timer.stop()

    def _check_nas_available(self, timeout: int = 5) -> bool:
        """Teste rapidement si le NAS est joignable via TCP sur le port SFTP.

        Returns:
            True si la connexion TCP réussit dans *timeout* secondes, False sinon.
        """
        import socket
        try:
            host = self.config.nas.host
            port = self.config.nas.port
            if not host:
                return False
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception as exc:
            logger.warning("NAS inaccessible (%s:%s) : %s", self.config.nas.host, self.config.nas.port, exc)
            return False

    def _cleanup_session_state(self, keep_session_dir: bool = False) -> None:
        """Libère la session et les annotations côté application.

        Args:
            keep_session_dir: Si True, le répertoire de cache de session n'est
                              pas supprimé (le sous-process d'upload s'en charge).
        """
        import shutil

        if self.session is not None:
            session_dir = self.session.session_dir
            self.session.release()
            self.session = None
            if not keep_session_dir and session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
                logger.info("Cache session local supprimé : %s", session_dir)

        self._current_job = None
        self.label_manager.clear_annotations()
        self.current_frame_index = 0

    # Raisons de rejet prédéfinies (label affiché, clé interne)
    _REJECT_REASONS = [
        ("Données manquantes (NaN / capteur hors ligne)",   "donnees_manquantes"),
        ("Faux geste / geste non représentatif",            "faux_geste"),
        ("Capteur cassé ou mal calibré",                    "capteur_casse"),
        ("Problème d'alignement temporel",                  "alignement"),
        ("Vidéo corrompue ou illisible",                    "video_corrompue"),
        ("Session trop courte / incomplète",                "session_courte"),
        ("Bruit excessif dans les données",                 "bruit_excessif"),
        ("Autre raison (préciser ci-dessous)",              "autre"),
    ]

    def _ask_reject_reason(self, extra_context: str = "") -> "Optional[str]":
        """Affiche un dialogue pour choisir la raison de rejet du job.

        Args:
            extra_context: Texte affiché en haut du dialogue (ex : liste de
                           problèmes CSV détectés automatiquement).

        Returns:
            La raison de rejet (str) si l'utilisateur confirme,
            None si l'utilisateur annule.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Raison du rejet")
        dlg.setMinimumWidth(500)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)

        # Contexte optionnel (ex : avertissements CSV)
        if extra_context:
            ctx_label = QLabel(extra_context)
            ctx_label.setWordWrap(True)
            ctx_label.setStyleSheet(
                "background:#181825; color:#fab387; padding:8px; "
                "font-family:monospace; font-size:11px; border-radius:4px;"
            )
            layout.addWidget(ctx_label)

            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet("color:#313244;")
            layout.addWidget(sep)

        # En-tête
        header = QLabel("<b>Sélectionnez la raison du rejet :</b>")
        header.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(header)

        # Liste des raisons
        reason_list = QListWidget()
        reason_list.setStyleSheet(
            "QListWidget { background:#181825; color:#cdd6f4; border:1px solid #313244; "
            "border-radius:6px; font-size:12px; }"
            "QListWidget::item { padding:6px 10px; }"
            "QListWidget::item:selected { background:#313244; color:#cdd6f4; }"
            "QListWidget::item:hover { background:#252535; }"
        )
        reason_list.setFixedHeight(220)
        for label, _ in self._REJECT_REASONS:
            item = QListWidgetItem(label)
            reason_list.addItem(item)
        reason_list.setCurrentRow(0)
        layout.addWidget(reason_list)

        # Champ libre (activé automatiquement quand "Autre" est sélectionné)
        other_input = QLineEdit()
        other_input.setPlaceholderText("Précisez la raison…")
        other_input.setStyleSheet(
            "background:#181825; color:#cdd6f4; border:1px solid #313244; "
            "border-radius:6px; padding:6px; font-size:12px;"
        )
        other_input.setEnabled(False)
        layout.addWidget(other_input)

        def _on_selection_changed():
            row = reason_list.currentRow()
            is_other = row == len(self._REJECT_REASONS) - 1
            other_input.setEnabled(is_other)
            if is_other:
                other_input.setFocus()

        reason_list.currentRowChanged.connect(_on_selection_changed)

        # Boutons
        buttons = QDialogButtonBox()
        confirm_btn = buttons.addButton("Rejeter", QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_btn  = buttons.addButton("Annuler", QDialogButtonBox.ButtonRole.RejectRole)
        confirm_btn.setStyleSheet(
            "background:#e64553; color:white; font-weight:bold; "
            "padding:6px 16px; border-radius:6px;"
        )
        cancel_btn.setStyleSheet(
            "background:#45475a; color:#cdd6f4; font-weight:bold; "
            "padding:6px 16px; border-radius:6px;"
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        row = reason_list.currentRow()
        label, key = self._REJECT_REASONS[row]
        if key == "autre" and other_input.text().strip():
            return other_input.text().strip()
        return label

    def _delete_session(self) -> None:
        """Supprime la session locale défectueuse et revient à l'écran d'attente."""
        if self.session is None:
            QMessageBox.warning(self, "Erreur", "Aucune session chargée.")
            return

        confirm = QMessageBox.question(
            self,
            "Supprimer la session",
            "Supprimer cette session du cache local et revenir à l'écran d'attente ?\n\n"
            "Les fichiers NAS/HDD sources ne seront pas affectés.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        logger.info("Session locale supprimée par l'utilisateur : %s", self.session.session_dir)
        self._cleanup_session_state()
        self._go_to_next_job()

    def _reject_session(self) -> None:
        """Marque la session comme ratée, déplace les fichiers HDFS source vers
        /trash, nettoie la session locale et passe au job suivant."""
        if self.session is None:
            QMessageBox.warning(self, "Erreur", "Aucune session chargée.")
            return

        reason = self._ask_reject_reason()
        if reason is None:
            return  # Annulé par l'utilisateur

        logger.info("Session rejetée manuellement — raison : %s", reason)
        self.statusbar.showMessage(f"Rejet en cours ({reason[:60]})…")

        # Les fichiers source sont sur le NAS en lecture seule (bronze/landing).
        # On ne les déplace pas — ils sont gérés par le pipeline d'ingestion.
        if self._current_job is not None:
            logger.info(
                "Session rejetée : session_id=%s (fichiers NAS conservés dans %s)",
                self._current_job.session_id,
                self._current_job.zone,
            )

        self._cleanup_session_state()
        self._go_to_next_job()

    def _go_to_next_job(self) -> None:
        """Revenir à l'écran d'attente sans démarrer le polling automatique."""
        if self.is_playing:
            self._frame_loader.stop_playback()
            self.is_playing = False

        self._show_waiting_screen()

    def _show_waiting_screen(self) -> None:
        """Affiche l'écran d'attente et lance automatiquement le téléchargement HDD."""
        self._stop_prefetcher()
        if self._poller_thread is not None and self._poller_thread.isRunning():
            self._poller_thread.quit()
            self._poller_thread.wait(2000)
            self._poller_thread = None
        self.stack.setCurrentIndex(0)
        self.waiting_widget.stop_animation()
        self.waiting_widget.set_status("Connexion au serveur HDD…")
        self.waiting_widget.set_queue_info("")
        self.waiting_widget.hide_skip_button()
        self.statusbar.showMessage("HDD : recherche de la prochaine session…")
        self._on_load_from_spool()

    # ------------------------------------------------------------------
    # Mode Annotation / Vérification
    # ------------------------------------------------------------------

    def _show_workspace(self) -> None:
        """Affiche la page de travail correspondant au mode actif (annotation ou vérification)."""
        if self._mode_btn.isChecked():
            self.stack.setCurrentIndex(2)
        else:
            self.stack.setCurrentIndex(1)

    def _on_mode_toggled(self, verification_mode: bool) -> None:
        """Switch entre le mode annotation (page 1) et le mode vérification (page 2)."""
        # Désactiver les raccourcis annotation quand le mode vérification est actif
        # pour que Espace/←/→/Home/End soient capturés par le VerificationWidget.
        for sc in (self._sc_space, self._sc_left, self._sc_right,
                   self._sc_home, self._sc_end):
            sc.setEnabled(not verification_mode)

        if verification_mode:
            self._mode_btn.setText("✏ Mode Annotation")
            if self.session is not None:
                # Alimenter le widget de vérification avec la session courante
                self.verification_widget.load_session(self.session)
                self.verification_widget.set_current_frame(self.current_frame_index, self.session)
                self.verification_widget.set_info(
                    f"Session : {self.session.metadata.session_id}  |  "
                    f"{self.session.frame_count} frames @ {self.session.fps:.0f} FPS"
                )
                self.verification_widget.set_buttons_enabled(True)
                frames = self.session.get_all_frames(self.current_frame_index)
                self.verification_widget.set_frames(frames)
                self.stack.setCurrentIndex(2)
                self.verification_widget.setFocus()
            else:
                # Pas encore de session — lancer le téléchargement automatique depuis le HDD
                self.stack.setCurrentIndex(0)
                self._start_hdd_verification_download()
        else:
            self._mode_btn.setText("🔍 Mode Vérification")
            if self.session is not None:
                self.stack.setCurrentIndex(1)
            else:
                self.stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # HDD verification pipeline
    # ------------------------------------------------------------------

    def _hdd_local_dir(self) -> Path:
        base = Path(self.config.data.cache_dir or (Path.home() / ".cache" / "vive_labeler"))
        return base / "hdd_verification"

    def _make_hdd_worker(self, slot: str) -> HddVerificationWorker:
        """Crée un HddVerificationWorker connecté aux bons slots selon son rôle."""
        hdd = self.config.hdd
        worker = HddVerificationWorker(
            host=hdd.host, port=hdd.port,
            username=hdd.username, password=hdd.password,
            inbox_base=hdd.inbox_base,
            local_dir=self._hdd_local_dir(),
        )
        if slot == "main":
            worker.download_finished.connect(self._on_hdd_download_finished)
            worker.file_progress.connect(self._on_hdd_file_progress)
            worker.no_session_available.connect(self._on_hdd_no_session)
            worker.error_occurred.connect(self._on_hdd_download_error)
        else:  # prefetch
            worker.download_finished.connect(self._on_hdd_prefetch_finished)
            worker.file_progress.connect(self._on_hdd_prefetch_progress)
            worker.no_session_available.connect(self._on_hdd_prefetch_no_session)
            worker.error_occurred.connect(self._on_hdd_prefetch_error)
        return worker

    def _start_hdd_verification_download(self) -> None:
        """Lance le téléchargement de la première session (aucune session active)."""
        hdd = self.config.hdd
        logger.info("HDD verification: connecting to %s inbox=%s", hdd.host, hdd.inbox_base)
        self.verification_widget.set_info("Connexion au serveur HDD…")
        self.verification_widget.set_buttons_enabled(False)
        self._hdd_verification_worker = self._make_hdd_worker("main")
        self._hdd_verification_worker.start()

    def _start_hdd_prefetch(self) -> None:
        """Démarre le téléchargement de la session suivante en arrière-plan."""
        if self._hdd_prefetch_worker is not None and self._hdd_prefetch_worker.isRunning():
            return  # déjà en cours
        self._hdd_prefetched = None
        logger.info("HDD prefetch: starting background download of next session")
        self._hdd_prefetch_worker = self._make_hdd_worker("prefetch")
        self._hdd_prefetch_worker.start()

    # --- Main download callbacks ---

    def _on_hdd_file_progress(self, label: str, done: int, total: int) -> None:
        if total > 0:
            pct = int(done * 100 / total)
            self.statusbar.showMessage(f"Téléchargement HDD : {label} — {pct}%")
        else:
            self.statusbar.showMessage(f"Téléchargement HDD : {label}…")

    def _on_hdd_no_session(self) -> None:
        logger.info("HDD inbox is empty — no session to verify")
        self.verification_widget.set_info("Aucune session disponible dans le HDD inbox.")
        self.statusbar.showMessage("HDD inbox vide — aucune session à vérifier.")

    def _on_hdd_download_error(self, error: str) -> None:
        logger.error("HDD download error: %s", error)
        self.verification_widget.set_info(f"Erreur de téléchargement HDD :\n{error}")
        self.statusbar.showMessage(f"Erreur HDD : {error}")
        QMessageBox.warning(self, "Erreur HDD",
            f"Impossible de télécharger la session depuis le serveur HDD :\n{error}")

    def _on_hdd_download_finished(self, session_id: str, local_files: object) -> None:
        """Session principale téléchargée — charger, afficher, lancer le prefetch."""
        logger.info("HDD download finished: %s", session_id)
        self._hdd_current_session_id = session_id
        self._hdd_session_decided = False
        self.statusbar.showMessage(f"Session '{session_id}' prête.")
        self._load_session(str(local_files.tracker.parent))
        if self._mode_btn.isChecked():
            self.stack.setCurrentIndex(2)
        # Démarrer immédiatement le téléchargement de la session suivante
        self._start_hdd_prefetch()

    # --- Prefetch callbacks ---

    def _on_hdd_prefetch_progress(self, label: str, done: int, total: int) -> None:
        if total > 0:
            pct = int(done * 100 / total)
            self.statusbar.showMessage(
                f"Session courante : {self._hdd_current_session_id or '?'}  |  "
                f"Prefetch : {label} {pct}%"
            )

    def _on_hdd_prefetch_no_session(self) -> None:
        logger.info("HDD prefetch: inbox vide, aucune session suivante disponible")
        self._hdd_prefetched = None
        # Si on attend déjà → afficher le message vide
        if self._hdd_waiting_for_prefetch:
            self._hdd_waiting_for_prefetch = False
            self.verification_widget.set_info("Aucune session suivante dans le HDD inbox.")
            self.statusbar.showMessage("HDD inbox vide après cette session.")

    def _on_hdd_prefetch_error(self, error: str) -> None:
        logger.warning("HDD prefetch error (non-blocking): %s", error)
        self._hdd_prefetched = None
        if self._hdd_waiting_for_prefetch:
            self._hdd_waiting_for_prefetch = False
            # Retry main download
            self.session = None
            self._hdd_current_session_id = None
            self.stack.setCurrentIndex(0)
            self._start_hdd_verification_download()

    def _on_hdd_prefetch_finished(self, session_id: str, local_files: object) -> None:
        """Session prefetchée prête — stocker ou charger immédiatement si on attend."""
        logger.info("HDD prefetch finished: %s", session_id)
        self._hdd_prefetched = (session_id, local_files)
        if self._hdd_waiting_for_prefetch:
            self._hdd_waiting_for_prefetch = False
            self._load_prefetched_session()

    def _load_prefetched_session(self) -> None:
        """Charge la session prefetchée comme session courante et lance le prefetch suivant."""
        if self._hdd_prefetched is None:
            return
        session_id, local_files = self._hdd_prefetched
        self._hdd_prefetched = None
        self._hdd_current_session_id = session_id
        self._hdd_session_decided = False
        logger.info("Loading prefetched session: %s", session_id)
        self.statusbar.showMessage(f"Session '{session_id}' prête (prefetch).")
        self._load_session(str(local_files.tracker.parent))
        if self._mode_btn.isChecked():
            self.stack.setCurrentIndex(2)
        # Enchaîner le prefetch de la session d'après
        self._start_hdd_prefetch()

    # --- Upload ---

    def _upload_hdd_session(self, dest_base: str) -> None:
        """Envoie la session courante vers dest_base sur le HDD."""
        if self.session is None:
            return
        session_dir = self.session.session_dir
        session_id = self._hdd_current_session_id or session_dir.name
        hdd = self.config.hdd
        self._hdd_upload_worker = HddUploadWorker(
            local_dir=Path(session_dir),
            dest_base=dest_base,
            session_id=session_id,
            host=hdd.host, port=hdd.port,
            username=hdd.username, password=hdd.password,
            delete_after=False,
        )
        self.statusbar.showMessage(f"Upload HDD en cours → {dest_base}/{session_id}…")
        self._hdd_upload_worker.upload_finished.connect(
            lambda dest: self.statusbar.showMessage(f"✓ Session envoyée → {dest}")
        )
        self._hdd_upload_worker.error_occurred.connect(
            lambda err: (
                self.statusbar.showMessage("✗ Échec de l'upload HDD"),
                QMessageBox.critical(
                    self, "Erreur upload HDD",
                    f"Impossible d'envoyer la session vers :\n{dest_base}/{session_id}\n\n{err}"
                ),
            )
        )
        self._hdd_upload_worker.start()

    # --- Validate / Reject ---

    def _on_verification_validate(self) -> None:
        self.verification_widget.set_buttons_enabled(False)
        logger.info("Verification: session validated")
        self._hdd_session_decided = True
        self._upload_hdd_session(self.config.hdd.silver_base)
        self._advance_to_next_hdd_session("✓ Session validée")

    def _on_verification_reject(self) -> None:
        self.verification_widget.set_buttons_enabled(False)
        logger.info("Verification: session rejected")
        self._hdd_session_decided = True
        self._upload_hdd_session(self.config.hdd.retry_base)
        self._advance_to_next_hdd_session("✕ Session rejetée")

    def _advance_to_next_hdd_session(self, status: str) -> None:
        """Passe à la session suivante : immédiate si prefetch prêt, sinon attend."""
        if self._hdd_prefetched is not None:
            # Déjà téléchargée — chargement instantané
            self.verification_widget.set_info(f"{status} — chargement de la session suivante…")
            self.statusbar.showMessage(f"{status} — session suivante disponible immédiatement.")
            self._load_prefetched_session()
        elif self._hdd_prefetch_worker is not None and self._hdd_prefetch_worker.isRunning():
            # En cours de téléchargement — attendre la fin
            self._hdd_waiting_for_prefetch = True
            self.verification_widget.set_info(
                f"{status} — téléchargement de la session suivante en cours…"
            )
            self.statusbar.showMessage(f"{status} — en attente du prefetch…")
        else:
            # Pas de prefetch (inbox était vide ou erreur) — relancer un téléchargement
            self.verification_widget.set_info(f"{status} — recherche de la prochaine session…")
            self.session = None
            self._hdd_current_session_id = None
            self.stack.setCurrentIndex(0)
            self._start_hdd_verification_download()

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _show_shortcuts(self) -> None:
        QMessageBox.information(
            self, "Raccourcis clavier",
            "NAVIGATION\n"
            "  Space        — Lecture / Pause\n"
            "  ← / →        — Reculer / Avancer (pas variable)\n"
            "  ↑ / ↓        — Augmenter / Réduire le pas\n"
            "  Home / End   — Premier / Dernier frame\n"
            "\n"
            "ANNOTATIONS\n"
            "  I            — Définir point IN\n"
            "  O            — Définir point OUT\n"
            "\n"
            "AFFICHAGE\n"
            "  T            — Afficher / Masquer les trackers 3D\n"
            "  G            — Afficher / Masquer le graphe pince\n"
            "\n"
            "FICHIERS\n"
            "  Ctrl+S       — Sauvegarder les annotations\n"
            "  Ctrl+O       — Ouvrir un répertoire de session\n"
            "  Ctrl+Q       — Quitter\n"
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About VIVE Labeler",
            "VIVE Labeler v1.0.0\n\n"
            "State-of-the-art multi-camera annotation tool for\n"
            "synchronized video, VIVE tracking, and gripper data.\n\n"
            "Features:\n"
            "- 3-camera synchronized playback\n"
            "- Real-time sensor data visualization\n"
            "- Multi-track interval annotations\n"
            "- 3D tracker trajectory view\n"
            "- JSON / CSV / LeRobot export\n"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _collect_hdd_sessions_to_return(self) -> list[tuple[Path, str]]:
        """Retourne la liste (local_dir, session_id) des sessions à renvoyer au HDD inbox.

        N'inclut pas la session courante si elle a déjà été validée/rejetée
        (elle part vers send/retry, pas vers l'inbox).
        """
        sessions = []

        # Session en cours de vérification — seulement si pas encore décidée
        if (self._hdd_current_session_id
                and self.session is not None
                and not self._hdd_session_decided):
            session_dir = Path(self.session.session_dir)
            if session_dir.exists():
                sessions.append((session_dir, self._hdd_current_session_id))

        # Session prefetchée (déjà téléchargée, pas encore affichée) — toujours à renvoyer
        if self._hdd_prefetched is not None:
            sid, local_files = self._hdd_prefetched
            local_dir = local_files.tracker.parent
            if local_dir.exists() and sid != (sessions[0][1] if sessions else None):
                sessions.append((local_dir, sid))

        return sessions

    def _return_sessions_to_hdd(self) -> None:
        """Renvoie les sessions locales vers le HDD inbox avant fermeture.

        Lance un upload synchrone (subprocess.wait) avec un dialog de progression.
        Si aucune session HDD n'est en cours, ne fait rien.
        """
        sessions = self._collect_hdd_sessions_to_return()
        if not sessions:
            return

        hdd = self.config.hdd

        # Arrêter le worker prefetch pour éviter qu'il modifie les dossiers pendant l'upload
        if self._hdd_prefetch_worker is not None and self._hdd_prefetch_worker.isRunning():
            self._hdd_prefetch_worker.cancel()
            self._hdd_prefetch_worker.wait(5000)

        # Dialog de progression
        dlg = QDialog(self)
        dlg.setWindowTitle("Renvoi des sessions au HDD")
        dlg.setModal(True)
        dlg.setFixedSize(420, 120)
        dlg.setStyleSheet("background: #1e1e2e; color: #cdd6f4;")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lbl = QLabel(f"Renvoi de {len(sessions)} session(s) vers {hdd.inbox_base}…")
        lbl.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        lay.addWidget(lbl)
        sub_lbl = QLabel("")
        sub_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        lay.addWidget(sub_lbl)
        dlg.show()
        QApplication.processEvents()

        errors = []
        for local_dir, session_id in sessions:
            sub_lbl.setText(f"Upload : {session_id}…")
            QApplication.processEvents()
            logger.info("Closing: returning session '%s' to HDD inbox", session_id)
            try:
                proc = upload_directory_sftp_background(
                    local_dir=local_dir,
                    nas_dest=f"{hdd.inbox_base.rstrip('/')}/{session_id}",
                    host=hdd.host,
                    port=hdd.port,
                    username=hdd.username,
                    password=hdd.password,
                    key_path=None,
                    delete_after=False,  # ne pas supprimer le cache local ici
                )
                # Attendre la fin de l'upload (bloquant)
                while proc.poll() is None:
                    QApplication.processEvents()
                if proc.returncode != 0:
                    out = proc.stdout.read() if proc.stdout else ""
                    errors.append(f"{session_id}: exit {proc.returncode}\n{out}")
                    logger.error("Return upload failed for %s (exit %d)", session_id, proc.returncode)
                else:
                    logger.info("Session '%s' returned to HDD inbox successfully", session_id)
            except Exception as exc:
                errors.append(f"{session_id}: {exc}")
                logger.error("Return upload error for %s: %s", session_id, exc)

        dlg.close()

        if errors:
            QMessageBox.warning(
                self, "Erreur renvoi HDD",
                "Certaines sessions n'ont pas pu être renvoyées au HDD :\n\n"
                + "\n".join(errors),
            )

    def closeEvent(self, event) -> None:
        if self.is_playing:
            self._frame_loader.stop_playback()
        # Stop async frame loader
        self._frame_loader.stop()
        self._frame_loader.wait(1000)
        if self._poller_thread and self._poller_thread.isRunning():
            self._poller_thread.stop()
            self._poller_thread.wait(2000)
        # Stop prefetcher and discard any pre-downloaded scenario to free disk
        if self._prefetcher is not None:
            self._prefetcher.stop()
            self._prefetcher.wait(3000)
            self._prefetcher.discard()
            self._prefetcher = None
        # # Stop Seqensor worker if still running
        # if self._seqensor_worker is not None and self._seqensor_worker.isRunning():
        #     self._seqensor_worker.quit()
        #     self._seqensor_worker.wait(2000)
        #     self._seqensor_worker = None

        # Renvoyer les sessions HDD téléchargées vers l'inbox avant de fermer
        self._return_sessions_to_hdd()

        if self.session:
            self.session.release()
        if self.mongo_client:
            self.mongo_client.close()

        # Les sous-process d'upload continuent de tourner après fermeture
        # (ils sont indépendants du process principal).
        if self._bg_upload_procs:
            running = [p for p in self._bg_upload_procs if p.poll() is None]
            if running:
                logger.info(
                    "%d upload(s) HDFS encore en cours (PID %s) — ils "
                    "se termineront indépendamment de l'application.",
                    len(running),
                    [p.pid for p in running],
                )

        event.accept()
