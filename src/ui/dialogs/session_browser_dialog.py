"""Session browser dialog — browse local filesystem or AWS S3 to open a session."""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QTreeView,
    QLineEdit,
    QPushButton,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QAbstractItemView,
)
from PyQt6.QtCore import (
    Qt,
    QDir,
    QThread,
    pyqtSignal,
    QSortFilterProxyModel,
    QModelIndex,
)
from PyQt6.QtGui import QFont, QStandardItemModel, QStandardItem, QFileSystemModel

from ...utils.config import S3Config, HDDConfig
from ...storage.s3_client import S3Client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 background workers
# ---------------------------------------------------------------------------

class S3ListWorker(QThread):
    """List one level of an S3 prefix in a background thread."""

    entries_ready = pyqtSignal(str, list)   # (prefix, [{"name","type","size","full_prefix"}, ...])
    error_occurred = pyqtSignal(str, str)   # (prefix, error_message)

    def __init__(self, s3_client: S3Client, prefix: str, parent=None):
        super().__init__(parent)
        self._client = s3_client
        self._prefix = prefix

    def run(self) -> None:
        try:
            entries = self._client.list_prefixes(self._prefix)
            self.entries_ready.emit(self._prefix, entries)
        except Exception as exc:
            self.error_occurred.emit(self._prefix, str(exc))


class S3DownloadSessionWorker(QThread):
    """Download an entire S3 session to a local temp folder.

    All files are downloaded in parallel (one thread per file).
    Progress is reported via progress_update(file_name, bytes_done, bytes_total).
    On success, finished(local_path) is emitted with the temp directory path.
    """

    finished = pyqtSignal(str)                    # local_path as string
    error_occurred = pyqtSignal(str)              # error message
    progress_update = pyqtSignal(str, int, int)   # (filename, bytes_done, bytes_total)
    file_done = pyqtSignal(str, int)              # (filename, files_done_count)

    def __init__(self, s3_client: S3Client, session_prefix: str, parent=None):
        super().__init__(parent)
        self._client = s3_client
        self._prefix = session_prefix
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        import tempfile
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            # 1. Resolve all S3 keys for this session
            paths = self._client.resolve_session_paths(self._prefix)
            session_name = self._prefix.rstrip("/").split("/")[-1]

            # 2. Create local temp directory
            local_root = Path(tempfile.mkdtemp(prefix="vive_labeler_s3_")) / session_name
            local_root.mkdir(parents=True, exist_ok=True)
            (local_root / "videos").mkdir(exist_ok=True)

            # 3. Build download task list: (s3_key, local_path)
            tasks = [
                (paths.metadata,       local_root / "metadata.json"),
                (paths.tracker,        local_root / "tracker_positions.csv"),
                (paths.gripper_left,   local_root / "gripper_left_data.csv"),
                (paths.gripper_right,  local_root / "gripper_right_data.csv"),
                (paths.cam_head,       local_root / "videos" / "head.mp4"),
                (paths.cam_left,       local_root / "videos" / "left.mp4"),
                (paths.cam_right,      local_root / "videos" / "right.mp4"),
                (paths.cam_head_jsonl, local_root / "videos" / "head.jsonl"),
                (paths.cam_left_jsonl, local_root / "videos" / "left.jsonl"),
                (paths.cam_right_jsonl, local_root / "videos" / "right.jsonl"),
            ]
            tasks = [(k, p) for k, p in tasks if k]  # skip empty keys

            done_count = 0
            lock = threading.Lock()

            def _download_one(key: str, dest: Path) -> None:
                nonlocal done_count
                if self._cancelled:
                    return
                filename = dest.name
                # Get object size for progress
                try:
                    head = self._client._s3.head_object(
                        Bucket=self._client.bucket, Key=key
                    )
                    total = head.get("ContentLength", 0)
                except Exception:
                    total = 0

                self.progress_update.emit(filename, 0, total)

                # Stream download with progress
                resp = self._client._s3.get_object(
                    Bucket=self._client.bucket, Key=key
                )
                body = resp["Body"]
                chunk_size = 256 * 1024  # 256 KB
                downloaded = 0
                with open(dest, "wb") as f:
                    while not self._cancelled:
                        chunk = body.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress_update.emit(filename, downloaded, total or downloaded)

                if not self._cancelled:
                    with lock:
                        done_count += 1
                        self.file_done.emit(filename, done_count)

            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {pool.submit(_download_one, k, p): p.name for k, p in tasks}
                for future in as_completed(futures):
                    if self._cancelled:
                        break
                    try:
                        future.result()
                    except Exception as exc:
                        if not self._cancelled:
                            self.error_occurred.emit(str(exc))
                            return

            if not self._cancelled:
                self.finished.emit(str(local_root))

        except Exception as exc:
            if not self._cancelled:
                self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# HDD (SFTP archive) background workers
# ---------------------------------------------------------------------------

class HDDListWorker(QThread):
    """List one directory on the HDD SFTP server."""

    entries_ready = pyqtSignal(str, list)   # (path, [{"name", "type", "full_path"}, ...])
    error_occurred = pyqtSignal(str, str)   # (path, error_message)

    def __init__(self, config: "HDDConfig", remote_path: str, parent=None):
        super().__init__(parent)
        self._config = config
        self._path = remote_path

    def run(self) -> None:
        import paramiko
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                self._config.host,
                port=self._config.port,
                username=self._config.username,
                password=self._config.password,
                timeout=10,
            )
            sftp = ssh.open_sftp()
            entries = []
            for attr in sftp.listdir_attr(self._path):
                import stat as stat_mod
                is_dir = stat_mod.S_ISDIR(attr.st_mode)
                full = self._path.rstrip("/") + "/" + attr.filename
                entries.append({
                    "name": attr.filename,
                    "type": "DIRECTORY" if is_dir else "FILE",
                    "full_path": full,
                })
            entries.sort(key=lambda e: (e["type"] == "FILE", e["name"]))
            sftp.close()
            ssh.close()
            self.entries_ready.emit(self._path, entries)
        except Exception as exc:
            self.error_occurred.emit(self._path, str(exc))


class HDDDownloadSessionWorker(QThread):
    """Download a session folder from HDD SFTP to a local temp directory."""

    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str, int, int)   # (filename, done, total)
    file_done = pyqtSignal(str, int)

    def __init__(self, config: "HDDConfig", remote_path: str, parent=None):
        super().__init__(parent)
        self._config = config
        self._remote_path = remote_path
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        import paramiko, tempfile, stat as stat_mod

        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                self._config.host,
                port=self._config.port,
                username=self._config.username,
                password=self._config.password,
                timeout=10,
            )
            sftp = ssh.open_sftp()

            session_name = self._remote_path.rstrip("/").split("/")[-1]
            local_root = Path(tempfile.mkdtemp(prefix="vive_labeler_hdd_")) / session_name
            local_root.mkdir(parents=True, exist_ok=True)

            # Collect all files recursively
            all_files: list[tuple[str, Path]] = []

            def _collect(remote_dir: str, local_dir: Path) -> None:
                local_dir.mkdir(parents=True, exist_ok=True)
                for attr in sftp.listdir_attr(remote_dir):
                    remote_full = remote_dir.rstrip("/") + "/" + attr.filename
                    local_full = local_dir / attr.filename
                    if stat_mod.S_ISDIR(attr.st_mode):
                        _collect(remote_full, local_full)
                    else:
                        all_files.append((remote_full, local_full))

            _collect(self._remote_path, local_root)

            done_count = 0
            for remote_file, local_file in all_files:
                if self._cancelled:
                    break
                filename = local_file.name
                try:
                    stat = sftp.stat(remote_file)
                    total = stat.st_size or 0
                except Exception:
                    total = 0

                self.progress_update.emit(filename, 0, total)

                def _progress_cb(done: int, tot: int, fn=filename, t=total) -> None:
                    self.progress_update.emit(fn, done, tot or t)

                sftp.get(remote_file, str(local_file), callback=_progress_cb)
                done_count += 1
                self.file_done.emit(filename, done_count)

            sftp.close()
            ssh.close()

            if not self._cancelled:
                self.finished.emit(str(local_root))

        except Exception as exc:
            if not self._cancelled:
                self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

_STYLE = """
QDialog { background: #1e1e2e; }
QTabWidget::pane {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 6px;
}
QTabBar::tab {
    background: #181825; color: #a6adc8;
    padding: 8px 20px; border-radius: 4px 4px 0 0;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #313244; color: #cdd6f4; }
QTabBar::tab:hover    { background: #313244; }
QLabel { color: #cdd6f4; font-size: 13px; }
QLineEdit {
    background: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 6px;
    padding: 6px 10px; font-size: 12px;
}
QLineEdit:focus { border: 1px solid #89b4fa; }
QTreeView {
    background: #181825; color: #cdd6f4;
    border: 1px solid #313244; border-radius: 6px;
    alternate-background-color: #1e1e2e;
    selection-background-color: #313244;
}
QTreeView::item:hover    { background: #2a2a3e; }
QTreeView::item:selected { background: #45475a; color: #cdd6f4; }
QHeaderView::section {
    background: #181825; color: #a6adc8;
    border: none; padding: 4px 8px; font-size: 11px;
}
QPushButton {
    background: #89b4fa; color: #1e1e2e; border: none;
    border-radius: 6px; padding: 10px 24px;
    font-size: 13px; font-weight: bold;
}
QPushButton:hover   { background: #74c7ec; }
QPushButton:pressed { background: #89dceb; }
QPushButton:disabled { background: #45475a; color: #6c7086; }
QPushButton#cancel_btn { background: #45475a; color: #cdd6f4; }
QPushButton#cancel_btn:hover { background: #585b70; }
QPushButton#refresh_btn {
    background: #313244; color: #a6adc8;
    border: 1px solid #45475a; padding: 6px 14px;
    font-size: 12px; font-weight: normal;
}
QPushButton#refresh_btn:hover { background: #45475a; color: #cdd6f4; }
QScrollBar:vertical {
    background: #181825; width: 8px; border: none;
}
QScrollBar::handle:vertical {
    background: #45475a; border-radius: 4px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class SessionBrowserDialog(QDialog):
    """Modal dialog to browse and open a session (local filesystem or AWS S3).

    Signals:
        session_selected(str): emitted with the local path on success
                               (both local and S3 sessions use this signal).
    """

    session_selected = pyqtSignal(str)       # local path (local file or S3 download)

    def __init__(
        self,
        s3_config: Optional[S3Config] = None,
        hdd_config: Optional["HDDConfig"] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._s3_config = s3_config
        self._hdd_config = hdd_config
        self._s3_client: Optional[S3Client] = None
        if s3_config:
            self._s3_client = S3Client(
                bucket=s3_config.bucket,
                region=s3_config.region,
                bronze_prefix=s3_config.bronze_prefix,
                aws_access_key_id=s3_config.aws_access_key_id,
                aws_secret_access_key=s3_config.aws_secret_access_key,
                presign_ttl=s3_config.presign_ttl,
            )

        self._selected_local_path: Optional[str] = None
        self._selected_s3_prefix: Optional[str] = None
        self._selected_hdd_path: Optional[str] = None

        self._s3_model = QStandardItemModel()
        self._hdd_model = QStandardItemModel()
        self._list_workers: list = []
        self._download_worker = None
        self._total_files = 0
        self._done_files = 0

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle("Ouvrir une session")
        self.setMinimumSize(800, 560)
        self.resize(860, 620)
        self.setModal(True)
        self.setStyleSheet(_STYLE)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(12)

        title = QLabel("Rechercher une session")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #cdd6f4; margin-bottom: 4px;")
        root_layout.addWidget(title)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, stretch=1)

        local_tab = QWidget()
        self.tabs.addTab(local_tab, "  Local  ")
        self._build_local_tab(local_tab)

        if self._s3_client:
            s3_tab = QWidget()
            self.tabs.addTab(s3_tab, "  S3  ")
            self._build_s3_tab(s3_tab)

        if self._hdd_config:
            hdd_tab = QWidget()
            self.tabs.addTab(hdd_tab, "  HDD  ")
            self._build_hdd_tab(hdd_tab)

        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #313244;")
        root_layout.addWidget(sep)

        self._path_label = QLabel("Aucune sélection")
        self._path_label.setStyleSheet("color: #6c7086; font-size: 11px; font-style: italic;")
        self._path_label.setWordWrap(True)
        root_layout.addWidget(self._path_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        cancel_btn = QPushButton("Annuler")
        cancel_btn.setObjectName("cancel_btn")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        btn_row.addStretch()

        self._load_btn = QPushButton("Charger la session")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load)
        btn_row.addWidget(self._load_btn)

        root_layout.addLayout(btn_row)

        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_local_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filtrer :"))
        self._local_filter = QLineEdit()
        self._local_filter.setPlaceholderText("Nom de dossier…")
        self._local_filter.setClearButtonEnabled(True)
        filter_row.addWidget(self._local_filter)
        layout.addLayout(filter_row)

        self._fs_model = QFileSystemModel()
        self._fs_model.setRootPath(QDir.rootPath())
        self._fs_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot)

        self._local_proxy = QSortFilterProxyModel()
        self._local_proxy.setSourceModel(self._fs_model)
        self._local_proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._local_proxy.setRecursiveFilteringEnabled(True)

        self._local_tree = QTreeView()
        self._local_tree.setModel(self._local_proxy)
        self._local_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._local_tree.setAnimated(True)
        self._local_tree.setSortingEnabled(True)
        self._local_tree.hideColumn(1)
        self._local_tree.hideColumn(2)
        self._local_tree.hideColumn(3)
        self._local_tree.header().setStretchLastSection(True)

        root_idx = self._fs_model.index(QDir.rootPath())
        self._local_tree.setRootIndex(self._local_proxy.mapFromSource(root_idx))
        home_idx = self._fs_model.index(str(Path.home()))
        home_proxy = self._local_proxy.mapFromSource(home_idx)
        self._local_tree.scrollTo(home_proxy)
        self._local_tree.expand(home_proxy)

        layout.addWidget(self._local_tree, stretch=1)

        self._local_tree.selectionModel().currentChanged.connect(self._on_local_selection_changed)
        self._local_filter.textChanged.connect(self._local_proxy.setFilterWildcard)

    def _build_s3_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)

        # Connection info + refresh button
        top_row = QHBoxLayout()
        conn_lbl = QLabel(
            f"S3 : s3://{self._s3_config.bucket}/{self._s3_config.bronze_prefix}  "
            f"({self._s3_config.region})"
        )
        conn_lbl.setStyleSheet("color: #6c7086; font-size: 11px;")
        top_row.addWidget(conn_lbl)
        top_row.addStretch()

        self._s3_refresh_btn = QPushButton("Actualiser")
        self._s3_refresh_btn.setObjectName("refresh_btn")
        self._s3_refresh_btn.clicked.connect(self._s3_refresh_root)
        top_row.addWidget(self._s3_refresh_btn)
        layout.addLayout(top_row)

        # Filter bar
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filtrer :"))
        self._s3_filter = QLineEdit()
        self._s3_filter.setPlaceholderText("Nom de scénario…")
        self._s3_filter.setClearButtonEnabled(True)
        filter_row.addWidget(self._s3_filter)
        layout.addLayout(filter_row)

        # S3 tree (lazy QStandardItemModel)
        self._s3_model.setHorizontalHeaderLabels(["Nom", "Type", "Taille"])

        self._s3_proxy = QSortFilterProxyModel()
        self._s3_proxy.setSourceModel(self._s3_model)
        self._s3_proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._s3_proxy.setRecursiveFilteringEnabled(True)
        self._s3_proxy.setFilterKeyColumn(0)

        self._s3_tree = QTreeView()
        self._s3_tree.setModel(self._s3_proxy)
        self._s3_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._s3_tree.setAnimated(True)
        layout.addWidget(self._s3_tree, stretch=1)

        self._s3_status = QLabel("")
        self._s3_status.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(self._s3_status)

        self._s3_tree.selectionModel().currentChanged.connect(self._on_s3_selection_changed)
        self._s3_tree.expanded.connect(self._on_s3_item_expanded)
        self._s3_filter.textChanged.connect(self._s3_proxy.setFilterWildcard)

    def _build_hdd_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        conn_lbl = QLabel(
            f"HDD : sftp://{self._hdd_config.host}:{self._hdd_config.port}"
            f"  {self._hdd_config.inbox_base}"
        )
        conn_lbl.setStyleSheet("color: #6c7086; font-size: 11px;")
        top_row.addWidget(conn_lbl)
        top_row.addStretch()

        self._hdd_refresh_btn = QPushButton("Actualiser")
        self._hdd_refresh_btn.setObjectName("refresh_btn")
        self._hdd_refresh_btn.clicked.connect(self._hdd_refresh_root)
        top_row.addWidget(self._hdd_refresh_btn)
        layout.addLayout(top_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filtrer :"))
        self._hdd_filter = QLineEdit()
        self._hdd_filter.setPlaceholderText("Nom de session…")
        self._hdd_filter.setClearButtonEnabled(True)
        filter_row.addWidget(self._hdd_filter)
        layout.addLayout(filter_row)

        self._hdd_model.setHorizontalHeaderLabels(["Nom", "Type"])

        self._hdd_proxy = QSortFilterProxyModel()
        self._hdd_proxy.setSourceModel(self._hdd_model)
        self._hdd_proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._hdd_proxy.setRecursiveFilteringEnabled(True)
        self._hdd_proxy.setFilterKeyColumn(0)

        self._hdd_tree = QTreeView()
        self._hdd_tree.setModel(self._hdd_proxy)
        self._hdd_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._hdd_tree.setAnimated(True)
        layout.addWidget(self._hdd_tree, stretch=1)

        self._hdd_status = QLabel("")
        self._hdd_status.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(self._hdd_status)

        self._hdd_tree.selectionModel().currentChanged.connect(self._on_hdd_selection_changed)
        self._hdd_tree.expanded.connect(self._on_hdd_item_expanded)
        self._hdd_filter.textChanged.connect(self._hdd_proxy.setFilterWildcard)

    def _hdd_refresh_root(self) -> None:
        self._hdd_model.clear()
        self._hdd_model.setHorizontalHeaderLabels(["Nom", "Type"])
        self._hdd_status.setText("Connexion au HDD…")
        self._hdd_refresh_btn.setEnabled(False)

        worker = HDDListWorker(self._hdd_config, self._hdd_config.inbox_base, parent=self)
        worker.entries_ready.connect(self._on_hdd_root_entries)
        worker.error_occurred.connect(self._on_hdd_list_error)
        worker.finished.connect(worker.deleteLater)
        self._list_workers.append(worker)
        worker.start()

    def _on_hdd_root_entries(self, path: str, entries: list) -> None:
        self._hdd_status.setText(path)
        self._hdd_refresh_btn.setEnabled(True)
        root = self._hdd_model.invisibleRootItem()
        for entry in entries:
            self._append_hdd_entry(root, entry)

    def _append_hdd_entry(self, parent_item: QStandardItem, entry: dict) -> None:
        name_item = QStandardItem(entry["name"])
        name_item.setData(entry["full_path"], Qt.ItemDataRole.UserRole)
        name_item.setEditable(False)

        if entry["type"] == "DIRECTORY":
            placeholder = QStandardItem("Chargement…")
            placeholder.setData("__placeholder__", Qt.ItemDataRole.UserRole)
            placeholder.setEditable(False)
            name_item.appendRow(placeholder)

        type_item = QStandardItem("Dossier" if entry["type"] == "DIRECTORY" else "Fichier")
        type_item.setEditable(False)
        parent_item.appendRow([name_item, type_item])

    def _on_hdd_item_expanded(self, proxy_index: QModelIndex) -> None:
        source_idx = self._hdd_proxy.mapToSource(proxy_index)
        item = self._hdd_model.itemFromIndex(source_idx)
        if item is None:
            return
        if item.rowCount() == 1:
            child = item.child(0)
            if child and child.data(Qt.ItemDataRole.UserRole) == "__placeholder__":
                item.removeRow(0)
                path = item.data(Qt.ItemDataRole.UserRole)
                self._hdd_status.setText(f"Chargement : {path}")
                worker = HDDListWorker(self._hdd_config, path, parent=self)
                worker.entries_ready.connect(
                    lambda p, entries, parent=item: self._on_hdd_children(p, entries, parent)
                )
                worker.error_occurred.connect(self._on_hdd_list_error)
                worker.finished.connect(worker.deleteLater)
                self._list_workers.append(worker)
                worker.start()

    def _on_hdd_children(self, path: str, entries: list, parent_item: QStandardItem) -> None:
        self._hdd_status.setText("")
        for entry in entries:
            self._append_hdd_entry(parent_item, entry)

    def _on_hdd_selection_changed(self, current: QModelIndex, _) -> None:
        source_idx = self._hdd_proxy.mapToSource(current)
        item = self._hdd_model.itemFromIndex(source_idx)
        if item is None:
            self._load_btn.setEnabled(False)
            return

        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or path == "__placeholder__":
            self._selected_hdd_path = None
            self._load_btn.setEnabled(False)
            return

        name = path.rstrip("/").split("/")[-1]
        is_session = name.startswith("session_")
        self._selected_hdd_path = path if is_session else None
        self._selected_local_path = None
        self._selected_s3_prefix = None

        if is_session:
            self._path_label.setText(
                f"sftp://{self._hdd_config.host}{path}"
            )
            self._path_label.setStyleSheet("color: #a6e3a1; font-size: 11px;")
            self._load_btn.setEnabled(True)
        else:
            self._path_label.setText("Sélectionnez un dossier session_*")
            self._path_label.setStyleSheet("color: #6c7086; font-size: 11px; font-style: italic;")
            self._load_btn.setEnabled(False)

    def _on_hdd_list_error(self, path: str, error: str) -> None:
        self._hdd_status.setText(f"Erreur : {error[:120]}")
        self._hdd_refresh_btn.setEnabled(True)
        logger.error("HDD browse error at %s: %s", path, error)

    # ------------------------------------------------------------------
    # Tab switching
    # ------------------------------------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        self._selected_local_path = None
        self._selected_s3_prefix = None
        self._selected_hdd_path = None
        self._path_label.setText("Aucune sélection")
        self._path_label.setStyleSheet("color: #6c7086; font-size: 11px; font-style: italic;")
        self._load_btn.setEnabled(False)
        # Lazy-connect to S3 when the S3 tab is first opened
        if self._s3_client and index == 1 and self._s3_model.rowCount() == 0:
            self._s3_refresh_root()
        # Lazy-connect to HDD when HDD tab is first opened
        tab_widget = self.tabs.widget(index)
        if (
            self._hdd_config
            and tab_widget is not None
            and self.tabs.tabText(index).strip() == "HDD"
            and self._hdd_model.rowCount() == 0
        ):
            self._hdd_refresh_root()

    # ------------------------------------------------------------------
    # Local tab
    # ------------------------------------------------------------------

    def _on_local_selection_changed(self, current: QModelIndex, _) -> None:
        source_idx = self._local_proxy.mapToSource(current)
        path = self._fs_model.filePath(source_idx)
        if path:
            self._selected_local_path = path
            self._selected_s3_prefix = None
            self._path_label.setText(path)
            self._path_label.setStyleSheet("color: #89b4fa; font-size: 11px;")
            self._load_btn.setEnabled(True)
        else:
            self._selected_local_path = None
            self._load_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # S3 tab — listing
    # ------------------------------------------------------------------

    def _s3_refresh_root(self) -> None:
        self._s3_model.clear()
        self._s3_model.setHorizontalHeaderLabels(["Nom", "Type", "Taille"])
        self._s3_status.setText("Connexion à S3…")
        self._s3_refresh_btn.setEnabled(False)

        root_prefix = self._s3_config.bronze_prefix + "/"
        worker = S3ListWorker(self._s3_client, root_prefix, parent=self)
        worker.entries_ready.connect(self._on_s3_root_entries)
        worker.error_occurred.connect(self._on_s3_list_error)
        worker.finished.connect(worker.deleteLater)
        self._list_workers.append(worker)
        worker.start()

    def _on_s3_root_entries(self, prefix: str, entries: list) -> None:
        self._s3_status.setText(f"s3://{self._s3_config.bucket}/{prefix}")
        self._s3_refresh_btn.setEnabled(True)
        root = self._s3_model.invisibleRootItem()

        for entry in entries:
            self._append_entry(root, entry)

    def _append_entry(self, parent_item: QStandardItem, entry: dict) -> None:
        name_item = QStandardItem(entry["name"])
        name_item.setData(entry["full_prefix"], Qt.ItemDataRole.UserRole)
        name_item.setEditable(False)

        if entry["type"] == "DIRECTORY":
            placeholder = QStandardItem("Chargement…")
            placeholder.setData("__placeholder__", Qt.ItemDataRole.UserRole)
            placeholder.setEditable(False)
            name_item.appendRow(placeholder)

        type_item = QStandardItem("Dossier" if entry["type"] == "DIRECTORY" else "Fichier")
        type_item.setEditable(False)

        size_str = (
            f"{entry['size'] / 1_000_000:.1f} Mo"
            if entry["type"] == "FILE" and entry["size"] > 0
            else "—"
        )
        size_item = QStandardItem(size_str)
        size_item.setEditable(False)

        parent_item.appendRow([name_item, type_item, size_item])

    def _on_s3_item_expanded(self, proxy_index: QModelIndex) -> None:
        source_idx = self._s3_proxy.mapToSource(proxy_index)
        item = self._s3_model.itemFromIndex(source_idx)
        if item is None:
            return

        if item.rowCount() == 1:
            child = item.child(0)
            if child and child.data(Qt.ItemDataRole.UserRole) == "__placeholder__":
                item.removeRow(0)
                prefix = item.data(Qt.ItemDataRole.UserRole)
                self._s3_status.setText(f"Chargement : {prefix}")

                worker = S3ListWorker(self._s3_client, prefix, parent=self)
                worker.entries_ready.connect(
                    lambda p, entries, parent=item: self._on_s3_children(p, entries, parent)
                )
                worker.error_occurred.connect(self._on_s3_list_error)
                worker.finished.connect(worker.deleteLater)
                self._list_workers.append(worker)
                worker.start()

    def _on_s3_children(self, prefix: str, entries: list, parent_item: QStandardItem) -> None:
        self._s3_status.setText("")
        for entry in entries:
            self._append_entry(parent_item, entry)

    def _on_s3_selection_changed(self, current: QModelIndex, _) -> None:
        source_idx = self._s3_proxy.mapToSource(current)
        item = self._s3_model.itemFromIndex(source_idx)
        if item is None:
            self._load_btn.setEnabled(False)
            return

        prefix = item.data(Qt.ItemDataRole.UserRole)
        if not prefix or prefix == "__placeholder__":
            self._selected_s3_prefix = None
            self._load_btn.setEnabled(False)
            return

        # Only enable "Load" if this looks like a session folder
        name = prefix.rstrip("/").split("/")[-1]
        is_session = name.startswith("session_")
        self._selected_s3_prefix = prefix if is_session else None
        self._selected_local_path = None

        if is_session:
            self._path_label.setText(
                f"s3://{self._s3_config.bucket}/{prefix}"
            )
            self._path_label.setStyleSheet("color: #89dceb; font-size: 11px;")
            self._load_btn.setEnabled(True)
        else:
            self._path_label.setText("Sélectionnez un dossier session_*")
            self._path_label.setStyleSheet("color: #6c7086; font-size: 11px; font-style: italic;")
            self._load_btn.setEnabled(False)

    def _on_s3_list_error(self, prefix: str, error: str) -> None:
        self._s3_status.setText(f"Erreur : {error[:120]}")
        self._s3_refresh_btn.setEnabled(True)
        logger.error("S3 browse error at %s: %s", prefix, error)

    # ------------------------------------------------------------------
    # Load action
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        if self._selected_local_path:
            self.session_selected.emit(self._selected_local_path)
            self.accept()
        elif self._selected_s3_prefix:
            self._open_s3_session(self._selected_s3_prefix)
        elif self._selected_hdd_path:
            self._open_hdd_session(self._selected_hdd_path)

    def _open_hdd_session(self, remote_path: str) -> None:
        self._load_btn.setEnabled(False)
        self._hdd_refresh_btn.setEnabled(False)
        session_name = remote_path.rstrip("/").split("/")[-1]

        self._progress_dlg = QProgressDialog(
            f"Téléchargement de {session_name} depuis HDD…",
            "Annuler", 0, 100, self,
        )
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumWidth(480)
        self._progress_dlg.setValue(0)
        self._progress_dlg.setStyleSheet("""
            QProgressDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; }
            QPushButton {
                background: #45475a; color: #cdd6f4;
                border: none; border-radius: 4px; padding: 6px 16px;
            }
            QProgressBar {
                background: #313244; border: none; border-radius: 4px;
                text-align: center; color: #cdd6f4;
            }
            QProgressBar::chunk { background: #a6e3a1; border-radius: 4px; }
        """)
        self._progress_dlg.show()

        self._file_bytes: dict = {}
        self._download_worker = HDDDownloadSessionWorker(
            self._hdd_config, remote_path, parent=self
        )
        self._download_worker.finished.connect(self._on_hdd_download_finished)
        self._download_worker.error_occurred.connect(self._on_hdd_download_error)
        self._download_worker.progress_update.connect(self._on_s3_progress)   # reuse same handler
        self._download_worker.file_done.connect(self._on_hdd_file_done)
        self._progress_dlg.canceled.connect(self._download_worker.cancel)
        self._download_worker.start()

    def _on_hdd_file_done(self, filename: str, count: int) -> None:
        self._hdd_status.setText(f"Téléchargé : {filename} ({count} fichier(s))")

    def _on_hdd_download_finished(self, local_path: str) -> None:
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.close()
        self._hdd_status.setText("")
        self.session_selected.emit(local_path)
        self.accept()

    def _on_hdd_download_error(self, error: str) -> None:
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.close()
        self._hdd_status.setText("")
        self._load_btn.setEnabled(True)
        self._hdd_refresh_btn.setEnabled(True)
        QMessageBox.critical(
            self, "Erreur HDD",
            f"Impossible de télécharger la session depuis le HDD :\n{error}",
        )

    def _open_s3_session(self, prefix: str) -> None:
        self._load_btn.setEnabled(False)
        self._s3_refresh_btn.setEnabled(False)
        session_name = prefix.rstrip("/").split("/")[-1]

        self._progress_dlg = QProgressDialog(
            f"Téléchargement de {session_name}…",
            "Annuler",
            0, 100,
            self,
        )
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumWidth(480)
        self._progress_dlg.setValue(0)
        self._progress_dlg.setStyleSheet("""
            QProgressDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; }
            QPushButton {
                background: #45475a; color: #cdd6f4;
                border: none; border-radius: 4px; padding: 6px 16px;
            }
            QProgressBar {
                background: #313244; border: none; border-radius: 4px;
                text-align: center; color: #cdd6f4;
            }
            QProgressBar::chunk { background: #89b4fa; border-radius: 4px; }
        """)
        self._progress_dlg.show()

        self._done_files = 0
        self._file_bytes: dict = {}   # filename -> (done, total)

        self._download_worker = S3DownloadSessionWorker(
            self._s3_client, prefix, parent=self
        )
        self._download_worker.finished.connect(self._on_s3_download_finished)
        self._download_worker.error_occurred.connect(self._on_s3_download_error)
        self._download_worker.progress_update.connect(self._on_s3_progress)
        self._download_worker.file_done.connect(self._on_s3_file_done)
        self._progress_dlg.canceled.connect(self._download_worker.cancel)
        self._download_worker.start()

    def _on_s3_progress(self, filename: str, done: int, total: int) -> None:
        self._file_bytes[filename] = (done, total)
        total_done = sum(v[0] for v in self._file_bytes.values())
        total_all = sum(v[1] for v in self._file_bytes.values())
        pct = int(total_done * 100 / total_all) if total_all > 0 else 0
        self._progress_dlg.setValue(pct)
        mb_done = total_done / 1e6
        mb_total = total_all / 1e6
        self._progress_dlg.setLabelText(
            f"Téléchargement en cours…\n{filename}\n{mb_done:.1f} / {mb_total:.1f} Mo"
        )

    def _on_s3_file_done(self, filename: str, count: int) -> None:
        self._s3_status.setText(f"Téléchargé : {filename} ({count} fichier(s))")

    def _on_s3_download_finished(self, local_path: str) -> None:
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.close()
        self._s3_status.setText("")
        self.session_selected.emit(local_path)
        self.accept()

    def _on_s3_download_error(self, error: str) -> None:
        if hasattr(self, "_progress_dlg"):
            self._progress_dlg.close()
        self._s3_status.setText("")
        self._load_btn.setEnabled(True)
        self._s3_refresh_btn.setEnabled(True)
        QMessageBox.critical(
            self,
            "Erreur S3",
            f"Impossible de télécharger la session S3 :\n{error}",
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        for w in self._list_workers:
            if w.isRunning():
                w.quit()
                w.wait(1000)
        if self._download_worker and self._download_worker.isRunning():
            self._download_worker.cancel()
            self._download_worker.wait(2000)
        super().closeEvent(event)
