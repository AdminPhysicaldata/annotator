"""SPOOL SFTP client — récupère les scénarios depuis le serveur SPOOL.

Structure attendue sur le SPOOL :
  <inbox_base>/<session_id>/
      metadata.json
      tracker_positions.csv
      gripper_left_data.csv
      gripper_right_data.csv
      videos/
          head.mp4
          left.mp4
          right.mp4
          head.jsonl
          left.jsonl
          right.jsonl

Le SpoolClient liste les dossiers disponibles dans inbox_base, télécharge
un scénario choisi localement, et construit un AnnotationJob compatible avec
le reste du pipeline.
"""

import logging
import stat as stat_mod
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import paramiko
from PyQt6.QtCore import QThread, pyqtSignal

from ..queue.rabbitmq_consumer import AnnotationJob
from ..storage.nas_client import LocalJobFiles, upload_directory_sftp_background

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level SFTP helpers
# ---------------------------------------------------------------------------

class SpoolClient:
    """Client SFTP pour le serveur SPOOL."""

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: str = "spool",
        password: Optional[str] = None,
        inbox_base: str = "/srv/exoria/",
        local_dir: Optional[Path] = None,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.inbox_base = inbox_base.rstrip("/")

        if local_dir is not None:
            self.local_dir = local_dir
            self.local_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.local_dir = Path(tempfile.mkdtemp(prefix="vive_spool_"))

        self._ssh: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kw = dict(
            hostname=self.host, port=self.port, username=self.username, timeout=15,
            look_for_keys=False, allow_agent=False,  # skip pubkey probes → connexion 4× plus rapide
        )
        if self.password:
            kw["password"] = self.password
        self._ssh.connect(**kw)
        self._sftp = self._ssh.open_sftp()
        logger.info("SpoolClient connected to %s:%d", self.host, self.port)

    def disconnect(self) -> None:
        try:
            if self._sftp:
                self._sftp.close()
        except Exception:
            pass
        try:
            if self._ssh:
                self._ssh.close()
        except Exception:
            pass
        self._sftp = None
        self._ssh = None

    def _ensure_connected(self) -> None:
        if self._sftp is None:
            self.connect()

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    # Fichiers obligatoires pour qu'une session soit considérée complète
    _REQUIRED_FILES = {"tracker_positions.csv", "metadata.json", "videos"}

    def list_scenarios(self) -> List[str]:
        """Retourne la liste des session_id disponibles dans inbox_base.

        Chaque session_id correspond à un sous-dossier direct de inbox_base
        qui contient tous les fichiers requis (tracker, metadata, videos/).
        Les entrées sont triées par ordre alphabétique (chronologique).
        """
        self._ensure_connected()
        try:
            entries = self._sftp.listdir_attr(self.inbox_base)
        except FileNotFoundError:
            logger.warning("SPOOL inbox_base not found: %s", self.inbox_base)
            return []

        all_dirs = sorted(
            e.filename
            for e in entries
            if stat_mod.S_ISDIR(e.st_mode)
        )
        logger.info("SPOOL: %d dossier(s) trouvés dans %s", len(all_dirs), self.inbox_base)
        return all_dirs

    def find_latest_complete_session(self) -> Optional[str]:
        """Parcourt l'inbox de la fin vers le début et retourne la première session complète.

        Beaucoup plus rapide que de scanner tous les dossiers quand l'inbox
        est grande (ex: 28 000 entrées) : on s'arrête dès la première trouvée.
        """
        self._ensure_connected()
        try:
            entries = self._sftp.listdir_attr(self.inbox_base)
        except FileNotFoundError:
            logger.warning("inbox_base not found: %s", self.inbox_base)
            return None

        dirs = sorted(
            [e.filename for e in entries if stat_mod.S_ISDIR(e.st_mode)],
            reverse=True,   # plus récente en premier
        )
        logger.info("HDD: %d dossier(s), recherche de la dernière session complète…", len(dirs))

        for name in dirs:
            path = f"{self.inbox_base.rstrip('/')}/{name}"
            try:
                children = {e.filename for e in self._sftp.listdir_attr(path)}
                if self._REQUIRED_FILES.issubset(children):
                    logger.info("HDD: session complète trouvée : %s", name)
                    return name
            except Exception:
                pass

        logger.warning("HDD: aucune session complète trouvée dans %s", self.inbox_base)
        return None

    def list_dir(self, path: str) -> list:
        """Liste le contenu d'un dossier.

        Returns:
            Liste de dicts triés (dossiers d'abord, puis fichiers, alpha) :
              {"name": str, "is_dir": bool, "size": int, "is_scenario": bool}
        """
        self._ensure_connected()
        try:
            entries = self._sftp.listdir_attr(path)
        except Exception as exc:
            logger.warning("Cannot list_dir %s: %s", path, exc)
            return []

        items = []
        for e in entries:
            is_dir = stat_mod.S_ISDIR(e.st_mode)
            child_path = f"{path.rstrip('/')}/{e.filename}"
            is_scenario = False
            if is_dir:
                try:
                    child_names = {c.filename for c in self._sftp.listdir_attr(child_path)}
                    is_scenario = "metadata.json" in child_names or "videos" in child_names
                except Exception:
                    pass
            items.append({
                "name": e.filename,
                "is_dir": is_dir,
                "size": e.st_size or 0,
                "is_scenario": is_scenario,
            })

        # Dossiers d'abord, puis fichiers, tri alpha dans chaque groupe
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        return items

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_scenario(
        self,
        session_id: str,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        cancelled_flag: Optional[threading.Event] = None,
    ) -> "LocalJobFiles":
        """Télécharge un scénario complet depuis le SPOOL vers local_dir.

        Args:
            session_id: Nom du dossier dans inbox_base.
            progress_cb: Callable(label, bytes_done, bytes_total) optionnel.
            cancelled_flag: threading.Event — si set(), annule le téléchargement.

        Returns:
            LocalJobFiles pointant vers les fichiers locaux téléchargés.
        """
        self._ensure_connected()

        remote_root = f"{self.inbox_base}/{session_id}"
        local_root = self.local_dir / session_id

        # Supprimer le cache local existant pour forcer un téléchargement propre
        if local_root.exists():
            import shutil
            shutil.rmtree(local_root)
            logger.info("Cleared stale local cache: %s", local_root)

        local_root.mkdir(parents=True, exist_ok=True)

        self._download_recursive(remote_root, local_root, progress_cb, cancelled_flag)

        return self._build_local_job_files(local_root)

    def _download_recursive(
        self,
        remote_dir: str,
        local_dir: Path,
        progress_cb: Optional[Callable[[str, int, int], None]],
        cancelled_flag: Optional[threading.Event],
    ) -> None:
        try:
            entries = self._sftp.listdir_attr(remote_dir)
        except Exception as exc:
            logger.warning("Cannot list %s: %s", remote_dir, exc)
            return

        for attr in entries:
            if cancelled_flag and cancelled_flag.is_set():
                return
            remote_child = f"{remote_dir.rstrip('/')}/{attr.filename}"
            local_child = local_dir / attr.filename

            if stat_mod.S_ISDIR(attr.st_mode):
                local_child.mkdir(parents=True, exist_ok=True)
                self._download_recursive(remote_child, local_child, progress_cb, cancelled_flag)
            else:
                total = attr.st_size or 0
                label = attr.filename
                if progress_cb:
                    progress_cb(label, 0, total)

                def _cb(done: int, _total: int, _label=label, _total2=total) -> None:
                    if progress_cb:
                        progress_cb(_label, done, _total2)

                local_child.parent.mkdir(parents=True, exist_ok=True)
                self._sftp.get(remote_child, str(local_child), callback=_cb if total > 0 else None)
                if progress_cb:
                    size = local_child.stat().st_size
                    progress_cb(label, size, total or size)
                logger.debug("Downloaded %s (%.2f MB)", remote_child, (local_child.stat().st_size / 1e6))

    def _remove_remote_dir(self, remote_dir: str) -> None:
        """Supprime récursivement un dossier sur le serveur SFTP.

        Tente d'abord via SSH exec (rm -rf), plus fiable quand le serveur
        restreint les opérations SFTP rmdir. Fallback sur suppression SFTP
        fichier par fichier si SSH exec échoue.
        """
        # Tentative via SSH exec — beaucoup plus rapide et fiable
        if self._ssh is not None:
            try:
                # Échapper le chemin pour éviter les injections de shell
                safe_path = remote_dir.replace("'", "'\\''")
                _stdin, _stdout, _stderr = self._ssh.exec_command(
                    f"rm -rf '{safe_path}'", timeout=30
                )
                exit_code = _stdout.channel.recv_exit_status()
                if exit_code == 0:
                    logger.info("Deleted remote directory via SSH exec: %s", remote_dir)
                    return
                err = _stderr.read().decode(errors="replace").strip()
                logger.warning("SSH exec rm -rf failed (exit %d): %s", exit_code, err)
            except Exception as exc:
                logger.warning("SSH exec rm -rf error: %s — falling back to SFTP", exc)

        # Fallback : suppression SFTP récursive fichier par fichier
        try:
            entries = self._sftp.listdir_attr(remote_dir)
        except Exception as exc:
            logger.warning("Cannot list remote dir for deletion %s: %s", remote_dir, exc)
            return

        for attr in entries:
            child = f"{remote_dir.rstrip('/')}/{attr.filename}"
            if stat_mod.S_ISDIR(attr.st_mode):
                self._remove_remote_dir(child)
            else:
                try:
                    self._sftp.remove(child)
                    logger.debug("Deleted remote file %s", child)
                except Exception as exc:
                    logger.warning("Cannot delete remote file %s: %s", child, exc)

        try:
            self._sftp.rmdir(remote_dir)
            logger.info("Deleted remote directory %s", remote_dir)
        except Exception as exc:
            logger.warning("Cannot rmdir %s: %s", remote_dir, exc)

    @staticmethod
    def _build_local_job_files(local_root: Path) -> LocalJobFiles:
        """Construit un LocalJobFiles à partir d'un dossier de session local."""
        videos = local_root / "videos"

        def _p(rel: str) -> Path:
            return local_root / rel

        return LocalJobFiles(
            cam_head=videos / "head.mp4",
            cam_left=videos / "left.mp4",
            cam_right=videos / "right.mp4",
            cam_head_jsonl=videos / "head.jsonl",
            cam_left_jsonl=videos / "left.jsonl",
            cam_right_jsonl=videos / "right.jsonl",
            metadata=_p("metadata.json") if (_p("metadata.json")).exists() else None,
            gripper_left=_p("gripper_left_data.csv"),
            gripper_right=_p("gripper_right_data.csv"),
            tracker=_p("tracker_positions.csv"),
        )

    @staticmethod
    def build_annotation_job(session_id: str, local_root: Path) -> AnnotationJob:
        """Construit un AnnotationJob synthétique à partir d'un dossier local."""
        videos = local_root / "videos"

        def _s(p: Path) -> str:
            return str(p)

        return AnnotationJob(
            session_id=session_id,
            cam_head=_s(videos / "head.mp4"),
            cam_left=_s(videos / "left.mp4"),
            cam_right=_s(videos / "right.mp4"),
            cam_head_jsonl=_s(videos / "head.jsonl"),
            cam_left_jsonl=_s(videos / "left.jsonl"),
            cam_right_jsonl=_s(videos / "right.jsonl"),
            metadata=_s(local_root / "metadata.json"),
            gripper_left=_s(local_root / "gripper_left_data.csv"),
            gripper_right=_s(local_root / "gripper_right_data.csv"),
            tracker=_s(local_root / "tracker_positions.csv"),
            zone="spool/inbox",
        )


# ---------------------------------------------------------------------------
# Qt worker thread — liste les scénarios disponibles
# ---------------------------------------------------------------------------

class SpoolListWorker(QThread):
    """Liste les scénarios disponibles sur le SPOOL dans un thread de fond."""

    scenarios_ready = pyqtSignal(list)   # list[str] — session_ids
    error_occurred = pyqtSignal(str)     # message d'erreur

    def __init__(self, host: str, port: int, username: str, password: str,
                 inbox_base: str, parent=None):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._inbox_base = inbox_base

    def run(self) -> None:
        client = SpoolClient(
            host=self._host, port=self._port,
            username=self._username, password=self._password,
            inbox_base=self._inbox_base,
        )
        try:
            client.connect()
            scenarios = client.list_scenarios()
            client.disconnect()
            self.scenarios_ready.emit(scenarios)
        except Exception as exc:
            try:
                client.disconnect()
            except Exception:
                pass
            logger.error("SpoolListWorker error: %s", exc)
            self.error_occurred.emit(str(exc))


class SpoolBrowseWorker(QThread):
    """Liste le contenu d'un dossier SFTP dans un thread de fond."""

    listing_ready = pyqtSignal(str, list)  # (path, items)
    error_occurred = pyqtSignal(str)

    def __init__(self, host: str, port: int, username: str, password: str,
                 path: str, parent=None):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._path = path

    def run(self) -> None:
        client = SpoolClient(
            host=self._host, port=self._port,
            username=self._username, password=self._password,
            inbox_base=self._path,
        )
        try:
            client.connect()
            items = client.list_dir(self._path)
            client.disconnect()
            self.listing_ready.emit(self._path, items)
        except Exception as exc:
            try:
                client.disconnect()
            except Exception:
                pass
            logger.error("SpoolBrowseWorker error: %s", exc)
            self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# Qt worker thread — télécharge un scénario
# ---------------------------------------------------------------------------

class SpoolDownloadWorker(QThread):
    """Télécharge un scénario depuis le SPOOL dans un thread de fond."""

    download_finished = pyqtSignal(object)   # LocalJobFiles
    file_progress = pyqtSignal(str, int, int)  # (label, done, total)
    error_occurred = pyqtSignal(str)

    def __init__(self, host: str, port: int, username: str, password: str,
                 inbox_base: str, session_id: str,
                 local_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._inbox_base = inbox_base
        self._session_id = session_id
        self._local_dir = local_dir
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        client = SpoolClient(
            host=self._host, port=self._port,
            username=self._username, password=self._password,
            inbox_base=self._inbox_base,
            local_dir=self._local_dir,
        )

        def _progress(label: str, done: int, total: int) -> None:
            self.file_progress.emit(label, done, total)

        try:
            client.connect()
            local_files = client.download_scenario(
                self._session_id,
                progress_cb=_progress,
                cancelled_flag=self._cancel,
            )
            client.disconnect()

            if not self._cancel.is_set():
                self.download_finished.emit(local_files)
        except Exception as exc:
            try:
                client.disconnect()
            except Exception:
                pass
            if not self._cancel.is_set():
                logger.error("SpoolDownloadWorker error: %s", exc, exc_info=True)
                self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# HDD verification pipeline — list + download from HDD inbox
# ---------------------------------------------------------------------------

class HddVerificationWorker(QThread):
    """Récupère automatiquement la dernière session disponible depuis le HDD inbox.

    Flux:
      1. Se connecte au serveur HDD.
      2. Liste les dossiers dans inbox_base (tri alphabétique → la dernière = la plus récente).
      3. Télécharge ce dossier localement.
      4. Supprime le dossier sur le serveur HDD (retrait de l'inbox).
      5. Émet download_finished(session_id, LocalJobFiles).
    """

    download_finished = pyqtSignal(str, object)    # (session_id, LocalJobFiles)
    file_progress = pyqtSignal(str, int, int)       # (label, done, total)
    no_session_available = pyqtSignal()             # inbox vide
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        inbox_base: str,
        local_dir: Optional[Path] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._inbox_base = inbox_base
        self._local_dir = local_dir
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        client = SpoolClient(
            host=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
            inbox_base=self._inbox_base,
            local_dir=self._local_dir,
        )

        def _progress(label: str, done: int, total: int) -> None:
            self.file_progress.emit(label, done, total)

        try:
            client.connect()
            session_id = client.find_latest_complete_session()
            if session_id is None:
                client.disconnect()
                self.no_session_available.emit()
                return

            logger.info("HddVerificationWorker: downloading session '%s'", session_id)

            local_files = client.download_scenario(
                session_id,
                progress_cb=_progress,
                cancelled_flag=self._cancel,
            )

            if self._cancel.is_set():
                client.disconnect()
                return

            # Supprimer le dossier sur le HDD (retrait de l'inbox)
            remote_path = f"{self._inbox_base.rstrip('/')}/{session_id}"
            logger.info("HddVerificationWorker: removing remote session '%s'", remote_path)
            client._remove_remote_dir(remote_path)

            client.disconnect()
            self.download_finished.emit(session_id, local_files)

        except Exception as exc:
            try:
                client.disconnect()
            except Exception:
                pass
            if not self._cancel.is_set():
                logger.error("HddVerificationWorker error: %s", exc, exc_info=True)
                self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# HDD upload worker — envoie la session vers send_base ou retry_base
# ---------------------------------------------------------------------------

class HddUploadWorker(QThread):
    """Envoie un dossier local vers le serveur HDD (send ou retry) en arrière-plan.

    Attend la fin du subprocess SFTP et émet upload_finished(remote_dest) en cas
    de succès, ou error_occurred(message) en cas d'échec.
    """

    upload_finished = pyqtSignal(str)   # remote_dest
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        local_dir: Path,
        dest_base: str,          # send_base ou retry_base
        session_id: str,
        host: str,
        port: int,
        username: str,
        password: str,
        delete_after: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._local_dir = local_dir
        self._dest_base = dest_base
        self._session_id = session_id
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._delete_after = delete_after

    def run(self) -> None:
        remote_dest = f"{self._dest_base.rstrip('/')}/{self._session_id}"
        # Vérifier que le dossier local existe et n'est pas vide
        files = list(self._local_dir.rglob("*")) if self._local_dir.exists() else []
        files = [f for f in files if f.is_file()]
        if not files:
            msg = f"Dossier local vide ou introuvable : {self._local_dir}"
            logger.error("HddUploadWorker: %s", msg)
            self.error_occurred.emit(msg)
            return
        logger.info("HddUploadWorker: %d fichier(s) à uploader depuis %s", len(files), self._local_dir)
        try:
            proc = upload_directory_sftp_background(
                local_dir=self._local_dir,
                nas_dest=remote_dest,
                host=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                key_path=None,
                delete_after=self._delete_after,
            )
            logger.info(
                "HddUploadWorker: upload started PID %d → sftp://%s%s",
                proc.pid, self._host, remote_dest,
            )
            stdout, _ = proc.communicate()
            exit_code = proc.returncode
            if exit_code == 0:
                logger.info("HddUploadWorker: upload terminé avec succès → %s", remote_dest)
                self.upload_finished.emit(remote_dest)
            else:
                output = (stdout or "").strip()
                msg = f"Le subprocess d'upload a retourné le code {exit_code}.\n{output}"
                logger.error("HddUploadWorker: %s", msg)
                self.error_occurred.emit(msg)
        except Exception as exc:
            logger.error("HddUploadWorker error: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))
