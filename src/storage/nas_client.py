

import logging
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import paramiko

from ..queue.rabbitmq_consumer import AnnotationJob

logger = logging.getLogger(__name__)


def upload_directory_sftp_background(
    local_dir: Path,
    nas_dest: str,           # absolute POSIX path on the NAS, e.g. /DB-EXORIA/lakehouse/silver/...
    host: str,
    port: int = 22,
    username: str = "admin",
    password: str = "",
    key_path: Optional[str] = None,
    delete_after: bool = True,
    delete_session_dir: Optional[Path] = None,
) -> subprocess.Popen:
    """Upload *local_dir* to the NAS via SFTP in an independent subprocess.

    The caller does not wait for the upload to finish.  The subprocess
    deletes ``local_dir`` once the upload is done (if ``delete_after`` is
    True).  If ``delete_session_dir`` is given and the upload is error-free,
    that directory is also removed.

    Returns:
        The ``subprocess.Popen`` of the background process.
    """
    script = r"""
import sys, logging, shutil
from pathlib import Path, PurePosixPath
import paramiko

logging.basicConfig(
    level=logging.INFO,
    format="[sftp-upload] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("sftp_upload")

local_dir       = Path(sys.argv[1])
nas_dest        = sys.argv[2]
host            = sys.argv[3]
port            = int(sys.argv[4])
username        = sys.argv[5]
password        = sys.argv[6] if sys.argv[6] != "__none__" else None
key_path        = sys.argv[7] if sys.argv[7] != "__none__" else None
delete_after    = sys.argv[8] == "1"
session_dir_raw = sys.argv[9] if len(sys.argv) > 9 else "__none__"
delete_session  = session_dir_raw != "__none__"
session_dir     = Path(session_dir_raw) if delete_session else None

logger.info("local_dir  = %s (exists=%s)", local_dir, local_dir.exists())
logger.info("nas_dest   = %s", nas_dest)
logger.info("host       = %s:%s", host, port)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
connect_kwargs = dict(
    hostname=host, port=port, username=username, timeout=30,
    look_for_keys=False, allow_agent=False,
)
if key_path:
    connect_kwargs["key_filename"] = key_path
elif password:
    connect_kwargs["password"] = password
ssh.connect(**connect_kwargs)
sftp = ssh.open_sftp()

def sftp_makedirs(path_str):
    # Crée l'arborescence via SFTP niveau par niveau (respecte les ACL Synology DSM)
    from pathlib import PurePosixPath as PPP
    current = ""
    for part in PPP(path_str).parts:
        current = str(PPP(current) / part) if current else part
        if current in ("/", ""):
            continue
        try:
            sftp.stat(current)
        except IOError:
            try:
                sftp.mkdir(current)
            except IOError:
                pass  # existe déjà (race) ou parent manquant — géré au niveau suivant

def ssh_exec(cmd):
    try:
        _, stdout, _ = ssh.exec_command(cmd)
        return stdout.channel.recv_exit_status()
    except Exception:
        return -1

def makedirs(path_str):
    # 1. SFTP mkdir niveau par niveau (ACL DSM)
    sftp_makedirs(path_str)
    # Vérifier si le répertoire existe maintenant
    try:
        sftp.stat(path_str)
        return True
    except IOError:
        pass
    # 2. SSH exec mkdir -p
    if ssh_exec(f"mkdir -p '{path_str}'") == 0:
        try:
            sftp.stat(path_str)
            return True
        except IOError:
            pass
    # 3. SSH exec sudo mkdir -p (Synology admin sans mot de passe)
    if ssh_exec(f"sudo mkdir -p '{path_str}' && sudo chown {username} '{path_str}'") == 0:
        try:
            sftp.stat(path_str)
            return True
        except IOError:
            pass
    return False

def ssh_rm(path_str):
    try:
        sftp.remove(path_str)
        return
    except IOError:
        pass
    ssh_exec(f"rm -f '{path_str}'")
    ssh_exec(f"sudo rm -f '{path_str}'")

files = sorted(f for f in local_dir.rglob("*") if f.is_file())
total = len(files)
logger.info("Fichiers à uploader : %d", total)
uploaded = 0
errors = 0

# --- Vérification précoce des permissions d'écriture ---
if not makedirs(nas_dest):
    logger.error(
        "PERMISSION REFUSÉE : impossible de créer le dossier distant '%s'.\n"
        "Vérifiez que l'utilisateur '%s' a les droits d'écriture sur ce répertoire "
        "dans le panneau d'administration du NAS (DSM > Dossiers partagés > Permissions).",
        nas_dest, username,
    )
    sftp.close()
    ssh.close()
    sys.exit(1)

# Créer tous les sous-dossiers nécessaires en amont
seen_dirs = {nas_dest}
for file_path in files:
    relative_posix = PurePosixPath(*file_path.relative_to(local_dir).parts)
    remote_dir = str((PurePosixPath(nas_dest) / relative_posix).parent)
    if remote_dir not in seen_dirs:
        makedirs(remote_dir)
        seen_dirs.add(remote_dir)

for file_path in files:
    relative_posix = PurePosixPath(*file_path.relative_to(local_dir).parts)
    remote_path_str = str(PurePosixPath(nas_dest) / relative_posix)
    try:
        # Supprimer l'éventuel fichier existant (peut appartenir à un autre utilisateur)
        ssh_rm(remote_path_str)
        sftp.put(str(file_path), remote_path_str, confirm=False)
        uploaded += 1
        logger.info("[%d/%d] %s", uploaded, total, relative_posix)
    except Exception as exc:
        errors += 1
        logger.error("FAILED %s: %s", relative_posix, exc)

sftp.close()
ssh.close()
logger.info("Upload terminé : %d/%d fichiers, %d erreurs", uploaded, total, errors)

if delete_after:
    try:
        shutil.rmtree(local_dir, ignore_errors=True)
        logger.info("Répertoire d'export supprimé : %s", local_dir)
    except Exception as exc:
        logger.warning("Impossible de supprimer %s : %s", local_dir, exc)

if delete_session and errors == 0 and session_dir is not None:
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.info("Cache de session supprimé : %s", session_dir)
    except Exception as exc:
        logger.warning("Impossible de supprimer le cache de session %s : %s", session_dir, exc)

sys.exit(0 if errors == 0 else 1)
"""

    cmd = [
        sys.executable, "-c", script,
        str(local_dir),
        nas_dest,
        host,
        str(port),
        username,
        password if password else "__none__",
        key_path if key_path else "__none__",
        "1" if delete_after else "0",
        str(delete_session_dir) if delete_session_dir else "__none__",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    logger.info(
        "SFTP upload background lancé (PID %d) : %s -> %s",
        proc.pid, local_dir, nas_dest,
    )
    return proc


def silver_dest_path(session_id: str, nas_silver_root: str = "/DB-EXORIA/lakehouse/silver/annotated") -> str:
    """Compute the NAS silver destination path for a given session_id.

    Pattern: <nas_silver_root>/<YYYY>/<MM>/<DD>/<session_id>

    Example:
        session_id = "session_20260308_161838"
        → /DB-EXORIA/lakehouse/silver/annotated/2026/03/08/session_20260308_161838
    """
    # Extract date from session_id: session_YYYYMMDD_HHMMSS
    try:
        date_part = session_id.split("_")[1]   # "20260308"
        yyyy, mm, dd = date_part[:4], date_part[4:6], date_part[6:8]
    except (IndexError, ValueError):
        yyyy, mm, dd = "unknown", "unknown", "unknown"

    return f"{nas_silver_root.rstrip('/')}/{yyyy}/{mm}/{dd}/{session_id}"


@dataclass
class LocalJobFiles:
    """Local paths to the downloaded job files."""

    cam_head: Path
    cam_left: Path
    cam_right: Path
    cam_head_jsonl: Path
    cam_left_jsonl: Path
    cam_right_jsonl: Path
    metadata: Optional[Path]
    gripper_left: Path
    gripper_right: Path
    tracker: Path

    @property
    def video_paths(self) -> dict[str, Path]:
        return {"head": self.cam_head, "left": self.cam_left, "right": self.cam_right}

    @property
    def jsonl_paths(self) -> dict[str, Path]:
        return {
            "head": self.cam_head_jsonl,
            "left": self.cam_left_jsonl,
            "right": self.cam_right_jsonl,
        }

    @property
    def sensor_paths(self) -> dict[str, Path]:
        return {
            "gripper_left": self.gripper_left,
            "gripper_right": self.gripper_right,
            "tracker": self.tracker,
        }


class NASClient:
    """Download files from the NAS (SFTP) to a local working directory.

    Each public method is thread-safe when called from separate NASClient
    instances (one per thread).  The shared ``local_dir`` is written to by
    multiple workers concurrently — this is safe because each worker writes
    to a distinct file path.
    """

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: str = "admin",
        password: Optional[str] = None,
        key_path: Optional[str] = None,
        local_dir: Optional[Path] = None,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_path = key_path

        if local_dir is not None:
            self.local_dir = local_dir
            self.local_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.local_dir = Path(tempfile.mkdtemp(prefix="vive_labeler_nas_"))

        self._ssh: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open SSH + SFTP connection to the NAS."""
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = dict(
            hostname=self.host,
            port=self.port,
            username=self.username,
            timeout=15,
        )
        if self.key_path:
            connect_kwargs["key_filename"] = self.key_path
        elif self.password:
            connect_kwargs["password"] = self.password

        self._ssh.connect(**connect_kwargs)
        self._sftp = self._ssh.open_sftp()
        logger.info("Connected to NAS at %s:%d", self.host, self.port)

    def disconnect(self) -> None:
        """Close SFTP and SSH connections."""
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
    # Single-file download
    # ------------------------------------------------------------------

    def download_file(
        self,
        remote_path: str,
        local_filename: str,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> Path:
        """Download one file from the NAS via SFTP.

        Args:
            remote_path: Absolute path on the NAS.
            local_filename: Relative path under self.local_dir.
            progress_cb: Optional callable(label, bytes_done, bytes_total).

        Returns:
            Path to the downloaded local file.
        """
        self._ensure_connected()

        local_path = self.local_dir / local_filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        label = Path(local_filename).name

        # Get remote file size for progress reporting
        try:
            stat = self._sftp.stat(remote_path)
            total = stat.st_size or 0
        except Exception:
            total = 0

        if progress_cb is not None:
            progress_cb(label, 0, total)

        logger.info("Downloading NAS %s -> %s", remote_path, local_path)

        if progress_cb is not None and total > 0:
            # Stream with progress via get() callback
            def _cb(done: int, _total: int) -> None:
                progress_cb(label, done, total)

            self._sftp.get(remote_path, str(local_path), callback=_cb)
        else:
            self._sftp.get(remote_path, str(local_path))

        size_mb = local_path.stat().st_size / 1e6
        logger.info("Downloaded %s (%.2f MB)", label, size_mb)

        if progress_cb is not None:
            progress_cb(label, local_path.stat().st_size, total or local_path.stat().st_size)

        return local_path

    # ------------------------------------------------------------------
    # Parallel job download
    # ------------------------------------------------------------------

    def download_job(
        self,
        job: AnnotationJob,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> "LocalJobFiles":
        """Alias for download_job_parallel (used by ScenarioPrefetcher)."""
        return self.download_job_parallel(job, progress_cb=progress_cb)

    def download_job_parallel(
        self,
        job: AnnotationJob,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        max_workers: int = 7,
    ) -> LocalJobFiles:
        """Download all files of a job concurrently, one thread per file.

        Args:
            job: The annotation job with NAS paths.
            progress_cb: Optional callable(label, bytes_done, bytes_total),
                         called from worker threads — must be thread-safe.
            max_workers: Max parallel SFTP connections.

        Returns:
            A LocalJobFiles with local paths to all downloaded files.
        """
        # Build the list of (remote_path, local_filename, key) to download.
        # key is the stable identifier used to retrieve results.
        tasks = [
            (job.metadata,        "metadata.json",           "metadata"),
            (job.tracker,         "tracker_positions.csv",   "tracker"),
            (job.gripper_left,    "gripper_left_data.csv",   "gripper_left"),
            (job.gripper_right,   "gripper_right_data.csv",  "gripper_right"),
            (job.cam_head,        "videos/head.mp4",         "cam_head"),
            (job.cam_left,        "videos/left.mp4",         "cam_left"),
            (job.cam_right,       "videos/right.mp4",        "cam_right"),
            (job.cam_head_jsonl,  "videos/head.jsonl",       "cam_head_jsonl"),
            (job.cam_left_jsonl,  "videos/left.jsonl",       "cam_left_jsonl"),
            (job.cam_right_jsonl, "videos/right.jsonl",      "cam_right_jsonl"),
        ]

        # Filter out empty paths (file absent from nas_paths)
        tasks = [(r, l, k) for r, l, k in tasks if r]

        if not tasks:
            raise FileNotFoundError(
                "Aucun fichier référencé dans ce job — nas_paths est vide."
            )

        results: dict[str, Path] = {}
        errors: list[Exception] = []

        def _download_one(remote: str, local_name: str, key: str) -> tuple[str, Path]:
            # Each thread gets its own NASClient so SFTP channels don't collide.
            worker = NASClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                key_path=self.key_path,
                local_dir=self.local_dir,
            )
            try:
                path = worker.download_file(remote, local_name, progress_cb)
                return key, path
            finally:
                worker.disconnect()

        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
            futures = {
                pool.submit(_download_one, remote, local_name, key): key
                for remote, local_name, key in tasks
            }
            for future in as_completed(futures):
                try:
                    key, path = future.result()
                    results[key] = path
                except Exception as exc:
                    logger.error(
                        "NAS download error for %s: %s", futures[future], exc
                    )
                    errors.append(exc)

        if errors:
            raise errors[0]

        missing = [k for k in ("tracker",) if k not in results]
        if missing:
            raise FileNotFoundError(
                f"Fichiers obligatoires manquants après téléchargement : {missing}"
            )

        return LocalJobFiles(
            cam_head=results.get("cam_head", Path("videos/head.mp4")),
            cam_left=results.get("cam_left", Path("videos/left.mp4")),
            cam_right=results.get("cam_right", Path("videos/right.mp4")),
            cam_head_jsonl=results.get("cam_head_jsonl", Path("videos/head.jsonl")),
            cam_left_jsonl=results.get("cam_left_jsonl", Path("videos/left.jsonl")),
            cam_right_jsonl=results.get("cam_right_jsonl", Path("videos/right.jsonl")),
            metadata=results.get("metadata"),
            gripper_left=results.get("gripper_left", Path("gripper_left_data.csv")),
            gripper_right=results.get("gripper_right", Path("gripper_right_data.csv")),
            tracker=results["tracker"],
        )
