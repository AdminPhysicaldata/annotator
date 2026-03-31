"""HDFS client for downloading job files to local storage.

Uses the hdfs library (WebHDFS / HttpFS) to fetch files referenced
in AnnotationJob messages coming from RabbitMQ.
"""

import logging
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from hdfs import InsecureClient

from ..queue.rabbitmq_consumer import AnnotationJob

logger = logging.getLogger(__name__)


def upload_directory_background(
    local_dir: Path,
    hdfs_dest: str,
    hdfs_url: str,
    hdfs_user: Optional[str] = None,
    delete_after: bool = True,
    delete_session_dir: Optional[Path] = None,
) -> subprocess.Popen:
    """Lance l'upload d'un répertoire vers HDFS dans un sous-process indépendant.

    L'appelant n'attend pas la fin de l'upload — le sous-process tourne en
    arrière-plan et supprime ``local_dir`` une fois l'upload terminé
    (si ``delete_after`` est True).  Si ``delete_session_dir`` est fourni,
    ce répertoire (le cache de session brut) est également supprimé après
    un upload réussi.

    Returns:
        Le ``subprocess.Popen`` du processus lancé (on ne l'attend pas).
    """
    # Script Python inline exécuté par le sous-process.
    # Utilise uniquement la stdlib + hdfs pour ne pas dépendre du package
    # applicatif complet.
    script = r"""
import sys, logging, shutil
from pathlib import Path
from urllib.parse import urlparse
import requests
from hdfs import InsecureClient

logging.basicConfig(
    level=logging.INFO,
    format="[hdfs-upload] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("hdfs_upload")

local_dir         = Path(sys.argv[1])
hdfs_dest         = sys.argv[2]
hdfs_url          = sys.argv[3]
hdfs_user         = sys.argv[4] if sys.argv[4] != "__none__" else None
delete_after      = sys.argv[5] == "1"
session_dir_raw   = sys.argv[6] if len(sys.argv) > 6 else "__none__"
delete_session    = session_dir_raw != "__none__"
session_dir       = Path(session_dir_raw) if delete_session else None

# Rebuild the DataNode-redirect session
parsed = urlparse(hdfs_url)
namenode_host = parsed.hostname
session = requests.Session()

def _rewrite_redirect(resp, **kwargs):
    if resp.is_redirect or resp.is_permanent_redirect:
        location = resp.headers.get("Location", "")
        if location:
            from urllib.parse import urlparse as _up
            p = _up(location)
            if p.hostname and p.hostname != namenode_host:
                fixed = p._replace(
                    netloc=f"{namenode_host}:{p.port}" if p.port else namenode_host
                )
                resp.headers["Location"] = fixed.geturl()

session.hooks["response"].append(_rewrite_redirect)
client = InsecureClient(hdfs_url, user=hdfs_user, session=session)

files = sorted(local_dir.rglob("*"))
total = sum(1 for f in files if f.is_file())
uploaded = 0
errors = 0

for file_path in files:
    if not file_path.is_file():
        continue
    relative = file_path.relative_to(local_dir)
    hdfs_path = f"{hdfs_dest}/{relative}"
    parent = str(Path(hdfs_path).parent)
    try:
        client.makedirs(parent)
        client.upload(hdfs_path, str(file_path), overwrite=True)
        uploaded += 1
        logger.info("[%d/%d] %s", uploaded, total, relative)
    except Exception as exc:
        errors += 1
        logger.error("FAILED %s: %s", relative, exc)

logger.info("Upload terminé : %d/%d fichiers, %d erreurs", uploaded, total, errors)

if delete_after:
    try:
        shutil.rmtree(local_dir, ignore_errors=True)
        logger.info("Répertoire d'export supprimé : %s", local_dir)
    except Exception as exc:
        logger.warning("Impossible de supprimer %s : %s", local_dir, exc)

# Supprimer le cache de session brut uniquement si l'upload est sans erreur
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
        hdfs_dest,
        hdfs_url,
        hdfs_user if hdfs_user else "__none__",
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
        "Upload background lancé (PID %d) : %s -> %s",
        proc.pid, local_dir, hdfs_dest,
    )
    return proc


# LocalJobFiles is now defined in nas_client to avoid circular imports.
# Re-exported here for backwards compatibility with existing imports.
from .nas_client import LocalJobFiles  # noqa: E402


class HDFSClient:
    """Download files from HDFS to a local working directory."""

    def __init__(
        self,
        hdfs_url: str = "http://192.168.30.10:9870",
        hdfs_user: Optional[str] = None,
        local_dir: Optional[Path] = None,
    ):
        """
        Args:
            hdfs_url: WebHDFS base URL (e.g. http://host:9870).
            hdfs_user: HDFS user for authentication (InsecureClient).
            local_dir: Root directory where files are downloaded.
                       Defaults to a temporary directory.
        """
        self.hdfs_url = hdfs_url
        self.hdfs_user = hdfs_user
        if local_dir is not None:
            self.local_dir = local_dir
            self.local_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.local_dir = Path(tempfile.mkdtemp(prefix="vive_labeler_"))
        self._client: Optional[InsecureClient] = None

    def _make_datanode_session(self):
        """Create a requests.Session that rewrites DataNode redirect URLs.

        HDFS DataNodes inside Docker advertise their container hostname
        (e.g. '144777b5fcf8') which is not resolvable from outside the
        Docker network.  This hook replaces the DataNode hostname in
        redirect URLs with the NameNode host so the request reaches the
        correct machine.
        """
        import requests

        parsed = urlparse(self.hdfs_url)
        namenode_host = parsed.hostname  # e.g. "192.168.30.10"

        session = requests.Session()

        def _rewrite_redirect(resp, **kwargs):
            if resp.is_redirect or resp.is_permanent_redirect:
                location = resp.headers.get("Location", "")
                if location:
                    p = urlparse(location)
                    # Replace unresolvable Docker hostname with namenode host
                    if p.hostname and p.hostname != namenode_host:
                        fixed = p._replace(netloc=f"{namenode_host}:{p.port}" if p.port else namenode_host)
                        resp.headers["Location"] = fixed.geturl()
                        logger.debug("Rewrote DataNode redirect: %s -> %s", location, resp.headers["Location"])

        session.hooks["response"].append(_rewrite_redirect)
        return session

    def connect(self) -> None:
        """Create the WebHDFS client."""
        session = self._make_datanode_session()
        self._client = InsecureClient(self.hdfs_url, user=self.hdfs_user, session=session)
        logger.info("Connected to HDFS at %s", self.hdfs_url)

    def disconnect(self) -> None:
        """Release the client (no persistent connection to close)."""
        self._client = None

    @staticmethod
    def _hdfs_path(uri: str) -> str:
        """Extract the HDFS path from a full hdfs:// URI.

        Example:
            "hdfs://namenode:8020/data/cam1.mp4" -> "/data/cam1.mp4"
            "/data/cam1.mp4" -> "/data/cam1.mp4"
        """
        parsed = urlparse(uri)
        if parsed.scheme == "hdfs":
            return parsed.path
        return uri

    def _resolve_path(self, remote_path: str) -> str:
        """If *remote_path* is a directory, return the first file inside it.

        The Operator stores camera recordings as ``cam0/output.avi`` so the
        job message may reference the directory (``cam0``) rather than the
        file itself.  This helper transparently resolves that.
        """
        status = self._client.status(remote_path, strict=False)
        if status is None:
            raise FileNotFoundError(f"File does not exist: {remote_path}")
        if status["type"] == "DIRECTORY":
            entries = self._client.list(remote_path)
            if not entries:
                raise FileNotFoundError(f"Directory is empty: {remote_path}")
            # Pick the first (usually only) file
            return f"{remote_path}/{entries[0]}"
        return remote_path

    def list_directory(self, hdfs_path: str) -> list[dict]:
        """List entries in an HDFS directory.

        Returns a list of dicts with keys:
            name (str)  — entry name only (not full path)
            type (str)  — "DIRECTORY" or "FILE"
            size (int)  — length in bytes (0 for directories)

        Sorted: directories first, then alphabetically.
        """
        if self._client is None:
            self.connect()
        entries = self._client.list(hdfs_path, status=True)
        result = [
            {"name": name, "type": status["type"], "size": status.get("length", 0)}
            for name, status in entries
        ]
        return sorted(result, key=lambda e: (e["type"] != "DIRECTORY", e["name"].lower()))

    def download_file(self, hdfs_uri: str, local_filename: Optional[str] = None) -> Path:
        """Download a single file from HDFS.

        Args:
            hdfs_uri: HDFS URI or path of the remote file.
            local_filename: Override local filename. If None, keeps the
                            original filename from the HDFS path.

        Returns:
            Path to the downloaded local file.
        """
        if self._client is None:
            self.connect()

        remote_path = self._resolve_path(self._hdfs_path(hdfs_uri))
        if local_filename is None:
            local_filename = Path(remote_path).name
        elif not Path(local_filename).suffix:
            # Preserve the remote file extension (e.g. "cam1" + ".avi" -> "cam1.avi")
            local_filename = local_filename + Path(remote_path).suffix

        local_path = self.local_dir / local_filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading %s -> %s", remote_path, local_path)
        self._client.download(remote_path, str(local_path), overwrite=True)
        logger.info("Downloaded %s (%.2f MB)", local_filename, local_path.stat().st_size / 1e6)

        return local_path

    def _stream_download(
        self,
        resolved_hdfs_path: str,
        local_filename: str,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        chunk_size: int = 1 << 17,  # 128 KB
    ) -> Path:
        """Stream-download an already-resolved HDFS path with progress reporting.

        Unlike ``download_file_with_progress`` this method does **not** call
        ``_resolve_path`` — the caller must pass the exact file path on HDFS.
        This is used by ``download_job_parallel`` where path resolution is done
        once in the calling thread before spawning workers.
        """
        import requests

        local_path = self.local_dir / local_filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        namenode_url = self.hdfs_url.rstrip("/")
        open_url = f"{namenode_url}/webhdfs/v1{resolved_hdfs_path}?op=OPEN"

        parsed = urlparse(self.hdfs_url)
        namenode_host = parsed.hostname

        session = requests.Session()

        def _rewrite_redirect(resp, **kwargs):
            if resp.is_redirect or resp.is_permanent_redirect:
                location = resp.headers.get("Location", "")
                if location:
                    p = urlparse(location)
                    if p.hostname and p.hostname != namenode_host:
                        fixed = p._replace(
                            netloc=f"{namenode_host}:{p.port}" if p.port else namenode_host
                        )
                        resp.headers["Location"] = fixed.geturl()

        session.hooks["response"].append(_rewrite_redirect)

        params = {}
        if self.hdfs_user:
            params["user.name"] = self.hdfs_user

        logger.info("Streaming %s -> %s", resolved_hdfs_path, local_path)
        resp = session.get(open_url, params=params, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        label = Path(local_filename).name

        with open(local_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    done += len(chunk)
                    if progress_cb is not None:
                        progress_cb(label, done, total)

        logger.info("Done %s (%.2f MB)", local_filename, local_path.stat().st_size / 1e6)
        return local_path

    def download_file_with_progress(
        self,
        hdfs_uri: str,
        local_filename: str,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        chunk_size: int = 1 << 17,  # 128 KB
    ) -> Path:
        """Download a single file from HDFS with byte-level progress reporting.

        Uses the WebHDFS streaming HTTP endpoint directly so we can read
        the Content-Length header and report bytes-transferred incrementally.

        Args:
            hdfs_uri: HDFS URI or path of the remote file.
            local_filename: Local filename (relative to self.local_dir).
            progress_cb: Optional callable(label, bytes_done, bytes_total)
                         called periodically during the download.
            chunk_size: HTTP streaming chunk size in bytes.

        Returns:
            Path to the downloaded local file.
        """
        import requests

        if self._client is None:
            self.connect()

        remote_path = self._resolve_path(self._hdfs_path(hdfs_uri))
        if not Path(local_filename).suffix:
            local_filename = local_filename + Path(remote_path).suffix

        local_path = self.local_dir / local_filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Build WebHDFS open URL (OPEN operation triggers a redirect to DataNode)
        namenode_url = self.hdfs_url.rstrip("/")
        open_url = f"{namenode_url}/webhdfs/v1{remote_path}?op=OPEN&noredirect=false"

        parsed = urlparse(self.hdfs_url)
        namenode_host = parsed.hostname

        session = requests.Session()

        def _rewrite_redirect(resp, **kwargs):
            if resp.is_redirect or resp.is_permanent_redirect:
                location = resp.headers.get("Location", "")
                if location:
                    p = urlparse(location)
                    if p.hostname and p.hostname != namenode_host:
                        fixed = p._replace(
                            netloc=f"{namenode_host}:{p.port}" if p.port else namenode_host
                        )
                        resp.headers["Location"] = fixed.geturl()

        session.hooks["response"].append(_rewrite_redirect)

        params = {}
        if self.hdfs_user:
            params["user.name"] = self.hdfs_user

        logger.info("Streaming download %s -> %s", remote_path, local_path)
        resp = session.get(open_url, params=params, stream=True, timeout=30)
        resp.raise_for_status()

        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        label = Path(local_filename).name

        with open(local_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    done += len(chunk)
                    if progress_cb is not None:
                        progress_cb(label, done, total)

        logger.info("Downloaded %s (%.2f MB)", local_filename, local_path.stat().st_size / 1e6)
        return local_path

    def download_job_parallel(
        self,
        job: "AnnotationJob",
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        max_workers: int = 7,
    ) -> "LocalJobFiles":
        """Download all files of a job concurrently, one thread per file.

        Args:
            job: The annotation job containing HDFS paths.
            progress_cb: Optional callable(label, bytes_done, bytes_total)
                         called from worker threads during each file download.
                         Must be thread-safe (e.g. emit a Qt signal via
                         a QueuedConnection).
            max_workers: Maximum number of parallel download threads.

        Returns:
            A LocalJobFiles instance with paths to the local copies.
        """
        if self._client is None:
            self.connect()

        # Resolve all HDFS paths up-front in the calling thread using the
        # already-connected client.  This handles directory → file resolution
        # before spawning worker threads, avoiding races and ensuring each
        # worker gets the exact file path.
        raw_files = [
            (job.metadata,      "metadata.json",            "metadata.json"),
            (job.tracker,       "tracker_positions.csv",    "tracker_positions.csv"),
            (job.gripper_left,  "gripper_left_data.csv",    "gripper_left_data.csv"),
            (job.gripper_right, "gripper_right_data.csv",   "gripper_right_data.csv"),
            (job.cam_head,      "videos/head",              "caméra head"),
            (job.cam_left,      "videos/left",              "caméra left"),
            (job.cam_right,     "videos/right",             "caméra right"),
        ]

        # (resolved_hdfs_path, local_name_with_ext, key, label)
        # key is the original local_name before extension is appended, used as
        # a stable dict key in results regardless of the resolved file extension.
        files_to_download: list[tuple[str, str, str, str]] = []
        for uri, local_name, label in raw_files:
            try:
                resolved = self._resolve_path(self._hdfs_path(uri))
            except FileNotFoundError:
                logger.warning("Fichier absent sur HDFS (ignoré) : %s", uri)
                continue
            key = local_name
            if not Path(local_name).suffix:
                local_name = local_name + Path(resolved).suffix
            files_to_download.append((resolved, local_name, key, label))
            logger.debug("Resolved %s -> %s (local: %s)", uri, resolved, local_name)

        results: dict[str, Path] = {}
        errors: list[Exception] = []

        def _download_one(resolved_path: str, local_name: str, key: str, label: str) -> tuple[str, Path]:
            # Each thread gets its own HDFSClient + requests session so HTTP
            # connections don't collide across threads.
            worker_client = HDFSClient(
                hdfs_url=self.hdfs_url,
                hdfs_user=self.hdfs_user,
                local_dir=self.local_dir,
            )

            def _cb(lbl: str, done: int, total: int) -> None:
                if progress_cb is not None:
                    progress_cb(label, done, total)

            # Emit immediately so the progress bar row appears at once.
            if progress_cb is not None:
                progress_cb(label, 0, 0)

            path = worker_client._stream_download(resolved_path, local_name, _cb)
            return key, path

        if not files_to_download:
            raise FileNotFoundError(
                "Aucun fichier résolu pour ce job — tous les chemins HDFS sont introuvables."
            )

        with ThreadPoolExecutor(max_workers=min(max_workers, len(files_to_download))) as pool:
            futures = {
                pool.submit(_download_one, resolved_path, local_name, key, label): key
                for resolved_path, local_name, key, label in files_to_download
            }
            for future in as_completed(futures):
                try:
                    key, path = future.result()
                    results[key] = path
                except Exception as exc:
                    logger.error("Parallel download error for %s: %s", futures[future], exc)
                    errors.append(exc)

        if errors:
            raise errors[0]

        # Determine actual video extensions (mp4 or avi)
        def _video_path(key: str) -> Path:
            for ext in (".mp4", ".avi", ""):
                candidate = results.get(key + ext) or results.get(key)
                if candidate is not None:
                    return candidate
            # Fallback: search results for key prefix
            for k, v in results.items():
                if k.startswith(key):
                    return v
            return results.get(key, Path(key))

        return LocalJobFiles(
            cam_head=_video_path("videos/head"),
            cam_left=_video_path("videos/left"),
            cam_right=_video_path("videos/right"),
            metadata=results.get("metadata.json"),
            gripper_left=results.get("gripper_left_data.csv", Path("gripper_left_data.csv")),
            gripper_right=results.get("gripper_right_data.csv", Path("gripper_right_data.csv")),
            tracker=results["tracker_positions.csv"],
        )

    def upload_file(self, local_path: Path, hdfs_path: str) -> None:
        """Upload a single file to HDFS.

        Args:
            local_path: Path to the local file.
            hdfs_path: Destination path on HDFS.
        """
        if self._client is None:
            self.connect()

        # Ensure parent directory exists on HDFS
        parent = str(Path(hdfs_path).parent)
        self._client.makedirs(parent)

        logger.info("Uploading %s -> %s", local_path, hdfs_path)
        self._client.upload(hdfs_path, str(local_path), overwrite=True)
        logger.info("Uploaded %s (%.2f MB)", local_path.name, local_path.stat().st_size / 1e6)

    def move_to_trash(self, hdfs_source_dir: str, trash_root: str = "/trash") -> str:
        """Déplace un répertoire HDFS vers le dossier trash.

        Args:
            hdfs_source_dir: Chemin HDFS du dossier source à rejeter.
            trash_root: Dossier de destination sur HDFS (défaut : /trash).

        Returns:
            Le chemin HDFS de destination dans le trash.
        """
        if self._client is None:
            self.connect()

        folder_name = Path(hdfs_source_dir).name
        dest = f"{trash_root}/{folder_name}"

        self._client.makedirs(trash_root)
        self._client.rename(hdfs_source_dir, dest)
        logger.info("Moved to trash: %s -> %s", hdfs_source_dir, dest)
        return dest

    def upload_directory(self, local_dir: Path, hdfs_base_path: str) -> None:
        """Upload all files in a local directory to HDFS.

        Args:
            local_dir: Local directory to upload.
            hdfs_base_path: Destination base path on HDFS.
        """
        if self._client is None:
            self.connect()

        for file_path in sorted(local_dir.rglob("*")):
            if file_path.is_file():
                relative = file_path.relative_to(local_dir)
                hdfs_dest = f"{hdfs_base_path}/{relative}"
                self.upload_file(file_path, hdfs_dest)

    def download_job(self, job: AnnotationJob, progress_cb=None) -> LocalJobFiles:
        """Download all files referenced by an AnnotationJob.

        Args:
            job: The annotation job containing HDFS paths.
            progress_cb: Optional callable(step: int, total: int, label: str)
                         called before each file download starts.

        Returns:
            A LocalJobFiles instance with paths to the local copies.
        """
        if self._client is None:
            self.connect()

        files_to_download = [
            (job.metadata,      "metadata.json",            "metadata.json"),
            (job.tracker,       "tracker_positions.csv",    "tracker_positions.csv"),
            (job.gripper_left,  "gripper_left_data.csv",    "gripper_left_data.csv"),
            (job.gripper_right, "gripper_right_data.csv",   "gripper_right_data.csv"),
            (job.cam_head,      "videos/head",              "caméra head (vidéo)"),
            (job.cam_left,      "videos/left",              "caméra left (vidéo)"),
            (job.cam_right,     "videos/right",             "caméra right (vidéo)"),
        ]
        total = len(files_to_download)
        results: dict[str, Path] = {}

        for step, (uri, local_name, label) in enumerate(files_to_download, start=1):
            if progress_cb is not None:
                progress_cb(step, total, label)
            try:
                key = local_name
                if not Path(local_name).suffix:
                    resolved = self._resolve_path(self._hdfs_path(uri))
                    local_name = local_name + Path(resolved).suffix
                results[key] = self.download_file(uri, local_name)
            except FileNotFoundError:
                logger.warning("Fichier absent sur HDFS (ignoré) : %s", uri)

        def _video_path(key: str) -> Path:
            if key in results:
                return results[key]
            for k, v in results.items():
                if k.startswith(key):
                    return v
            return Path(key)

        return LocalJobFiles(
            cam_head=_video_path("videos/head"),
            cam_left=_video_path("videos/left"),
            cam_right=_video_path("videos/right"),
            metadata=results.get("metadata.json"),
            gripper_left=results.get("gripper_left_data.csv", Path("gripper_left_data.csv")),
            gripper_right=results.get("gripper_right_data.csv", Path("gripper_right_data.csv")),
            tracker=results["tracker_positions.csv"],
        )
