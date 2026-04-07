"""FFmpeg manager — locate or download ffmpeg automatically.

Priority:
  1. <app_root>/assets/ffmpeg  (or ffmpeg.exe on Windows)
  2. ffmpeg found in PATH via shutil.which
  3. Download a static build from the internet into <app_root>/assets/
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download URLs for static ffmpeg builds
# ---------------------------------------------------------------------------
# macOS (arm64 & x86_64): evermeet.cx static builds
# Windows x64: gyan.dev static build
# Linux x64: johnvansickle.com static build
_DOWNLOAD_URLS: dict[tuple[str, str], str] = {
    ("Darwin", "arm64"): "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip",
    ("Darwin", "x86_64"): "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip",
    ("Windows", "AMD64"): (
        "https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip"
    ),
    ("Linux", "x86_64"): (
        "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    ),
}

_system = platform.system()
_machine = platform.machine()

# Path to the bundled ffmpeg binary inside the assets folder
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
_BUNDLED_FFMPEG = _ASSETS_DIR / ("ffmpeg.exe" if _system == "Windows" else "ffmpeg")

# Cached result
_ffmpeg_path: str | None = None


def get_ffmpeg_path() -> str:
    """Return the path to the ffmpeg binary (cached after first call).

    Raises RuntimeError if ffmpeg cannot be found or downloaded.
    """
    global _ffmpeg_path
    if _ffmpeg_path is not None:
        return _ffmpeg_path

    # 1. Bundled binary
    if _BUNDLED_FFMPEG.exists() and _is_executable(_BUNDLED_FFMPEG):
        logger.info("FFmpeg found (bundled): %s", _BUNDLED_FFMPEG)
        _ffmpeg_path = str(_BUNDLED_FFMPEG)
        return _ffmpeg_path

    # 2. System PATH
    found = shutil.which("ffmpeg")
    if found:
        logger.info("FFmpeg found (system PATH): %s", found)
        _ffmpeg_path = found
        return _ffmpeg_path

    # 3. Download
    logger.info("FFmpeg not found — downloading…")
    _download_ffmpeg()

    if _BUNDLED_FFMPEG.exists() and _is_executable(_BUNDLED_FFMPEG):
        logger.info("FFmpeg downloaded successfully: %s", _BUNDLED_FFMPEG)
        _ffmpeg_path = str(_BUNDLED_FFMPEG)
        return _ffmpeg_path

    raise RuntimeError(
        "FFmpeg introuvable et impossible à télécharger.\n"
        "Installez ffmpeg manuellement et réessayez."
    )


def is_ffmpeg_available() -> bool:
    """Return True if ffmpeg is already available without triggering a download."""
    if _BUNDLED_FFMPEG.exists() and _is_executable(_BUNDLED_FFMPEG):
        return True
    return shutil.which("ffmpeg") is not None


def _is_executable(path: Path) -> bool:
    return os.access(path, os.X_OK)


def _download_ffmpeg(progress_callback=None) -> None:
    """Download a static ffmpeg binary and place it in assets/.

    Args:
        progress_callback: optional callable(bytes_downloaded, total_bytes)
    """
    key = (_system, _machine)
    url = _DOWNLOAD_URLS.get(key)
    if url is None:
        raise RuntimeError(
            f"Pas d'URL de téléchargement pour la plateforme : {_system}/{_machine}.\n"
            "Installez ffmpeg manuellement."
        )

    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = ".zip" if url.endswith(".zip") else ".tar.xz"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        logger.info("Téléchargement de ffmpeg depuis %s …", url)
        _download_with_progress(url, tmp_path, progress_callback)

        if suffix == ".zip":
            _extract_ffmpeg_from_zip(tmp_path)
        else:
            _extract_ffmpeg_from_tar(tmp_path)

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _download_with_progress(url: str, dest: Path, callback) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 65536
        with open(dest, "wb") as f:
            while True:
                data = response.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if callback:
                    callback(downloaded, total)


def _extract_ffmpeg_from_zip(archive: Path) -> None:
    """Extract the ffmpeg binary from a zip archive."""
    with zipfile.ZipFile(archive, "r") as zf:
        # Find the ffmpeg binary inside the archive
        candidates = [
            name for name in zf.namelist()
            if _is_ffmpeg_binary_name(name)
        ]
        if not candidates:
            raise RuntimeError("Impossible de trouver le binaire ffmpeg dans l'archive ZIP.")
        # Pick the shortest path (most likely the root binary)
        candidates.sort(key=lambda n: len(n))
        binary_name = candidates[0]
        with zf.open(binary_name) as src, open(_BUNDLED_FFMPEG, "wb") as dst:
            dst.write(src.read())
    _make_executable(_BUNDLED_FFMPEG)


def _extract_ffmpeg_from_tar(archive: Path) -> None:
    """Extract the ffmpeg binary from a tar.xz archive."""
    with tarfile.open(archive, "r:xz") as tf:
        candidates = [
            m for m in tf.getmembers()
            if _is_ffmpeg_binary_name(m.name) and m.isfile()
        ]
        if not candidates:
            raise RuntimeError("Impossible de trouver le binaire ffmpeg dans l'archive TAR.")
        candidates.sort(key=lambda m: len(m.name))
        member = candidates[0]
        src = tf.extractfile(member)
        if src is None:
            raise RuntimeError("Impossible d'extraire ffmpeg de l'archive TAR.")
        with open(_BUNDLED_FFMPEG, "wb") as dst:
            dst.write(src.read())
    _make_executable(_BUNDLED_FFMPEG)


def _is_ffmpeg_binary_name(name: str) -> bool:
    base = Path(name).name.lower()
    return base in ("ffmpeg", "ffmpeg.exe")


def _make_executable(path: Path) -> None:
    current = stat.S_IMODE(os.stat(path).st_mode)
    os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def download_ffmpeg_with_dialog(parent_widget=None) -> bool:
    """Show a progress dialog while downloading ffmpeg.

    Returns True if ffmpeg is available after the call, False otherwise.
    """
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QLabel, QProgressBar, QApplication,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal

    if is_ffmpeg_available():
        return True

    class _DownloadThread(QThread):
        progress = pyqtSignal(int, int)   # downloaded, total
        finished_ok = pyqtSignal()
        finished_err = pyqtSignal(str)

        def run(self):
            try:
                _download_ffmpeg(progress_callback=lambda d, t: self.progress.emit(d, t))
                self.finished_ok.emit()
            except Exception as exc:
                self.finished_err.emit(str(exc))

    dlg = QDialog(parent_widget)
    dlg.setWindowTitle("Téléchargement de FFmpeg")
    dlg.setFixedSize(420, 120)
    dlg.setWindowFlags(
        dlg.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
    )
    layout = QVBoxLayout(dlg)

    lbl = QLabel("Téléchargement de FFmpeg en cours…\nCela peut prendre quelques secondes.")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(lbl)

    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    layout.addWidget(bar)

    success = [False]
    error_msg = [""]

    thread = _DownloadThread()

    def on_progress(downloaded, total):
        if total > 0:
            bar.setValue(int(downloaded * 100 / total))
        else:
            bar.setRange(0, 0)  # indeterminate

    def on_ok():
        success[0] = True
        dlg.accept()

    def on_err(msg):
        error_msg[0] = msg
        dlg.reject()

    thread.progress.connect(on_progress)
    thread.finished_ok.connect(on_ok)
    thread.finished_err.connect(on_err)
    thread.start()

    dlg.exec()

    if not success[0]:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(
            parent_widget,
            "Erreur de téléchargement",
            f"Impossible de télécharger FFmpeg :\n{error_msg[0]}\n\n"
            "Installez ffmpeg manuellement et relancez l'application.",
        )
        return False

    # Refresh cache
    global _ffmpeg_path
    _ffmpeg_path = None
    try:
        get_ffmpeg_path()
        return True
    except RuntimeError:
        return False
