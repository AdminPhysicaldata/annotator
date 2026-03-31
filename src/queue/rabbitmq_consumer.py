"""RabbitMQ consumer for receiving annotation jobs.

Connects to a RabbitMQ queue and retrieves jobs containing HDFS paths
for camera videos, metadata, and sensor CSV files.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError, ChannelClosedByBroker

from PyQt6.QtCore import QThread, pyqtSignal


@dataclass
class AnnotationJob:
    """Represents a single annotation job from the ingestion queue.

    Camera fields use position names: cam_head, cam_left, cam_right.
    Gripper fields: gripper_left, gripper_right.
    All paths are absolute NAS paths (e.g. /DB-EXORIA/lakehouse/...).
    """
    session_id: str
    cam_head: str
    cam_left: str
    cam_right: str
    cam_head_jsonl: str
    cam_left_jsonl: str
    cam_right_jsonl: str
    metadata: str
    gripper_left: str
    gripper_right: str
    tracker: str
    zone: str = "bronze/landing"
    manifest_remote: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationJob":
        """Parse a job from the ingestion_queue message body.

        Expected format:
        {
            "session_id":  "session_20260308_161838",
            "zone":        "bronze/landing",
            "nas_paths":   {
                "metadata.json":            "/DB-EXORIA/.../metadata.json",
                "tracker_positions.csv":    "/DB-EXORIA/.../tracker_positions.csv",
                "gripper_left_data.csv":    "/DB-EXORIA/.../gripper_left_data.csv",
                "gripper_right_data.csv":   "/DB-EXORIA/.../gripper_right_data.csv",
                "videos/head.mp4":          "/DB-EXORIA/.../videos/head.mp4",
                "videos/left.mp4":          "/DB-EXORIA/.../videos/left.mp4",
                "videos/right.mp4":         "/DB-EXORIA/.../videos/right.mp4",
                "videos/head.jsonl":        "/DB-EXORIA/.../videos/head.jsonl",
                "videos/left.jsonl":        "/DB-EXORIA/.../videos/left.jsonl",
                "videos/right.jsonl":       "/DB-EXORIA/.../videos/right.jsonl"
            },
            "manifest_remote": "/DB-EXORIA/.../_manifest.json",
            "created_at":      "2026-03-08T16:18:58Z"
        }
        """
        nas = data.get("nas_paths", {})

        # Dérive session_id depuis nas_paths si absent au top level
        session_id = data.get("session_id") or ""
        if not session_id:
            for path in nas.values():
                if path:
                    for part in path.replace("\\", "/").split("/"):
                        if part.startswith("session_"):
                            session_id = part
                            break
                if session_id:
                    break

        return cls(
            session_id=session_id,
            cam_head=nas.get("videos/head.mp4", ""),
            cam_left=nas.get("videos/left.mp4", ""),
            cam_right=nas.get("videos/right.mp4", ""),
            cam_head_jsonl=nas.get("videos/head.jsonl", ""),
            cam_left_jsonl=nas.get("videos/left.jsonl", ""),
            cam_right_jsonl=nas.get("videos/right.jsonl", ""),
            metadata=nas.get("metadata.json", ""),
            gripper_left=nas.get("gripper_left_data.csv", ""),
            gripper_right=nas.get("gripper_right_data.csv", ""),
            tracker=nas.get("tracker_positions.csv", ""),
            zone=data.get("zone", "bronze/landing"),
            manifest_remote=data.get("manifest_remote", ""),
            created_at=data.get("created_at", ""),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "session_id":      self.session_id,
            "zone":            self.zone,
            "nas_paths": {
                "metadata.json":          self.metadata,
                "tracker_positions.csv":  self.tracker,
                "gripper_left_data.csv":  self.gripper_left,
                "gripper_right_data.csv": self.gripper_right,
                "videos/head.mp4":        self.cam_head,
                "videos/left.mp4":        self.cam_left,
                "videos/right.mp4":       self.cam_right,
                "videos/head.jsonl":      self.cam_head_jsonl,
                "videos/left.jsonl":      self.cam_left_jsonl,
                "videos/right.jsonl":     self.cam_right_jsonl,
            },
            "manifest_remote": self.manifest_remote,
            "created_at":      self.created_at,
        }


class RabbitMQConsumer:
    """Consumes annotation jobs from a RabbitMQ queue.

    All network I/O is synchronous — call methods from a worker thread,
    never from the Qt main thread.
    """

    def __init__(
        self,
        host: str = "192.168.88.246",
        port: int = 5672,
        username: str = "admin",
        password: str = "Admin123456!",
        queue_name: str = "ingestion_queue",  # matches config default
        virtual_host: str = "/",
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.queue_name = queue_name
        self.virtual_host = virtual_host

        self._connection: Optional[pika.BlockingConnection] = None
        self._channel = None

    def _get_connection_params(self) -> pika.ConnectionParameters:
        credentials = pika.PlainCredentials(self.username, self.password)
        return pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=credentials,
            connection_attempts=3,
            retry_delay=2,
            socket_timeout=5,
        )

    def connect(self) -> None:
        """Establish connection to RabbitMQ.

        Raises on connection failure (caller's responsibility to handle).
        """
        params = self._get_connection_params()
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.basic_qos(prefetch_count=1)

        # passive=True: don't create/modify the queue, just check it exists.
        # If the queue doesn't exist pika closes the channel — we reopen it.
        try:
            self._channel.queue_declare(queue=self.queue_name, passive=True)
        except (ChannelClosedByBroker, AMQPChannelError) as exc:
            logger.warning(
                "Queue '%s' passive declare failed: %s — reopening channel and continuing",
                self.queue_name, exc,
            )
            try:
                self._channel = self._connection.channel()
                self._channel.basic_qos(prefetch_count=1)
            except Exception as reopen_exc:
                logger.error("Cannot reopen RabbitMQ channel: %s", reopen_exc)
                raise
        except Exception as exc:
            logger.warning("Queue declare unexpected error: %s — continuing", exc)

    def disconnect(self) -> None:
        """Close the RabbitMQ connection."""
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception as exc:
            logger.debug("RabbitMQ disconnect error (ignored): %s", exc)
        self._connection = None
        self._channel = None

    def fetch_one_job(self) -> Optional[AnnotationJob]:
        """Fetch a single job from the queue (non-blocking on the AMQP side).

        Returns:
            An AnnotationJob if one is available, None otherwise.

        Raises:
            ValueError: if the message is malformed (bad JSON or missing keys).
            Other AMQP exceptions propagate to the caller.
        """
        if self._channel is None:
            self.connect()

        try:
            method_frame, header_frame, body = self._channel.basic_get(
                queue=self.queue_name, auto_ack=False
            )
        except Exception as exc:
            logger.error("basic_get failed: %s", exc)
            # Invalidate channel so next call reconnects
            self._channel = None
            raise

        if method_frame is None:
            return None

        raw = body.decode("utf-8", errors="replace")
        logger.info("RabbitMQ message received (%d bytes)", len(raw))

        # --- Parse JSON ---
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Cannot decode JSON: %s — nacking message", e)
            self._safe_nack(method_frame.delivery_tag, requeue=False)
            raise ValueError(f"Invalid JSON: {e}") from e

        # --- Build job ---
        try:
            job = AnnotationJob.from_dict(data)
        except KeyError as e:
            logger.error(
                "Missing key %s in message. Available keys: %s",
                e, list(data.keys()) if isinstance(data, dict) else type(data),
            )
            # Requeue so we don't lose the message while debugging
            self._safe_nack(method_frame.delivery_tag, requeue=True)
            raise ValueError(f"Missing key in job message: {e}") from e

        self._safe_ack(method_frame.delivery_tag)
        return job

    # ------------------------------------------------------------------
    # Internal helpers for safe ack/nack
    # ------------------------------------------------------------------

    def _safe_ack(self, delivery_tag: int) -> None:
        try:
            self._channel.basic_ack(delivery_tag=delivery_tag)
        except Exception as exc:
            logger.error("basic_ack failed (delivery_tag=%s): %s — message may be redelivered", delivery_tag, exc)

    def _safe_nack(self, delivery_tag: int, requeue: bool) -> None:
        try:
            self._channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
        except Exception as exc:
            logger.error("basic_nack failed (delivery_tag=%s, requeue=%s): %s", delivery_tag, requeue, exc)


class RabbitMQPollerThread(QThread):
    """Worker thread that polls RabbitMQ at regular intervals.

    Signals:
        job_received(AnnotationJob): emitted when a job is fetched.
        poll_status(str, str): emitted after each poll with (status, detail).
        error_occurred(str): emitted on connection/parse errors.
    """

    job_received = pyqtSignal(object)      # AnnotationJob
    poll_status = pyqtSignal(str, str)     # (status_text, detail_text)
    error_occurred = pyqtSignal(str)       # error message

    def __init__(
        self,
        consumer: RabbitMQConsumer,
        poll_interval_s: float = 5.0,
        parent=None,
    ):
        super().__init__(parent)
        self._consumer = consumer
        self._poll_interval_s = poll_interval_s
        self._running = False
        self._poll_count = 0

    def run(self):
        """Thread main loop — polls RabbitMQ until stopped or a job arrives."""
        self._running = True
        self._poll_count = 0

        while self._running:
            self._poll_count += 1
            self.poll_status.emit(
                "Recherche de jobs dans la queue...",
                f"Queue: {self._consumer.queue_name}  |  "
                f"Tentative #{self._poll_count}",
            )

            try:
                job = self._consumer.fetch_one_job()

                if job is not None:
                    self.poll_status.emit("Job recu !", "")
                    self.job_received.emit(job)
                    self._running = False
                    break

                self.poll_status.emit(
                    "Aucun job disponible — nouvelle tentative dans "
                    f"{self._poll_interval_s:.0f}s",
                    f"Queue: {self._consumer.queue_name}  |  "
                    f"Tentative #{self._poll_count}",
                )

            except Exception as e:
                logger.error("Poller error: %s", e, exc_info=True)
                self.error_occurred.emit(str(e)[:200])
                # Force reconnect on next attempt
                self._consumer.disconnect()

            # Sleep in small increments so we can stop quickly
            for _ in range(int(self._poll_interval_s * 10)):
                if not self._running:
                    break
                self.msleep(100)

        try:
            self._consumer.disconnect()
        except Exception as exc:
            logger.debug("Poller thread disconnect error (ignored): %s", exc)

    def stop(self):
        """Request the thread to stop gracefully."""
        self._running = False


@dataclass
class PrefetchedScenario:
    """A scenario that has been fully downloaded and is ready to annotate."""
    job: AnnotationJob
    local_files: object  # LocalJobFiles — imported at runtime to avoid circular dep
    session_dir: Path


class ScenarioPrefetcher(QThread):
    """Downloads the next scenario in the background while the user annotates.

    Lifecycle
    ---------
    1. Started as soon as the current scenario is loaded into the workspace.
    2. Polls RabbitMQ for the next job, downloads it from HDFS.
    3. Stores the result in an internal slot (one-scenario buffer).
    4. When the annotator finishes, ``consume()`` hands the ready scenario
       instantly.  If it's not ready yet the caller can wait on
       ``ready_event``.

    Safety guarantees
    -----------------
    - The downloaded local files are **never deleted** until the caller
      explicitly calls ``consume()`` and takes ownership.
    - If the prefetcher is stopped before the scenario is consumed, the
      ``discard()`` method cleans up the temp dir so nothing leaks on disk.
    - Only one scenario is prefetched at a time (no over-fetching).
    """

    prefetch_status = pyqtSignal(str)   # human-readable status update
    prefetch_ready = pyqtSignal()       # emitted when the scenario is ready
    prefetch_error = pyqtSignal(str)    # emitted on unrecoverable error

    def __init__(
        self,
        consumer: "RabbitMQConsumer",
        hdfs_client,                    # NASClient — passed from MainWindow
        poll_interval_s: float = 5.0,
        parent=None,
    ):
        super().__init__(parent)
        self._consumer = consumer
        self._hdfs = hdfs_client
        self._poll_interval_s = poll_interval_s

        self._running = False
        self._scenario: Optional[PrefetchedScenario] = None
        self._lock = threading.Lock()
        self.ready_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API (called from the main thread)
    # ------------------------------------------------------------------

    def consume(self) -> Optional[PrefetchedScenario]:
        """Return the prefetched scenario and clear the internal slot."""
        with self._lock:
            scenario = self._scenario
            self._scenario = None
            self.ready_event.clear()
            return scenario

    def discard(self) -> None:
        """Clean up a prefetched scenario that will never be annotated."""
        import shutil
        with self._lock:
            scenario = self._scenario
            self._scenario = None
            self.ready_event.clear()

        if scenario is not None:
            try:
                if scenario.session_dir.exists():
                    shutil.rmtree(scenario.session_dir, ignore_errors=True)
                    logger.info("Prefetcher discarded local files: %s", scenario.session_dir)
            except Exception as exc:
                logger.warning("Could not clean up prefetched scenario: %s", exc)

    # ------------------------------------------------------------------
    # Thread main loop
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        poll_count = 0

        # --- Step 1: poll RabbitMQ until a job arrives ---
        job: Optional[AnnotationJob] = None
        while self._running and job is None:
            poll_count += 1
            self.prefetch_status.emit(
                f"Prefetch: recherche du prochain scénario... (tentative #{poll_count})"
            )
            try:
                job = self._consumer.fetch_one_job()
                if job is None:
                    for _ in range(int(self._poll_interval_s * 10)):
                        if not self._running:
                            break
                        self.msleep(100)
            except Exception as exc:
                logger.warning("Prefetch poll error: %s", exc)
                try:
                    self._consumer.disconnect()
                except Exception:
                    pass
                for _ in range(int(self._poll_interval_s * 10)):
                    if not self._running:
                        break
                    self.msleep(100)

        if not self._running or job is None:
            try:
                self._consumer.disconnect()
            except Exception:
                pass
            return

        self.prefetch_status.emit("Prefetch: job trouvé — téléchargement en arrière-plan...")

        # --- Step 2: download from HDFS ---
        local_files = None
        try:
            local_files = self._hdfs.download_job(job)
        except Exception as exc:
            logger.error("Prefetch HDFS download failed: %s", exc, exc_info=True)
            self.prefetch_error.emit(str(exc))
            try:
                self._consumer.disconnect()
            except Exception:
                pass
            return

        if not self._running:
            # Stopped during download — clean up immediately
            import shutil
            try:
                if local_files is not None:
                    session_dir = local_files.tracker.parent
                    if session_dir.exists():
                        shutil.rmtree(session_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning("Prefetch cleanup after stop failed: %s", exc)
            try:
                self._consumer.disconnect()
            except Exception:
                pass
            return

        # --- Step 3: store in buffer and signal readiness ---
        try:
            scenario = PrefetchedScenario(
                job=job,
                local_files=local_files,
                session_dir=local_files.tracker.parent,
            )
            with self._lock:
                self._scenario = scenario
                self.ready_event.set()

            self.prefetch_status.emit("Prefetch: prochain scénario prêt")
            self.prefetch_ready.emit()
        except Exception as exc:
            logger.error("Prefetch scenario store failed: %s", exc)
            self.prefetch_error.emit(str(exc))

        try:
            self._consumer.disconnect()
        except Exception:
            pass

    def stop(self):
        """Request the thread to stop gracefully."""
        self._running = False
