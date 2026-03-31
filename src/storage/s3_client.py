"""S3 client for browsing and streaming session data from AWS S3.

- Browse: lists scenario folders under s3://physical-data-storage/bronze
- Stream: generates presigned URLs so videos/CSVs are read directly from S3
          without downloading to disk first.

Presigned URLs expire after PRESIGN_TTL_SEC seconds (default 15 min).
Videos are opened via cv2.VideoCapture(url) — requires OpenCV built with
ffmpeg/http support (standard pip wheel includes this).
CSVs are fetched via requests.get(url) in memory.
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

PRESIGN_TTL_SEC = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class S3SessionPaths:
    """S3 keys for every file in a session (relative to the bucket root)."""

    session_id: str
    bucket: str

    # Video keys
    cam_head: str
    cam_left: str
    cam_right: str

    # JSONL frame-timing keys
    cam_head_jsonl: str
    cam_left_jsonl: str
    cam_right_jsonl: str

    # Sensor / metadata keys
    metadata: str
    gripper_left: str
    gripper_right: str
    tracker: str

    def all_keys(self) -> dict[str, str]:
        return {
            "cam_head":       self.cam_head,
            "cam_left":       self.cam_left,
            "cam_right":      self.cam_right,
            "cam_head_jsonl": self.cam_head_jsonl,
            "cam_left_jsonl": self.cam_left_jsonl,
            "cam_right_jsonl": self.cam_right_jsonl,
            "metadata":       self.metadata,
            "gripper_left":   self.gripper_left,
            "gripper_right":  self.gripper_right,
            "tracker":        self.tracker,
        }


@dataclass
class S3SessionURLs:
    """Presigned HTTPS URLs for every file in a session.

    Pass these directly to cv2.VideoCapture / requests.get / json.loads —
    no local files needed.
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

    @property
    def video_urls(self) -> dict[str, str]:
        return {"head": self.cam_head, "left": self.cam_left, "right": self.cam_right}

    @property
    def jsonl_urls(self) -> dict[str, str]:
        return {
            "head": self.cam_head_jsonl,
            "left": self.cam_left_jsonl,
            "right": self.cam_right_jsonl,
        }

    @property
    def sensor_urls(self) -> dict[str, str]:
        return {
            "gripper_left":  self.gripper_left,
            "gripper_right": self.gripper_right,
            "tracker":       self.tracker,
        }


# ---------------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------------

class S3Client:
    """Browse and stream session data from AWS S3.

    Credentials are resolved by boto3 in the standard order:
      1. Explicit aws_access_key_id / aws_secret_access_key in S3Config
      2. Environment variables AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
      3. ~/.aws/credentials profile
      4. IAM instance role (EC2 / ECS)

    All methods are synchronous — call from a worker thread (QThread),
    never from the Qt main thread.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "eu-west-3",
        bronze_prefix: str = "bronze",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        presign_ttl: int = PRESIGN_TTL_SEC,
    ):
        self.bucket = bucket
        self.bronze_prefix = bronze_prefix.strip("/")
        self.presign_ttl = presign_ttl

        session = boto3.Session(
            region_name=region,
            aws_access_key_id=aws_access_key_id or None,
            aws_secret_access_key=aws_secret_access_key or None,
        )
        self._s3 = session.client("s3")

    # ------------------------------------------------------------------
    # Bucket browsing
    # ------------------------------------------------------------------

    def list_prefixes(self, prefix: str) -> list[dict]:
        """Return immediate children (folders) under *prefix*.

        Returns a list of dicts: {"name": str, "type": "DIRECTORY"|"FILE", "size": int}
        """
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        try:
            resp = self._s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                Delimiter="/",
            )
        except (ClientError, NoCredentialsError) as exc:
            logger.error("S3 list_prefixes error at %s: %s", prefix, exc)
            raise

        entries: list[dict] = []

        # Sub-folders (CommonPrefixes)
        for cp in resp.get("CommonPrefixes") or []:
            name = cp["Prefix"].rstrip("/").split("/")[-1]
            entries.append({"name": name, "type": "DIRECTORY", "size": 0, "full_prefix": cp["Prefix"]})

        # Files at this level
        for obj in resp.get("Contents") or []:
            key = obj["Key"]
            if key == prefix:
                continue  # skip the "folder" object itself
            name = key.split("/")[-1]
            entries.append({"name": name, "type": "FILE", "size": obj.get("Size", 0), "full_prefix": key})

        entries.sort(key=lambda e: (e["type"] != "DIRECTORY", e["name"].lower()))
        return entries

    def list_bronze_root(self) -> list[dict]:
        """List top-level entries under the bronze prefix."""
        return self.list_prefixes(self.bronze_prefix + "/")

    def list_sessions_under(self, prefix: str) -> list[dict]:
        """Recursively find all session_* folders under *prefix*.

        Returns entries sorted by name (newest last, since session IDs are
        date-stamped: session_YYYYMMDD_HHMMSS).
        """
        sessions: list[dict] = []
        self._collect_sessions(prefix, sessions)
        sessions.sort(key=lambda e: e["name"])
        return sessions

    def _collect_sessions(self, prefix: str, out: list[dict]) -> None:
        for entry in self.list_prefixes(prefix):
            if entry["type"] != "DIRECTORY":
                continue
            if entry["name"].startswith("session_"):
                out.append(entry)
            else:
                self._collect_sessions(entry["full_prefix"], out)

    # ------------------------------------------------------------------
    # Session discovery: find files inside a session folder
    # ------------------------------------------------------------------

    def resolve_session_paths(self, session_prefix: str) -> S3SessionPaths:
        """Build S3SessionPaths by listing all objects under *session_prefix*.

        Raises FileNotFoundError if mandatory files are missing.
        """
        if not session_prefix.endswith("/"):
            session_prefix += "/"

        session_id = session_prefix.rstrip("/").split("/")[-1]

        # Collect all keys in the session folder
        keys: dict[str, str] = {}  # filename -> full S3 key
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=session_prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                rel = key[len(session_prefix):]  # relative path inside session
                keys[rel] = key

        def _require(rel: str) -> str:
            if rel not in keys:
                raise FileNotFoundError(
                    f"Fichier manquant dans la session S3 : {session_prefix}{rel}"
                )
            return keys[rel]

        def _optional(rel: str) -> str:
            return keys.get(rel, "")

        return S3SessionPaths(
            session_id=session_id,
            bucket=self.bucket,
            cam_head=_require("videos/head.mp4"),
            cam_left=_require("videos/left.mp4"),
            cam_right=_require("videos/right.mp4"),
            cam_head_jsonl=_optional("videos/head.jsonl"),
            cam_left_jsonl=_optional("videos/left.jsonl"),
            cam_right_jsonl=_optional("videos/right.jsonl"),
            metadata=_optional("metadata.json"),
            gripper_left=_require("gripper_left_data.csv"),
            gripper_right=_require("gripper_right_data.csv"),
            tracker=_require("tracker_positions.csv"),
        )

    # ------------------------------------------------------------------
    # Presigned URL generation
    # ------------------------------------------------------------------

    def generate_urls(self, paths: S3SessionPaths) -> S3SessionURLs:
        """Generate a presigned HTTPS URL for every file in *paths*.

        URLs expire after ``self.presign_ttl`` seconds.
        """
        def _sign(key: str) -> str:
            if not key:
                return ""
            try:
                return self._s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": key},
                    ExpiresIn=self.presign_ttl,
                )
            except Exception as exc:
                logger.error("Presign failed for key %s: %s", key, exc)
                return ""

        return S3SessionURLs(
            session_id=paths.session_id,
            cam_head=_sign(paths.cam_head),
            cam_left=_sign(paths.cam_left),
            cam_right=_sign(paths.cam_right),
            cam_head_jsonl=_sign(paths.cam_head_jsonl),
            cam_left_jsonl=_sign(paths.cam_left_jsonl),
            cam_right_jsonl=_sign(paths.cam_right_jsonl),
            metadata=_sign(paths.metadata),
            gripper_left=_sign(paths.gripper_left),
            gripper_right=_sign(paths.gripper_right),
            tracker=_sign(paths.tracker),
        )

    # ------------------------------------------------------------------
    # Convenience: one-shot browse + sign
    # ------------------------------------------------------------------

    def open_session(self, session_prefix: str) -> S3SessionURLs:
        """Resolve keys and generate presigned URLs for *session_prefix* in one call."""
        paths = self.resolve_session_paths(session_prefix)
        return self.generate_urls(paths)

    # ------------------------------------------------------------------
    # In-memory helpers for small files (CSV, JSON, JSONL)
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_text(url: str, timeout: int = 30) -> str:
        """Download a presigned URL and return its content as a string."""
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def fetch_csv(url: str, timeout: int = 30) -> pd.DataFrame:
        """Download a presigned URL and parse it as a CSV DataFrame."""
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))

    @staticmethod
    def fetch_json(url: str, timeout: int = 30) -> dict:
        """Download a presigned URL and parse it as JSON."""
        import json
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return json.loads(resp.text)
