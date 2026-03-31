#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
publish_nas_sessions_to_rabbitmq.py
===================================

Scanne les sessions présentes sur le NAS dans :
  /DB-EXORIA/lakehouse/bronze/landing

Publie chaque session trouvée dans RabbitMQ, une session = un message JSON.

Message publié :
{
  "session_id": "...",
  "session_dir": "/DB-EXORIA/lakehouse/bronze/landing/YYYY/MM/DD/session_xxx",
  "metadata": "/.../metadata.json",
  "tracker_positions": "/.../tracker_positions.csv",
  "gripper_left": "/.../gripper_left_data.csv",
  "gripper_right": "/.../gripper_right_data.csv",
  "head_video": "/.../videos/head.mp4",
  "left_video": "/.../videos/left.mp4",
  "right_video": "/.../videos/right.mp4",
  "head_jsonl": "/.../videos/head.jsonl",
  "left_jsonl": "/.../videos/left.jsonl",
  "right_jsonl": "/.../videos/right.jsonl",
  "published_at": "...Z",
  "source": "nas_session_publisher"
}

Important :
- Ce script publie des chemins NAS.
- Ton consumer actuel `inspect_session.py` ne sait pas traiter des chemins SFTP/NAS
  comme `session_dir` local.
- Si tu veux consommer directement ces messages avec `inspect_session.py`,
  il faut soit :
    1. monter le NAS localement sur la machine du consumer,
    2. soit modifier `resolve_session_dir()` pour gérer le NAS,
    3. soit consommer ces messages avec un autre worker.

Dépendances :
  pip install paramiko pika
"""

import os
import re
import sys
import json
import time
import argparse
import logging
import datetime as dt
import posixpath
from typing import Dict, List, Optional, Tuple

import paramiko
import pika

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# NAS
NAS_HOST        = os.environ.get("NAS_HOST", "192.168.88.248")
NAS_PORT        = int(os.environ.get("NAS_PORT", "22"))
NAS_USER        = os.environ.get("NAS_USER", "EXORIA")
NAS_PASS        = os.environ.get("NAS_PASS", "NasExori@2026!!#")
NAS_BASE_DIR    = "/DB-EXORIA/lakehouse"
NAS_LANDING     = "bronze/landing"
NAS_ROOT        = posixpath.join(NAS_BASE_DIR, NAS_LANDING)

SSH_TIMEOUT     = 20
BANNER_TIMEOUT  = 90
AUTH_TIMEOUT    = 30
KEEPALIVE_SEC   = 30

# RabbitMQ
RABBITMQ_HOST   = os.environ.get("RABBITMQ_HOST", "192.168.88.246")
RABBITMQ_PORT   = int(os.environ.get("RABBITMQ_PORT", "5672"))
RABBITMQ_USER   = os.environ.get("RABBITMQ_USER", "admin")
RABBITMQ_PASS   = os.environ.get("RABBITMQ_PASS", "Admin123456!")
RABBITMQ_VHOST  = os.environ.get("RABBITMQ_VHOST", "/")
RABBITMQ_QUEUE  = os.environ.get("RABBITMQ_QUEUE", "annotation_queue")

# Structure attendue
REQUIRED_FILES = [
    "metadata.json",
    "tracker_positions.csv",
    "gripper_left_data.csv",
    "gripper_right_data.csv",
]

REQUIRED_VIDEOS = [
    "videos/head.mp4",
    "videos/left.mp4",
    "videos/right.mp4",
    "videos/head.jsonl",
    "videos/left.jsonl",
    "videos/right.jsonl",
]

ALL_REQUIRED = REQUIRED_FILES + REQUIRED_VIDEOS

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("nas-publisher")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def is_session_dir_name(name: str) -> bool:
    return bool(re.match(r"^session_\d{8}_\d{6}$", name))


def joinp(*parts: str) -> str:
    return posixpath.join(*parts)


# ──────────────────────────────────────────────────────────────────────────────
# NAS SFTP CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class NASClient:
    def __init__(self) -> None:
        self.ssh: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None

    def connect(self) -> None:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=NAS_HOST,
            port=NAS_PORT,
            username=NAS_USER,
            password=NAS_PASS,
            look_for_keys=False,
            allow_agent=False,
            timeout=SSH_TIMEOUT,
            banner_timeout=BANNER_TIMEOUT,
            auth_timeout=AUTH_TIMEOUT,
        )
        tr = ssh.get_transport()
        if tr:
            tr.set_keepalive(KEEPALIVE_SEC)
        self.ssh = ssh
        self.sftp = ssh.open_sftp()
        log.info("NAS connecté: %s:%s", NAS_HOST, NAS_PORT)

    def close(self) -> None:
        try:
            if self.sftp:
                self.sftp.close()
        except Exception:
            pass
        try:
            if self.ssh:
                self.ssh.close()
        except Exception:
            pass
        self.sftp = None
        self.ssh = None

    def exists(self, path: str) -> bool:
        try:
            self.sftp.stat(path)  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            st = self.sftp.stat(path)  # type: ignore[union-attr]
            return (st.st_mode & 0o170000) == 0o040000
        except Exception:
            return False

    def is_file(self, path: str) -> bool:
        try:
            st = self.sftp.stat(path)  # type: ignore[union-attr]
            return (st.st_mode & 0o170000) == 0o100000
        except Exception:
            return False

    def listdir(self, path: str) -> List[str]:
        return self.sftp.listdir(path)  # type: ignore[union-attr]

    def listdir_attr(self, path: str):
        return self.sftp.listdir_attr(path)  # type: ignore[union-attr]


# ──────────────────────────────────────────────────────────────────────────────
# RABBITMQ PUBLISHER
# ──────────────────────────────────────────────────────────────────────────────

class RabbitPublisher:
    def __init__(self) -> None:
        self.conn: Optional[pika.BlockingConnection] = None
        self.ch = None

    def connect(self) -> None:
        params = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            virtual_host=RABBITMQ_VHOST,
            credentials=pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS),
            heartbeat=60,
            connection_attempts=3,
            retry_delay=2,
        )
        self.conn = pika.BlockingConnection(params)
        self.ch = self.conn.channel()
        self.ch.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
        log.info("RabbitMQ connecté: %s:%s queue=%s", RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_QUEUE)

    def publish(self, message: Dict) -> None:
        if not self.ch:
            raise RuntimeError("RabbitMQ non connecté")
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        self.ch.basic_publish(
            exchange="",
            routing_key=RABBITMQ_QUEUE,
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )

    def close(self) -> None:
        try:
            if self.conn and self.conn.is_open:
                self.conn.close()
        except Exception:
            pass
        self.conn = None
        self.ch = None


# ──────────────────────────────────────────────────────────────────────────────
# SESSION DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────

def session_has_required_structure(nas: NASClient, session_dir: str) -> Tuple[bool, List[str]]:
    missing = []
    for rel in ALL_REQUIRED:
        p = joinp(session_dir, rel)
        if not nas.is_file(p):
            missing.append(rel)
    return len(missing) == 0, missing


def build_message(session_dir: str) -> Dict:
    session_id = posixpath.basename(session_dir)
    return {
        "session_id": session_id,
        "session_dir": session_dir,
        "metadata": joinp(session_dir, "metadata.json"),
        "tracker_positions": joinp(session_dir, "tracker_positions.csv"),
        "gripper_left": joinp(session_dir, "gripper_left_data.csv"),
        "gripper_right": joinp(session_dir, "gripper_right_data.csv"),
        "head_video": joinp(session_dir, "videos/head.mp4"),
        "left_video": joinp(session_dir, "videos/left.mp4"),
        "right_video": joinp(session_dir, "videos/right.mp4"),
        "head_jsonl": joinp(session_dir, "videos/head.jsonl"),
        "left_jsonl": joinp(session_dir, "videos/left.jsonl"),
        "right_jsonl": joinp(session_dir, "videos/right.jsonl"),
        "published_at": now_iso(),
        "source": "nas_session_publisher",
    }


def find_all_sessions(nas: NASClient, root: str) -> List[str]:
    """
    Parcours attendu :
      /DB-EXORIA/lakehouse/bronze/landing/YYYY/MM/DD/session_xxx
    """
    found = []

    if not nas.is_dir(root):
        raise RuntimeError(f"Répertoire NAS introuvable: {root}")

    for year in sorted(nas.listdir(root)):
        year_dir = joinp(root, year)
        if not nas.is_dir(year_dir) or not re.match(r"^\d{4}$", year):
            continue

        for month in sorted(nas.listdir(year_dir)):
            month_dir = joinp(year_dir, month)
            if not nas.is_dir(month_dir) or not re.match(r"^\d{2}$", month):
                continue

            for day in sorted(nas.listdir(month_dir)):
                day_dir = joinp(month_dir, day)
                if not nas.is_dir(day_dir) or not re.match(r"^\d{2}$", day):
                    continue

                for name in sorted(nas.listdir(day_dir)):
                    session_dir = joinp(day_dir, name)
                    if not nas.is_dir(session_dir):
                        continue
                    if not is_session_dir_name(name):
                        continue
                    found.append(session_dir)

    return found


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publie toutes les sessions NAS dans RabbitMQ")
    p.add_argument("--nas-root", default=NAS_ROOT, help="Racine NAS à scanner")
    p.add_argument("--queue", default=RABBITMQ_QUEUE, help="Queue RabbitMQ cible")
    p.add_argument("--dry-run", action="store_true", help="N'envoie rien, affiche seulement")
    p.add_argument("--strict", action="store_true", help="Publie seulement les sessions complètes")
    p.add_argument("--limit", type=int, default=0, help="Limite le nombre de sessions publiées")
    p.add_argument("--contains", default="", help="Filtre les session_id contenant cette chaîne")
    return p.parse_args()


def main() -> int:
    global RABBITMQ_QUEUE
    args = parse_args()
    RABBITMQ_QUEUE = args.queue

    nas = NASClient()
    rabbit = RabbitPublisher()

    try:
        nas.connect()

        sessions = find_all_sessions(nas, args.nas_root)
        log.info("Sessions trouvées: %d", len(sessions))

        if args.contains:
            sessions = [s for s in sessions if args.contains in posixpath.basename(s)]
            log.info("Après filtre --contains='%s': %d", args.contains, len(sessions))

        if args.limit > 0:
            sessions = sessions[:args.limit]
            log.info("Après --limit=%d: %d", args.limit, len(sessions))

        if not args.dry_run:
            rabbit.connect()

        published = 0
        skipped = 0

        for session_dir in sessions:
            session_id = posixpath.basename(session_dir)
            ok, missing = session_has_required_structure(nas, session_dir)

            if args.strict and not ok:
                skipped += 1
                log.warning("SKIP %s: structure incomplète, manquants=%s", session_id, missing)
                continue

            msg = build_message(session_dir)
            if not ok:
                msg["missing_files"] = missing
                msg["structure_ok"] = False
            else:
                msg["structure_ok"] = True

            if args.dry_run:
                print(json.dumps(msg, ensure_ascii=False, indent=2))
            else:
                rabbit.publish(msg)

            published += 1
            log.info("Publié %s -> %s", session_id, RABBITMQ_QUEUE)

        log.info("Terminé. publiées=%d skip=%d total_scannées=%d", published, skipped, len(sessions))
        return 0

    except Exception as e:
        log.exception("Erreur fatale: %s", e)
        return 1

    finally:
        rabbit.close()
        nas.close()


if __name__ == "__main__":
    sys.exit(main())
