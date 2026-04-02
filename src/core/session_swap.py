"""Échange permanent de deux positions caméra dans une session sur disque.

Opérations effectuées (atomiquement via renommages temporaires) :
  1. videos/{a}.mp4  ↔  videos/{b}.mp4
  2. videos/{a}.jsonl ↔  videos/{b}.jsonl   (si présents)
  3. tracker_positions.csv : colonnes tracker_{a}_* ↔ tracker_{b}_*
  4. metadata.json : cameras[id].position swappé pour les deux caméras
                    camera_anchors[a] ↔ camera_anchors[b]

Toutes les modifications sont préparées en mémoire avant d'être écrites,
de sorte qu'une erreur laisse les fichiers intacts autant que possible.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ── API publique ──────────────────────────────────────────────────────────────

def swap_trackers_on_disk(session_dir: str | Path, tracker_a: str, tracker_b: str) -> None:
    """Échange uniquement les colonnes CSV de deux trackers (sans toucher aux vidéos).

    Args:
        session_dir: Répertoire racine de la session.
        tracker_a: Nom du premier tracker (ex. "head").
        tracker_b: Nom du second tracker  (ex. "right").

    Raises:
        ValueError: Si tracker_a == tracker_b.
        FileNotFoundError: Si session_dir n'existe pas.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Répertoire de session introuvable : {session_dir}")
    if tracker_a == tracker_b:
        raise ValueError("Les deux trackers doivent être différents.")

    logger.info("Swap trackers '%s' ↔ '%s' dans %s", tracker_a, tracker_b, session_dir)
    _swap_tracker_columns(session_dir, tracker_a, tracker_b)
    logger.info("Swap trackers terminé.")


def swap_cameras_on_disk(session_dir: str | Path, pos_a: str, pos_b: str) -> None:
    """Échange sur disque les données des caméras pos_a et pos_b.

    Args:
        session_dir: Répertoire racine de la session.
        pos_a: Position de la première caméra (ex. "head").
        pos_b: Position de la seconde caméra  (ex. "right").

    Raises:
        ValueError: Si pos_a == pos_b.
        FileNotFoundError: Si session_dir n'existe pas.
        RuntimeError: Si les deux fichiers vidéo sont introuvables.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Répertoire de session introuvable : {session_dir}")
    if pos_a == pos_b:
        raise ValueError("Les deux positions doivent être différentes.")

    logger.info("Swap caméras '%s' ↔ '%s' dans %s", pos_a, pos_b, session_dir)

    _swap_video_files(session_dir, pos_a, pos_b)
    _swap_tracker_columns(session_dir, pos_a, pos_b)
    _swap_metadata(session_dir, pos_a, pos_b)

    logger.info("Swap terminé.")


# ── Fichiers vidéo et jsonl ───────────────────────────────────────────────────

def _find_video(videos_dir: Path, pos: str) -> Path | None:
    """Retourne le chemin du fichier vidéo pour cette position, ou None."""
    for ext in (".mp4", ".avi"):
        p = videos_dir / f"{pos}{ext}"
        if p.exists():
            return p
    return None


def _swap_video_files(session_dir: Path, pos_a: str, pos_b: str) -> None:
    videos_dir = session_dir / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Dossier videos/ absent dans {session_dir}")

    path_a = _find_video(videos_dir, pos_a)
    path_b = _find_video(videos_dir, pos_b)

    if path_a is None and path_b is None:
        raise RuntimeError(
            f"Aucun fichier vidéo trouvé pour '{pos_a}' ni '{pos_b}' dans {videos_dir}"
        )

    # Swap vidéo (même extension obligatoire pour la simplicité ;
    # sinon on garde l'extension d'origine de chaque fichier)
    if path_a is not None and path_b is not None:
        _atomic_swap(path_a, path_b)
    elif path_a is not None:
        # Seulement A existe → renommer en B (même extension)
        path_a.rename(videos_dir / f"{pos_b}{path_a.suffix}")
        logger.warning("Seul '%s' existait — renommé en '%s'", path_a.name, f"{pos_b}{path_a.suffix}")
    else:
        path_b.rename(videos_dir / f"{pos_a}{path_b.suffix}")
        logger.warning("Seul '%s' existait — renommé en '%s'", path_b.name, f"{pos_a}{path_b.suffix}")

    # Swap jsonl (optionnels)
    jsonl_a = videos_dir / f"{pos_a}.jsonl"
    jsonl_b = videos_dir / f"{pos_b}.jsonl"
    if jsonl_a.exists() and jsonl_b.exists():
        _atomic_swap(jsonl_a, jsonl_b)
    elif jsonl_a.exists():
        jsonl_a.rename(videos_dir / f"{pos_b}.jsonl")
    elif jsonl_b.exists():
        jsonl_b.rename(videos_dir / f"{pos_a}.jsonl")


def _atomic_swap(p1: Path, p2: Path) -> None:
    """Échange deux fichiers via un fichier temporaire."""
    tmp = p1.with_name(f"__swap_tmp_{p1.name}")
    p1.rename(tmp)
    p2.rename(p1)
    tmp.rename(p2)
    logger.debug("Swappé : %s ↔ %s", p1.name, p2.name)


# ── CSV tracker ───────────────────────────────────────────────────────────────

def _swap_tracker_columns(session_dir: Path, pos_a: str, pos_b: str) -> None:
    """Échange les colonnes tracker_{pos_a}_* ↔ tracker_{pos_b}_* dans le CSV."""
    csv_path = session_dir / "tracker_positions.csv"
    if not csv_path.exists():
        logger.warning("tracker_positions.csv absent — skip swap colonnes tracker.")
        return

    df = pd.read_csv(csv_path)

    # Colonnes concernées
    prefix_a = f"tracker_{pos_a}_"
    prefix_b = f"tracker_{pos_b}_"

    cols_a = [c for c in df.columns if c.startswith(prefix_a)]
    cols_b = [c for c in df.columns if c.startswith(prefix_b)]

    if not cols_a and not cols_b:
        logger.warning(
            "Aucune colonne tracker_%s_* ou tracker_%s_* dans le CSV — skip.", pos_a, pos_b
        )
        return

    # Construire un mapping de renommage
    # tracker_{a}_x → __tmp_{a}_x, tracker_{b}_x → tracker_{a}_x, __tmp_{a}_x → tracker_{b}_x
    rename_to_tmp = {c: f"__tmp_{c}" for c in cols_a}
    rename_b_to_a = {c: prefix_a + c[len(prefix_b):] for c in cols_b}
    rename_tmp_to_b = {f"__tmp_{c}": prefix_b + c[len(prefix_a):] for c in cols_a}

    df = df.rename(columns=rename_to_tmp)
    df = df.rename(columns=rename_b_to_a)
    df = df.rename(columns=rename_tmp_to_b)

    df.to_csv(csv_path, index=False)
    logger.debug("Colonnes CSV tracker swappées : %s ↔ %s (%d cols each)", pos_a, pos_b, max(len(cols_a), len(cols_b)))


# ── metadata.json ─────────────────────────────────────────────────────────────

def _swap_metadata(session_dir: Path, pos_a: str, pos_b: str) -> None:
    """Met à jour metadata.json : cameras[].position et camera_anchors."""
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        logger.warning("metadata.json absent — skip mise à jour metadata.")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # cameras: {"0": {"position": "head", ...}, "1": {"position": "left", ...}, ...}
    cameras = raw.get("cameras", {})
    if isinstance(cameras, dict):
        for cam_id, cam_info in cameras.items():
            if not isinstance(cam_info, dict):
                continue
            pos = cam_info.get("position", "")
            if pos == pos_a:
                cam_info["position"] = pos_b
            elif pos == pos_b:
                cam_info["position"] = pos_a

    # camera_anchors: {"head": {"mono_offset_from_record": 1.65}, ...}
    anchors = raw.get("camera_anchors", {})
    if isinstance(anchors, dict) and pos_a in anchors and pos_b in anchors:
        anchors[pos_a], anchors[pos_b] = anchors[pos_b], anchors[pos_a]
    elif isinstance(anchors, dict) and pos_a in anchors:
        anchors[pos_b] = anchors.pop(pos_a)
    elif isinstance(anchors, dict) and pos_b in anchors:
        anchors[pos_a] = anchors.pop(pos_b)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    logger.debug("metadata.json mis à jour pour swap %s ↔ %s", pos_a, pos_b)


# ── Renommage d'une caméra ────────────────────────────────────────────────────

def rename_camera_on_disk(session_dir: str | Path, old_name: str, new_name: str) -> None:
    """Renomme une position caméra de old_name en new_name dans toute la session.

    Opérations effectuées :
      1. videos/{old_name}.mp4  →  videos/{new_name}.mp4
      2. videos/{old_name}.jsonl →  videos/{new_name}.jsonl  (si présent)
      3. tracker_positions.csv : colonnes tracker_{old_name}_* → tracker_{new_name}_*
      4. metadata.json : cameras[id].position et camera_anchors

    Args:
        session_dir: Répertoire racine de la session.
        old_name: Nom actuel de la position caméra.
        new_name: Nouveau nom de la position caméra.

    Raises:
        ValueError: Si old_name == new_name ou new_name est vide.
        FileNotFoundError: Si session_dir n'existe pas.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Répertoire de session introuvable : {session_dir}")
    old_name = old_name.strip()
    new_name = new_name.strip()
    if not old_name or not new_name:
        raise ValueError("old_name et new_name ne peuvent pas être vides.")
    if old_name == new_name:
        raise ValueError("Les noms doivent être différents.")

    logger.info("Renommage caméra '%s' → '%s' dans %s", old_name, new_name, session_dir)

    # 1. Fichiers vidéo et jsonl
    videos_dir = session_dir / "videos"
    if videos_dir.is_dir():
        vid = _find_video(videos_dir, old_name)
        if vid is not None:
            vid.rename(videos_dir / f"{new_name}{vid.suffix}")
            logger.debug("Vidéo renommée : %s → %s%s", vid.name, new_name, vid.suffix)
        jsonl = videos_dir / f"{old_name}.jsonl"
        if jsonl.exists():
            jsonl.rename(videos_dir / f"{new_name}.jsonl")
            logger.debug("JSONL renommé : %s.jsonl → %s.jsonl", old_name, new_name)

    # 2. CSV tracker
    csv_path = session_dir / "tracker_positions.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        prefix_old = f"tracker_{old_name}_"
        prefix_new = f"tracker_{new_name}_"
        rename_map = {c: prefix_new + c[len(prefix_old):] for c in df.columns if c.startswith(prefix_old)}
        if rename_map:
            df = df.rename(columns=rename_map)
            df.to_csv(csv_path, index=False)
            logger.debug("CSV : %d colonnes renommées %s → %s", len(rename_map), prefix_old, prefix_new)

    # 3. metadata.json
    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cameras = raw.get("cameras", {})
        if isinstance(cameras, dict):
            for cam_info in cameras.values():
                if isinstance(cam_info, dict) and cam_info.get("position") == old_name:
                    cam_info["position"] = new_name
        anchors = raw.get("camera_anchors", {})
        if isinstance(anchors, dict) and old_name in anchors:
            anchors[new_name] = anchors.pop(old_name)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        logger.debug("metadata.json mis à jour : %s → %s", old_name, new_name)

    logger.info("Renommage caméra terminé : '%s' → '%s'", old_name, new_name)
