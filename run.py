#!/usr/bin/env python3
"""Simple launcher for VIVE Labeler."""

import sys
import os
import subprocess
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


def _self_update() -> None:
    """git pull + pip install -r requirements.txt avant le démarrage."""
    sep = "─" * 50

    print(sep)
    print("  Mise à jour de l'application (git pull)…")
    print(sep)
    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=project_root,
            capture_output=False,   # affiche la sortie git directement
        )
        if result.returncode != 0:
            print("[WARN] git pull a retourné une erreur — lancement quand même.")
    except FileNotFoundError:
        print("[WARN] git introuvable — mise à jour ignorée.")

    print(sep)
    print("  Installation des dépendances (pip install -r requirements.txt)…")
    print(sep)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
            cwd=project_root,
            check=False,
        )
    except Exception as exc:
        print(f"[WARN] pip install a échoué : {exc} — lancement quand même.")

    print(sep)
    print("  Démarrage de l'application…")
    print(sep)


if __name__ == "__main__":
    _self_update()
    from src.main import main
    main()
