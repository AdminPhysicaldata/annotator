"""Main entry point for VIVE Labeler application."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Suppress ffmpeg/libavcodec warnings (must be set before cv2 is imported)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from .ui.main_window import MainWindow
from .ui.dialogs.mongo_login_dialog import MongoLoginDialog
from .storage.mongodb_client import MongoDBClient
from .utils.config import AppConfig, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="VIVE Labeler — Multi-Camera Annotation Tool")
    parser.add_argument(
        "session_dir",
        nargs="?",
        default=None,
        help="Path to session directory to load on startup",
    )
    args = parser.parse_args()

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("VIVE Labeler")
    app.setOrganizationName("VIVE Labeler")

    # --- Load configuration — never crashes, falls back to defaults ---
    try:
        config = AppConfig.load_from_file(DEFAULT_CONFIG_PATH)
    except Exception as exc:
        # load_from_file itself never raises, but guard defensively
        logger.error("Unexpected config load error: %s", exc)
        QMessageBox.warning(
            None, "Configuration",
            f"Fichier de configuration invalide — utilisation des valeurs par défaut.\n\n{exc}",
        )
        config = AppConfig()

    # --- Connect to MongoDB Atlas ---
    try:
        mongo_client = MongoDBClient(
            connection_string=config.mongodb.connection_string,
            database=config.mongodb.database,
            collection=config.mongodb.collection,
        )
        if not mongo_client.ping():
            raise ConnectionError("Le serveur n'a pas répondu au ping.")
        logger.info("Connected to MongoDB Atlas")
    except Exception as exc:
        logger.error("MongoDB connection failed: %s", exc)
        QMessageBox.critical(
            None, "Erreur de connexion",
            f"Impossible de se connecter au serveur MongoDB :\n{exc}",
        )
        sys.exit(1)

    # --- Annotator login dialog ---
    initial_scenario = ""
    while True:
        try:
            dialog = MongoLoginDialog(mongo_client=mongo_client)
            if dialog.exec() != MongoLoginDialog.DialogCode.Accepted:
                mongo_client.close()
                sys.exit(0)
            username, password = dialog.get_credentials()
            initial_mode = dialog.get_mode()
            initial_scenario = dialog.get_scenario()
        except Exception as exc:
            logger.error("Login dialog error: %s", exc)
            QMessageBox.critical(
                None, "Erreur",
                f"Impossible d'afficher la fenêtre de connexion :\n{exc}",
            )
            mongo_client.close()
            sys.exit(1)

        try:
            authenticated = mongo_client.authenticate_annotator(username, password)
        except Exception as exc:
            logger.error("Authentication error: %s", exc)
            authenticated = False

        if authenticated:
            poste = mongo_client.current_poste or "?"
            logger.info("Annotator '%s' authenticated (poste %s), scenario '%s'", username, poste, initial_scenario)
            break
        else:
            QMessageBox.warning(
                None, "Échec de connexion",
                "Nom d'utilisateur ou mot de passe incorrect.",
            )

    # --- Create and show main window ---
    try:
        window = MainWindow(config=config, session_dir=args.session_dir, mongo_client=mongo_client, annotator_name=username, initial_mode=initial_mode, selected_scenario=initial_scenario)
        # Show who is logged in and on which workstation
        user = mongo_client.current_user
        if user:
            poste = user.get("numero_poste", "?")
            name = user.get("username", username)
            window.setWindowTitle(f"VIVE Labeler — {name}  |  Poste {poste}")
        window.show()
    except Exception as exc:
        logger.critical("MainWindow initialisation failed: %s", exc, exc_info=True)
        QMessageBox.critical(
            None, "Erreur fatale",
            f"Impossible d'initialiser la fenêtre principale :\n{exc}\n\n"
            "Consultez les logs pour plus de détails.",
        )
        mongo_client.close()
        sys.exit(1)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
