"""MongoDB client for the annotator — connects to physical_data.annotators."""

import logging
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure, PyMongoError

logger = logging.getLogger(__name__)


class MongoDBClient:
    """Thin wrapper around pymongo for the annotator collection."""

    def __init__(
        self,
        connection_string: str,
        database: str = "physical_data",
        collection: str = "annotators",
    ):
        self._database_name = database
        self._collection_name = collection

        self._client: MongoClient = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            retryReads=True,
        )
        self._db: Database = self._client[database]
        self._collection: Collection = self._db[collection]
        self._current_user: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the server is reachable and credentials are valid."""
        try:
            self._client.admin.command("ping")
            return True
        except (ConnectionFailure, OperationFailure) as exc:
            logger.error("MongoDB ping failed: %s", exc)
            return False
        except Exception as exc:
            logger.error("MongoDB ping unexpected error: %s", exc)
            return False

    def close(self) -> None:
        try:
            self._client.close()
        except Exception as exc:
            logger.warning("MongoDB close error (ignored): %s", exc)

    # ------------------------------------------------------------------
    # Annotator authentication
    # ------------------------------------------------------------------

    def authenticate_annotator(self, username: str, password: str) -> bool:
        """Check username/password against the annotators collection (plaintext).

        The document structure is:
            { username, password (plaintext), numero_poste, email, created_at, role }

        Never raises — returns False on any DB or network error.
        """
        try:
            user = self._collection.find_one({"username": username})
        except Exception as exc:
            logger.error("MongoDB find_one failed during auth: %s", exc)
            return False

        if user is None:
            logger.warning("No user found with username '%s'", username)
            return False

        stored_password = user.get("password", "")
        if not stored_password:
            logger.warning("No password stored for user '%s'", username)
            return False

        if password == stored_password:
            self._current_user = user
            logger.info(
                "Authenticated user '%s' (poste %s)",
                username,
                user.get("numero_poste", "?"),
            )
            return True

        return False

    @property
    def current_poste(self) -> Optional[str]:
        """Return the numero_poste of the logged-in user, or None."""
        if self._current_user is None:
            return None
        return self._current_user.get("numero_poste")

    @property
    def current_user(self) -> Optional[Dict[str, Any]]:
        return self._current_user

    @property
    def current_role(self) -> str:
        """Return the role of the logged-in user: 'annotator' or 'chef'."""
        if self._current_user is None:
            return "annotator"
        return self._current_user.get("role", "annotator")

    @property
    def is_chef(self) -> bool:
        """Return True if the logged-in user has the 'chef' role."""
        return self.current_role == "chef"

    # ------------------------------------------------------------------
    # Scenario label management (collection: scenarios)
    # ------------------------------------------------------------------

    def get_scenario(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Find a scenario document by its 'nom' field."""
        try:
            return self._db["scenarios"].find_one({"nom": scenario_name})
        except Exception as exc:
            logger.error("MongoDB get_scenario failed: %s", exc)
            return None

    def get_scenario_labels(self, scenario_name: str) -> List[Dict[str, Any]]:
        """Return the labels list for a given scenario name.

        Each label dict has at minimum: { name, color }.
        Returns [] if the scenario is not found or on error.
        """
        scenario = self.get_scenario(scenario_name)
        if scenario is None:
            return []
        return scenario.get("labels", [])

    def set_scenario_labels(self, scenario_name: str, labels: List[Dict[str, Any]]) -> bool:
        """Replace the labels array for a scenario (chef only).

        labels: list of { name: str, color: str, description: str }
        Returns True on success, False on error.
        """
        try:
            from datetime import datetime, timezone
            result = self._db["scenarios"].update_one(
                {"nom": scenario_name},
                {"$set": {"labels": labels, "updated_at": datetime.now(timezone.utc)}},
            )
            return result.matched_count > 0
        except Exception as exc:
            logger.error("MongoDB set_scenario_labels failed: %s", exc)
            return False

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """Return all scenario documents (nom + description + actif).

        Fusionne MongoDB et le fichier JSON local : MongoDB a priorité pour les
        scénarios déjà présents, mais tout scénario absent de MongoDB est ajouté
        depuis le fichier (ex: scénario récemment créé non encore synchronisé).
        Falls back to the bundled JSON file only if the DB query fails.
        """
        file_docs = self._load_scenarios_from_file()
        try:
            docs = list(self._db["scenarios"].find(
                {},
                {"nom": 1, "description": 1, "actif": 1, "labels": 1}
            ))
            logger.info("list_scenarios: %d scénario(s) depuis MongoDB", len(docs))
            # Ajouter les scénarios du fichier JSON absents de MongoDB
            existing_noms = {d.get("nom") for d in docs}
            for fdoc in file_docs:
                nom = fdoc.get("nom")
                if nom and nom not in existing_noms:
                    logger.info("list_scenarios: '%s' absent de MongoDB — ajouté depuis le fichier local", nom)
                    docs.append({k: fdoc[k] for k in ("nom", "description", "actif", "labels") if k in fdoc})
            return docs
        except Exception as exc:
            logger.error("MongoDB list_scenarios failed: %s", exc)

        # Fallback complet : fichier local uniquement
        return file_docs

    def _load_scenarios_from_file(self) -> List[Dict[str, Any]]:
        """Load scenarios from the bundled physical_data.scenarios.json file."""
        import json
        from pathlib import Path
        candidates = [
            Path(__file__).parent.parent.parent / "data" / "physical_data.scenarios.json",
            Path(__file__).parent.parent / "data" / "physical_data.scenarios.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    logger.info("Scénarios chargés depuis le fichier local : %s (%d docs)", path, len(raw))
                    return raw
                except Exception as exc:
                    logger.error("Impossible de lire %s : %s", path, exc)
        logger.warning("Fichier de scénarios introuvable — aucun scénario disponible.")
        return []

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def db(self) -> Database:
        return self._db

    @property
    def collection(self) -> Collection:
        return self._collection

    # ------------------------------------------------------------------
    # Convenience CRUD — all operations are guarded against network errors
    # ------------------------------------------------------------------

    def insert(self, document: Dict[str, Any]) -> Optional[str]:
        """Insert a document. Returns inserted_id string, or None on error."""
        try:
            result = self._collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as exc:
            logger.error("MongoDB insert failed: %s", exc)
            return None

    def find(self, query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find documents matching query. Returns empty list on error."""
        try:
            return list(self._collection.find(query or {}))
        except Exception as exc:
            logger.error("MongoDB find failed: %s", exc)
            return []

    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document. Returns None on error."""
        try:
            return self._collection.find_one(query)
        except Exception as exc:
            logger.error("MongoDB find_one failed: %s", exc)
            return None

    def update(self, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        """Update documents matching query. Returns modified count, 0 on error."""
        try:
            result = self._collection.update_many(query, {"$set": update})
            return result.modified_count
        except Exception as exc:
            logger.error("MongoDB update failed: %s", exc)
            return 0

    def delete(self, query: Dict[str, Any]) -> int:
        """Delete documents matching query. Returns deleted count, 0 on error."""
        try:
            result = self._collection.delete_many(query)
            return result.deleted_count
        except Exception as exc:
            logger.error("MongoDB delete failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Session counter
    # ------------------------------------------------------------------

    def increment_session_count(self, username: str) -> bool:
        """Increment the session_count field for the given annotator.

        Uses $inc so the field is created automatically if absent.
        Returns True on success, False on any error.
        """
        from datetime import datetime, timezone
        try:
            result = self._collection.update_one(
                {"username": username},
                {
                    "$inc": {"session_count": 1},
                    "$set": {"last_session_at": datetime.now(timezone.utc)},
                },
            )
            if result.matched_count == 0:
                logger.warning("increment_session_count: user '%s' not found", username)
                return False
            logger.info("Session count incremented for '%s'", username)
            return True
        except Exception as exc:
            logger.error("MongoDB increment_session_count failed: %s", exc)
            return False
