"""Post-download CSV integrity validator.

Vérifie que les fichiers CSV téléchargés depuis HDFS sont complets avant de
charger la session.  Les problèmes sont classés en deux catégories :

* **Erreurs fatales** — fichier absent, illisible ou vide : la session ne peut
  pas être chargée du tout.
* **Avertissements** — valeurs NaN ou cellules vides dans les données : la
  session *peut* être chargée mais les données sont incomplètes.  L'utilisateur
  choisit s'il veut quand même travailler dessus ou rejeter le job.

Usage ::

    from src.core.csv_validator import validate_job_csvs, CSVValidationReport

    report = validate_job_csvs(local_files)

    if report.fatal_errors:
        # Impossible de continuer — afficher les erreurs et rejeter
        ...
    elif report.warnings:
        # Données incomplètes — demander à l'utilisateur
        ...
    else:
        # Tout est propre
        ...
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from ..storage.nas_client import LocalJobFiles

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class CSVValidationError(Exception):
    """Levée (compatibilité) quand au moins une erreur fatale est détectée."""

    def __init__(self, issues: List[str]) -> None:
        self.issues = issues
        joined = "\n".join(f"  • {i}" for i in issues)
        super().__init__(f"Données CSV incomplètes :\n{joined}")


@dataclass
class CSVValidationReport:
    """Résultat complet de la validation des CSV d'un job."""

    fatal_errors: List[str] = field(default_factory=list)
    """Erreurs bloquantes : fichier absent, illisible ou vide."""

    warnings: List[str] = field(default_factory=list)
    """Avertissements non bloquants : NaN ou cellules vides dans les données."""

    @property
    def has_issues(self) -> bool:
        return bool(self.fatal_errors or self.warnings)

    @property
    def is_fatal(self) -> bool:
        return bool(self.fatal_errors)

    def summary(self) -> str:
        lines = []
        if self.fatal_errors:
            lines.append("Erreurs bloquantes :")
            lines.extend(f"  • {e}" for e in self.fatal_errors)
        if self.warnings:
            lines.append("Avertissements (données incomplètes) :")
            lines.extend(f"  • {w}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _FileResult:
    label: str
    path: Path
    fatal_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("%s: UTF-8 decode failed, retrying as latin-1", path.name)
        return pd.read_csv(path, encoding="latin-1")


def _validate_csv(label: str, path: Path) -> _FileResult:
    """Valide un seul fichier CSV.

    Erreurs fatales : fichier absent / illisible / vide / sans données.
    Avertissements : NaN ou cellules vides dans les colonnes de données.
    """
    result = _FileResult(label=label, path=path)

    # --- Erreurs fatales ---

    if not path.exists():
        result.fatal_errors.append(f"Fichier absent : {path.name}")
        return result

    try:
        size = path.stat().st_size
    except Exception as exc:
        result.fatal_errors.append(f"{path.name} : impossible de lire la taille ({exc})")
        return result

    if size == 0:
        result.fatal_errors.append(f"{path.name} est vide (0 octet)")
        return result

    try:
        df = _read_csv_robust(path)
    except Exception as exc:
        result.fatal_errors.append(f"{path.name} illisible : {exc}")
        return result

    if df.empty:
        result.fatal_errors.append(f"{path.name} ne contient aucune ligne de données")
        return result

    if "time_seconds" not in df.columns and "timestamp" not in df.columns:
        result.fatal_errors.append(f"{path.name} : colonne 'time_seconds' (ou 'timestamp') absente")
        return result

    # --- Detect packed serial string format ---
    # Format: columns t_ms/opening_mm/gripper_side/sw are empty, all data is in angle_deg
    # as a raw string like "T=85524 ID=ARD-R-00001  SW=ON   Ouverture=  0.0 mm  Angle= -0.26°"
    # These columns are populated at load time by session_loader — skip them here.
    _PACKED_COLS = {"t_ms", "t_ms_corrected_ns", "gripper_side", "sw", "opening_mm", "angle_deg"}
    is_packed = (
        "angle_deg" in df.columns
        and all(
            col not in df.columns or pd.to_numeric(df[col], errors="coerce").notna().sum() == 0
            for col in ["t_ms", "opening_mm"]
        )
        and df["angle_deg"].dropna().astype(str).str.contains("Ouverture=").any()
    )
    cols_to_skip = _PACKED_COLS if is_packed else set()
    if is_packed:
        logger.info(
            "%s : format packed serial détecté — colonnes %s ignorées pour la validation",
            path.name, sorted(cols_to_skip),
        )

    # --- Avertissements (données présentes mais incomplètes) ---

    try:
        n_nan_ts = int(df["timestamp"].isna().sum())
        if n_nan_ts > 0:
            result.warnings.append(
                f"{path.name} : {n_nan_ts} valeur(s) manquante(s) dans 'timestamp'"
            )
    except Exception as exc:
        result.warnings.append(f"{path.name} : impossible de vérifier 'timestamp' ({exc})")

    try:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            if col in cols_to_skip:
                continue
            try:
                n_nan = int(df[col].isna().sum())
                if n_nan > 0:
                    total = len(df)
                    pct = f"{n_nan / total * 100:.1f}%" if total > 0 else "?"
                    result.warnings.append(
                        f"{path.name} : {n_nan} NaN ({pct}) dans '{col}'"
                    )
            except Exception as exc:
                result.warnings.append(
                    f"{path.name} : impossible de vérifier '{col}' ({exc})"
                )
    except Exception as exc:
        result.warnings.append(
            f"{path.name} : impossible d'inspecter les colonnes numériques ({exc})"
        )

    try:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        for col in str_cols:
            if col == "timestamp" or col in cols_to_skip:
                continue
            try:
                n_empty = int((df[col].astype(str).str.strip() == "").sum())
                if n_empty > 0:
                    result.warnings.append(
                        f"{path.name} : {n_empty} cellule(s) vide(s) dans '{col}'"
                    )
            except Exception as exc:
                result.warnings.append(
                    f"{path.name} : impossible de vérifier '{col}' ({exc})"
                )
    except Exception as exc:
        result.warnings.append(
            f"{path.name} : impossible d'inspecter les colonnes texte ({exc})"
        )

    if not result.fatal_errors and not result.warnings:
        logger.info("[CSV OK] %s — %d lignes, %d colonnes", path.name, len(df), len(df.columns))
    else:
        for e in result.fatal_errors:
            logger.error("[CSV FATAL] %s", e)
        for w in result.warnings:
            logger.warning("[CSV WARN] %s", w)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_job_csvs(local_files: LocalJobFiles) -> CSVValidationReport:
    """Valide les CSV d'un job et retourne un rapport structuré.

    Ne lève plus d'exception — l'appelant consulte ``report.is_fatal`` et
    ``report.warnings`` pour décider de la suite.

    Args:
        local_files: Chemins locaux issus du téléchargement HDFS.

    Returns:
        CSVValidationReport avec les erreurs fatales et les avertissements.
    """
    csv_files = [
        ("tracker_positions",  local_files.tracker),
        ("gripper_left_data",  local_files.gripper_left),
        ("gripper_right_data", local_files.gripper_right),
    ]

    report = CSVValidationReport()
    for label, path in csv_files:
        try:
            file_result = _validate_csv(label, path)
            report.fatal_errors.extend(file_result.fatal_errors)
            report.warnings.extend(file_result.warnings)
        except Exception as exc:
            report.fatal_errors.append(
                f"{label} : erreur inattendue lors de la validation ({exc})"
            )

    if not report.has_issues:
        logger.info("Validation CSV : tous les fichiers sont complets.")

    return report
