# """SeqensorWorker — thread de fond exécutant le pipeline de segmentation fluxseq.
#
# Rôle
# ----
# Ce module fait le pont entre le format de session de l'annotateur (colonnes
# timestamp ISO 8601) et le format attendu par la bibliothèque fluxseq
# (colonne ``time_seconds`` en flottant).
#
# Il exécute ensuite le pipeline complet en 5 étapes :
#     1. Chargement des CSVs bruts (trackers, pinces)
#     2. Adaptation au format fluxseq (conversion temporelle)
#     3. Construction d'une timeline unifiée + alignement de tous les flux
#     4. Extraction de features capteurs (position, angle, jerk, …)
#     5. Segmentation heuristique → clustering ensembliste
#
# À la fin du pipeline, le signal ``segments_ready`` émet la liste des segments
# sous la forme suivante :
#
#     [
#         {
#             "start_idx":  int,     # index de frame de début (= start_t * fps)
#             "end_idx":    int,     # index de frame de fin
#             "start_t":    float,   # temps de début en secondes
#             "end_t":      float,   # temps de fin en secondes
#             "duration_s": float,   # durée en secondes
#             "label":      int,     # identifiant du cluster (0-based)
#             "score_mean": float,   # score moyen de la segmentation
#             "grip_mean":  float,   # angle moyen de la pince sur le segment
#             "jerk_mean":  float,   # jerk moyen (changement d'accélération)
#             "colour":     str,     # couleur hexadécimale associée au cluster
#         },
#         ...
#     ]
# """
#
# import logging
# import sys
# import tempfile
# import time
# from pathlib import Path
# from typing import Optional
#
# import numpy as np
# import pandas as pd
# from PyQt6.QtCore import QThread, pyqtSignal
#
# logger = logging.getLogger(__name__)
#
# # Palette de couleurs attribuées aux clusters par l'UI.
# # Le modulo permet de gérer le cas où il y a plus de 10 clusters.
# LABEL_COLOURS = [
#     "#89b4fa",  # bleu
#     "#a6e3a1",  # vert
#     "#f38ba8",  # rouge
#     "#fab387",  # pêche
#     "#f9e2af",  # jaune
#     "#cba6f7",  # mauve
#     "#89dceb",  # ciel
#     "#74c7ec",  # saphir
#     "#b4befe",  # lavande
#     "#f5c2e7",  # rose
# ]
#
#
# def _label_colour(label: int) -> str:
#     """Retourne la couleur hex associée à un identifiant de cluster."""
#     return LABEL_COLOURS[label % len(LABEL_COLOURS)]
#
#
# # ---------------------------------------------------------------------------
# # Adaptateurs CSV — convertissent les CSVs de session au format fluxseq
# # ---------------------------------------------------------------------------
#
# def _adapt_tracker_csv(tracker_df: pd.DataFrame, ref_time: pd.Timestamp) -> pd.DataFrame:
#     """Convertit le DataFrame ``tracker_positions.csv`` au format fluxseq.
#
#     Format session  : colonne ``timestamp`` (chaîne ISO 8601) +
#                       colonnes ``tracker_{n}_{x,y,z,qw,qx,qy,qz}``
#     Format fluxseq  : colonne ``time_seconds`` (flottant relatif à ref_time) +
#                       mêmes colonnes spatiales
#
#     Si la colonne ``t`` est déjà présente (calculée par session_loader),
#     elle est réutilisée directement pour éviter un double calcul.
#     """
#     if tracker_df.empty:
#         return pd.DataFrame(columns=["time_seconds"])
#
#     df = tracker_df.copy()
#     if "t" in df.columns:
#         # session_loader a déjà calculé le temps relatif en secondes
#         df["time_seconds"] = df["t"]
#     elif "timestamp" in df.columns:
#         df["_abs_time"] = pd.to_datetime(df["timestamp"])
#         df["time_seconds"] = (df["_abs_time"] - ref_time).dt.total_seconds()
#     else:
#         raise ValueError("Tracker DataFrame has neither 't' nor 'timestamp' column")
#
#     # Ne conserver que les colonnes utiles à fluxseq
#     keep = ["time_seconds"] + [c for c in df.columns
#                                 if c.startswith("tracker_")]
#     return df[keep].sort_values("time_seconds").reset_index(drop=True)
#
#
# def _adapt_pince_csv(pince_df: pd.DataFrame, ref_time: pd.Timestamp) -> Optional[pd.DataFrame]:
#     """Convertit un DataFrame pince (pince1 ou pince2) au format fluxseq.
#
#     Format session  : colonne ``timestamp`` (ISO) +
#                       colonne ``raw_data`` (ex. "Angle relatif : 12.3 deg")
#                       OU colonne ``angle_deg`` déjà parsée.
#     Format fluxseq  : ``time_seconds`` (flottant) + ``angle_deg`` (flottant)
#
#     La colonne ``raw_data`` est parsée par une regex qui extrait le premier
#     nombre flottant signé de la chaîne. Les lignes sans valeur angle valide
#     sont supprimées (dropna).
#     """
#     if pince_df is None or pince_df.empty:
#         return None
#
#     df = pince_df.copy()
#
#     # Calcul du temps relatif (même logique que pour les trackers)
#     if "t" in df.columns:
#         df["time_seconds"] = df["t"]
#     elif "timestamp" in df.columns:
#         df["_abs_time"] = pd.to_datetime(df["timestamp"])
#         df["time_seconds"] = (df["_abs_time"] - ref_time).dt.total_seconds()
#     else:
#         return None
#
#     # Résolution de la colonne angle : priorité à angle_deg pré-parsé,
#     # sinon extraction depuis raw_data via regex
#     if "angle_deg" in df.columns:
#         angle = df["angle_deg"]
#     elif "raw_data" in df.columns:
#         import re
#         _re = re.compile(r"(-?\d+(?:\.\d+)?)")
#
#         def _parse(x):
#             """Extrait le premier nombre flottant signé de la chaîne x."""
#             if not isinstance(x, str):
#                 return float("nan")
#             m = _re.search(x)
#             return float(m.group(1)) if m else float("nan")
#
#         angle = df["raw_data"].map(_parse)
#     else:
#         return None
#
#     out = pd.DataFrame({"time_seconds": df["time_seconds"].values,
#                         "angle_deg": angle.values})
#     # Supprimer les lignes sans angle valide (NaN issus du parsing)
#     out = out.dropna(subset=["angle_deg"])
#     return out.sort_values("time_seconds").reset_index(drop=True)
#
#
# # ---------------------------------------------------------------------------
# # Thread de travail
# # ---------------------------------------------------------------------------
#
# class SeqensorWorker(QThread):
#     """Exécute le pipeline fluxseq complet dans un thread de fond Qt.
#
#     Le thread est non-bloquant : il émet des signaux pour communiquer
#     avec l'UI principale (progression, résultat, erreur).
#
#     Parameters
#     ----------
#     session_dir:
#         Chemin vers le répertoire de session. Doit contenir
#         ``tracker_positions.csv`` et ``pince1_data.csv`` ;
#         ``pince2_data.csv`` est optionnel.
#     ref_time:
#         Timestamp de début de session (pd.Timestamp) utilisé pour
#         convertir les timestamps ISO en temps relatifs.
#         Si None, le worker le déduit lui-même depuis le premier
#         timestamp du CSV tracker.
#     fps:
#         Fréquence cible pour l'analyse temporelle (défaut : 30 Hz).
#         Doit correspondre au FPS des vidéos de la session pour que les
#         indices de frames soient cohérents.
#
#     Signals
#     -------
#     segments_ready(list)
#         Émis à la fin du pipeline avec la liste des dicts de segments.
#     progress(str)
#         Émis à chaque étape du pipeline avec un message de statut.
#     error_occurred(str)
#         Émis si une exception est levée pendant le pipeline.
#     """
#
#     # Signal de succès : liste de dicts décrivant les segments détectés
#     segments_ready = pyqtSignal(list)
#     # Mise à jour de progression (texte affiché dans la barre de statut)
#     progress = pyqtSignal(str)
#     # Signal d'erreur : message de l'exception
#     error_occurred = pyqtSignal(str)
#
#     def __init__(
#         self,
#         session_dir: str,
#         ref_time: Optional[pd.Timestamp] = None,
#         fps: float = 30.0,
#         parent=None,
#     ):
#         super().__init__(parent)
#         self._session_dir = Path(session_dir)
#         self._ref_time = ref_time
#         self._fps = fps
#
#     # ------------------------------------------------------------------
#     def run(self) -> None:  # noqa: C901 — pipeline intentionnellement linéaire
#         """Point d'entrée du thread Qt. Capture toute exception du pipeline."""
#         try:
#             self._run_pipeline()
#         except Exception as exc:
#             logger.error("SeqensorWorker failed: %s", exc, exc_info=True)
#             self.error_occurred.emit(str(exc))
#
#     def _run_pipeline(self) -> None:
#         """Exécute les 5 étapes du pipeline fluxseq de manière séquentielle."""
#
#         # Ajoute le répertoire Seqensor/ au sys.path pour importer fluxseq.
#         seqensor_path = Path(__file__).resolve().parents[2] / "Seqensor"
#         if str(seqensor_path) not in sys.path:
#             sys.path.insert(0, str(seqensor_path))
#
#         from fluxseq import (  # type: ignore
#             build_timeline,
#             align_to_timeline,
#             build_sensor_features,
#             heuristic_segments,
#             segment_level_features,
#             cluster_segments_ensemble,
#             export_segments_csv,
#         )
#
#         # ------------------------------------------------------------------
#         # Étape 1 : Chargement des CSVs bruts
#         # ------------------------------------------------------------------
#         self.progress.emit("Seqensor [1/5] Chargement des données…")
#
#         tracker_path = self._session_dir / "tracker_positions.csv"
#         pince1_path  = self._session_dir / "pince1_data.csv"
#         pince2_path  = self._session_dir / "pince2_data.csv"
#
#         if not tracker_path.exists():
#             raise FileNotFoundError(f"tracker_positions.csv not found in {self._session_dir}")
#
#         raw_tracker = pd.read_csv(tracker_path)
#         raw_pince1  = pd.read_csv(pince1_path) if pince1_path.exists() else None
#         raw_pince2  = pd.read_csv(pince2_path) if pince2_path.exists() else None
#
#         # Déduction du ref_time depuis le premier timestamp du CSV tracker
#         if self._ref_time is None:
#             if "timestamp" in raw_tracker.columns:
#                 self._ref_time = pd.to_datetime(raw_tracker["timestamp"].iloc[0])
#             else:
#                 self._ref_time = pd.Timestamp("2000-01-01")
#
#         # ------------------------------------------------------------------
#         # Étape 2 : Adaptation au format fluxseq
#         # ------------------------------------------------------------------
#         self.progress.emit("Seqensor [2/5] Adaptation des formats…")
#
#         tr_fluxseq = _adapt_tracker_csv(raw_tracker, self._ref_time)
#         p1_fluxseq = _adapt_pince_csv(raw_pince1, self._ref_time)
#         p2_fluxseq = _adapt_pince_csv(raw_pince2, self._ref_time)
#
#         logger.info(
#             "Adapted: trackers=%d rows, pince1=%s rows, pince2=%s rows",
#             len(tr_fluxseq),
#             len(p1_fluxseq) if p1_fluxseq is not None else 0,
#             len(p2_fluxseq) if p2_fluxseq is not None else 0,
#         )
#
#         # La pince1 est obligatoire : elle fournit le signal d'ouverture/fermeture
#         if p1_fluxseq is None or len(p1_fluxseq) < 2:
#             raise ValueError("pince1_data.csv vide ou inutilisable — segmentation annulée.")
#
#         # ------------------------------------------------------------------
#         # Étape 3 : Construction de la timeline unifiée + alignement
#         # ------------------------------------------------------------------
#         self.progress.emit("Seqensor [3/5] Alignement temporel…")
#
#         streams = [df for df in [tr_fluxseq, p1_fluxseq, p2_fluxseq] if df is not None]
#         timeline = build_timeline(*streams, fps=self._fps)
#         logger.info(
#             "Timeline: %d frames @ %.1f Hz (%.3f–%.3fs)",
#             len(timeline.t), timeline.fps, timeline.t[0], timeline.t[-1],
#         )
#
#         tr_al = align_to_timeline(
#             tr_fluxseq, timeline,
#             columns=[c for c in tr_fluxseq.columns if c != "time_seconds"],
#         )
#         p1_al = align_to_timeline(p1_fluxseq, timeline)
#         p2_al = align_to_timeline(p2_fluxseq, timeline) if p2_fluxseq is not None else None
#
#         # ------------------------------------------------------------------
#         # Étape 4 : Extraction des features capteur
#         # ------------------------------------------------------------------
#         self.progress.emit("Seqensor [4/5] Extraction des features…")
#
#         feats = build_sensor_features(
#             tr_al,
#             aligned_pince1=p1_al,
#             aligned_pince2=p2_al,
#             fps=timeline.fps,
#             include_quat=True,
#             smooth_speed_ms=60,
#         )
#         logger.info("Features: %d dims × %d frames", len(feats.columns) - 1, len(feats))
#
#         # ------------------------------------------------------------------
#         # Étape 5 : Segmentation heuristique → clustering
#         # ------------------------------------------------------------------
#         self.progress.emit("Seqensor [5/5] Segmentation & clustering…")
#
#         segs = heuristic_segments(feats, fps=timeline.fps)
#         logger.info("Segments found: %d", len(segs))
#
#         if not segs:
#             self.progress.emit("Seqensor : aucun segment détecté.")
#             self.segments_ready.emit([])
#             return
#
#         X = segment_level_features(feats, segs, fps=timeline.fps)
#
#         segs, winner, diag = cluster_segments_ensemble(
#             X, segs,
#             n_labels=None,
#             n_labels_min=2,
#             n_labels_max=min(12, len(segs)),
#             use_gap=False,
#         )
#         logger.info(
#             "Clustering done: winner=%s, gmm_k=%s, hier_k=%s, pca_dims=%s",
#             winner,
#             diag.get("gmm_k", "?"),
#             diag.get("hier_k", "?"),
#             diag.get("pca_dims", "?"),
#         )
#
#         # Post-traitement : ajout des indices de frames et de la couleur UI
#         fps = timeline.fps
#         for seg in segs:
#             seg["start_idx"] = int(round(seg["start_t"] * fps))
#             seg["end_idx"]   = int(round(seg["end_t"]   * fps))
#             seg["colour"]    = _label_colour(seg["label"])
#
#         n_labels = len(set(s["label"] for s in segs))
#         self.progress.emit(
#             f"Seqensor : {len(segs)} segments, {n_labels} classes ({winner})"
#         )
#         self.segments_ready.emit(segs)
