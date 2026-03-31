from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------

def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    w = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(np.nan_to_num(x, nan=0.0), w, mode="same")


def _percentile_scale(a: np.ndarray, lo: float = 10.0, hi: float = 90.0) -> np.ndarray:
    """Scale so that [pLo, pHi] → [0, 1]. Values outside can exceed range."""
    a2 = np.nan_to_num(a, nan=0.0)
    p_lo = np.percentile(a2, lo)
    p_hi = np.percentile(a2, hi)
    denom = (p_hi - p_lo) if (p_hi - p_lo) > 1e-12 else 1.0
    return (a2 - p_lo) / denom


# ---------------------------------------------------------------------------
# Segmentation — two independent streams, merged
# ---------------------------------------------------------------------------

def _active_mask_from_signal(
    signal: np.ndarray,
    thr_high: float,
    thr_low: float,
    min_len: int,
    min_gap: int,
) -> np.ndarray:
    """Hysteresis thresholding → boolean active mask."""
    n = len(signal)
    active = np.zeros(n, dtype=bool)
    on = False
    for i in range(n):
        if not on:
            if signal[i] >= thr_high:
                on = True
        else:
            if signal[i] < thr_low:
                on = False
        active[i] = on

    # Merge short gaps
    i = 0
    while i < n:
        if active[i]:
            i += 1
            continue
        j = i
        while j < n and not active[j]:
            j += 1
        gap = j - i
        if gap <= min_gap:
            active[i:j] = True
        i = j + 1

    # Drop too-short bursts
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        if (j - i) < min_len:
            active[i:j] = False
        i = j

    return active


def _mask_to_segments(
    active: np.ndarray,
    t: np.ndarray,
    frame_features: pd.DataFrame,
    score: np.ndarray,
    grip_s: np.ndarray,
    jerk_s: np.ndarray,
) -> List[Dict]:
    n = len(active)
    out = []
    idx = 0
    seg_id = 0
    while idx < n:
        if not active[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and active[idx]:
            idx += 1
        end = idx - 1
        out.append({
            "segment_id": seg_id,
            "start_idx":  int(start),
            "end_idx":    int(end),
            "start_t":    float(t[start]),
            "end_t":      float(t[end]),
            "duration_s": float(t[end] - t[start]),
            "label":      None,
            "score_mean": float(np.nanmean(score[start:end + 1])),
            "grip_mean":  float(np.nanmean(grip_s[start:end + 1])),
            "jerk_mean":  float(np.nanmean(jerk_s[start:end + 1])),
        })
        seg_id += 1
    return out


def heuristic_segments(
    frame_features: pd.DataFrame,
    *,
    fps: float,
    motion_col: str = "motion_speed_sum",
    grip_cols: tuple[str, ...] = ("pince1_ang_vel_abs", "pince2_ang_vel_abs"),
    jerk_col: str = "motion_jerk_sum",
    # Motion segmentation params
    smooth_ms: int = 80,
    thr_motion: float = 0.25,
    # Grip segmentation params — independent stream
    thr_grip_abs: float = 0.20,   # threshold on |ang_vel| normalised by p80
    smooth_grip_ms: int = 40,     # short smoothing to preserve inter-cycle gaps
    # Shared post-processing
    min_action_ms: int = 120,
    min_gap_ms: int = 80,
) -> List[Dict]:
    """Segmentation driven by two independent streams: motion AND gripper.

    Key insight
    -----------
    Motion and gripper operate at very different scales and often at
    different times. Combining them into one score (as before) caused the
    gripper signal to be drowned out when motion was low, and vice versa.

    New approach: build two independent boolean active masks, then take
    their UNION. This guarantees that:
    - A pure gripper action (hand nearly still, fingers moving) is detected
    - A pure motion action (hand moving, gripper static) is detected
    - Coordinated actions (both moving) are also detected

    The final composite score stored per segment combines both signals
    but is only used for diagnostics, not for boundary detection.
    """
    n = len(frame_features)
    if n == 0:
        return []

    dt   = 1.0 / float(fps)
    t    = frame_features["time_seconds"].to_numpy(dtype=np.float64)
    min_len = max(1, int(round((min_action_ms / 1000.0) * fps)))
    min_gap = max(1, int(round((min_gap_ms  / 1000.0) * fps)))

    # ── 1. Motion stream ──────────────────────────────────────────────────
    motion_raw = (
        frame_features[motion_col].to_numpy(dtype=np.float64)
        if motion_col in frame_features.columns else np.zeros(n)
    )
    jerk_raw = (
        frame_features[jerk_col].to_numpy(dtype=np.float64)
        if jerk_col in frame_features.columns else np.zeros(n)
    )

    motion_s = _percentile_scale(motion_raw)
    jerk_s   = _percentile_scale(jerk_raw)

    k_m = max(1, int(round((smooth_ms / 1000.0) * fps)))
    motion_score = _smooth(
        np.maximum(motion_s, 0.0) + 0.3 * np.maximum(jerk_s, 0.0), k_m
    )

    mask_motion = _active_mask_from_signal(
        motion_score, thr_motion, thr_motion * 0.5, min_len, min_gap
    )

    # ── 2. Gripper stream — independent ────────────────────────────────────
    # Use the RAW angle signal directly, not just its derivative.
    # The derivative (angular velocity) has very large values (100–700 deg/s)
    # that collapse during robust scaling when most frames are static.
    # Instead: detect when the angle is MEANINGFULLY different from its local
    # baseline (= actively open or closing), using a rolling z-score.
    grip_raw = np.zeros(n, dtype=np.float64)
    grip_found = False
    for c in grip_cols:
        if c in frame_features.columns:
            arr = frame_features[c].to_numpy(dtype=np.float64)
            valid = np.isfinite(arr)
            if valid.sum() > 0:
                grip_raw += np.nan_to_num(arr, nan=0.0)
                grip_found = True
        else:
            # Try the signed velocity column and take abs
            legacy = c.replace("_ang_vel_abs", "_ang_vel")
            if legacy in frame_features.columns:
                arr = np.abs(frame_features[legacy].to_numpy(dtype=np.float64))
                valid = np.isfinite(arr)
                if valid.sum() > 0:
                    grip_raw += np.nan_to_num(arr, nan=0.0)
                    grip_found = True

    # Grip activity = |gripper angular velocity| (pince columns only).
    # This detects TRANSITIONS (opening/closing) rather than sustained open
    # positions, so each open→close cycle produces a distinct burst.
    # Excludes tracker_i_ang_vel (quaternion-derived) which is unrelated.
    angle_activity = np.zeros(n, dtype=np.float64)
    for col in frame_features.columns:
        # Only pince/grip columns, not tracker quaternion angular velocities
        if ("pince" in col or "grip" in col) and col.endswith("_ang_vel"):
            arr = frame_features[col].to_numpy(dtype=np.float64)
            arr_abs = np.abs(np.nan_to_num(arr, nan=0.0))
            p80 = np.percentile(arr_abs, 80)
            if p80 > 1.0:   # meaningful angular velocity present
                angle_activity += arr_abs / p80

    grip_s = _percentile_scale(grip_raw)

    k_g = max(1, int(round((smooth_grip_ms / 1000.0) * fps)))
    # Primary grip score: angular velocity magnitude (already robust-scaled)
    # Secondary: angle deviation from closed position
    # Use a HIGHER threshold for angle_activity (0.4) to avoid triggering
    # on sustained partial openings between action cycles.
    # angle_activity is already |ang_vel| normalised — use it directly.
    # Don't combine with grip_s (which is also |ang_vel|) to avoid double counting.
    grip_score = _smooth(np.maximum(angle_activity, 0.0), k_g)

    mask_grip = _active_mask_from_signal(
        grip_score, thr_grip_abs, thr_grip_abs * 0.5, min_len, min_gap
    )

    # ── 3. Combine streams intelligently ──────────────────────────────────
    # Core problem: motion stays high BETWEEN gripper cycles (continuous arm
    # movement while fingers open/close repeatedly). A simple OR would erase
    # the inter-cycle gaps and merge everything into one segment.
    #
    # Strategy:
    # A. If the gripper is active:
    #    - Use grip mask as the primary segmentation.
    #    - Only add motion-only segments in regions where the gripper has been
    #      quiet for a sustained period (> 3× min_gap). This catches genuine
    #      arm-only movements that occur when the gripper is resting.
    # B. If no gripper activity at all: fall back to motion mask.

    grip_has_activity = mask_grip.sum() > min_len

    if grip_has_activity:
        # Find inter-cycle gaps in grip mask that are > 3× min_gap
        # (i.e., genuine gripper-quiet periods, not just micro-pauses)
        grip_quiet = ~mask_grip
        sustained_quiet = np.zeros(n, dtype=bool)
        long_gap_thresh = min_gap * 3

        i = 0
        while i < n:
            if not grip_quiet[i]:
                i += 1
                continue
            j = i
            while j < n and grip_quiet[j]:
                j += 1
            gap_len = j - i
            if gap_len >= long_gap_thresh:
                sustained_quiet[i:j] = True
            i = j

        # Add motion segments only in sustained-quiet gripper periods
        motion_in_quiet = mask_motion & sustained_quiet
        active = mask_grip | motion_in_quiet
    else:
        active = mask_motion

    # One final gap-merge and min-length filter
    i = 0
    while i < n:
        if active[i]:
            i += 1
            continue
        j = i
        while j < n and not active[j]:
            j += 1
        if (j - i) <= min_gap:
            active[i:j] = True
        i = j + 1

    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        if (j - i) < min_len:
            active[i:j] = False
        i = j

    # Composite score for diagnostics
    composite = motion_score + 0.7 * grip_score

    return _mask_to_segments(active, t, frame_features, composite, grip_s, jerk_s)


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

def _preprocess(
    seg_features: np.ndarray,
    *,
    variance_threshold: float = 0.95,
    random_state: int = 0,
) -> np.ndarray:
    """RobustScale → adaptive PCA with whitening.

    PCA retains the minimum number of components that explain
    `variance_threshold` of total variance, then whitens (unit variance
    per axis). This removes noise, decorrelates features, and equalises
    the contribution of each dimension to distance calculations — critical
    for reliable automatic k selection.
    """
    X = np.nan_to_num(seg_features, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features < 1:
        return X

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    Z = scaler.fit_transform(X)

    max_comp = min(n_samples - 1, n_features)
    if max_comp < 1:
        return Z

    pca = PCA(n_components=max_comp, whiten=True, random_state=random_state)
    Z_pca = pca.fit_transform(Z)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_keep = max(2, min(n_keep, max_comp))

    return Z_pca[:, :n_keep]


# ---------------------------------------------------------------------------
# k selection — composite of 3 independent criteria
# ---------------------------------------------------------------------------

def _score_partition(Z: np.ndarray, labels: np.ndarray) -> float:
    """Composite clustering quality score ∈ [0, 1], higher = better.

    Combines three complementary indices:
    - Silhouette         : cohesion vs separation, scale-free [-1, 1]
    - Calinski-Harabasz  : ratio variance between/within, unbounded ↑ better
    - Davies-Bouldin     : mean ratio of within-cluster to between-cluster scatter ↓ better

    Each is normalised to [0, 1] before combining so no single criterion
    dominates. Weights: sil 50%, CH 30%, DB 20%.
    """
    n_unique = len(np.unique(labels))
    n_total  = len(labels)
    if n_unique < 2 or n_unique >= n_total:
        return 0.0

    try:
        sil = float(silhouette_score(Z, labels, sample_size=min(500, n_total)))
    except Exception:
        sil = -1.0
    try:
        ch = float(calinski_harabasz_score(Z, labels))
    except Exception:
        ch = 0.0
    try:
        db = float(davies_bouldin_score(Z, labels))
    except Exception:
        db = 10.0

    sil_n = (sil + 1.0) / 2.0
    # CH: unbounded, use log-softmax normalisation
    ch_n  = np.log1p(ch) / (np.log1p(ch) + 1.0)
    # DB: lower = better, clip at 5
    db_n  = 1.0 - min(db, 5.0) / 5.0

    return float(0.50 * sil_n + 0.30 * ch_n + 0.20 * db_n)


def _gap_statistic(Z: np.ndarray, k: int, n_refs: int = 10, random_state: int = 0) -> float:
    """Gap statistic: compare within-cluster dispersion vs uniform reference.

    Gap(k) = E[log W_ref] - log W_data
    Larger gap → k is a better fit than random data.
    """
    rng = np.random.default_rng(random_state)

    def _inertia(data: np.ndarray, k_: int, rs: int) -> float:
        km = KMeans(n_clusters=k_, n_init=3, random_state=rs, max_iter=100)
        return float(km.fit(data).inertia_)

    try:
        W = _inertia(Z, k, random_state)
    except Exception:
        return -np.inf

    mins, maxs = Z.min(axis=0), Z.max(axis=0)
    log_W_refs = []
    for b in range(n_refs):
        ref = rng.uniform(mins, maxs, size=Z.shape)
        try:
            log_W_refs.append(np.log(_inertia(ref, k, random_state + b + 1) + 1e-12))
        except Exception:
            pass

    if not log_W_refs:
        return -np.inf
    return float(np.mean(log_W_refs) - np.log(W + 1e-12))


def _select_k(
    Z: np.ndarray,
    k_min: int,
    k_max: int,
    use_gap: bool,
    random_state: int,
) -> int:
    """Evaluate k in [k_min, k_max] with GMM + Ward; return best k.

    For each candidate k:
    - Fit GMM (flexible, ellipsoidal clusters)
    - Fit Ward (compact, hierarchical)
    - Take the better of the two composite scores
    - Optionally add Gap statistic signal

    Final k = argmax of the combined score curve.
    """
    n = len(Z)
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return k_min

    k_range = list(range(k_min, k_max + 1))
    multi_scores = []
    gap_scores   = []

    for k in k_range:
        best = 0.0

        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  n_init=8, max_iter=300, random_state=random_state)
            s = _score_partition(Z, gmm.fit_predict(Z))
            best = max(best, s)
        except Exception:
            pass

        try:
            ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
            s = _score_partition(Z, ward.fit_predict(Z))
            best = max(best, s)
        except Exception:
            pass

        multi_scores.append(best)

        if use_gap:
            gap_scores.append(_gap_statistic(Z, k, n_refs=8, random_state=random_state))

    arr = np.array(multi_scores, dtype=np.float64)

    if use_gap and gap_scores:
        g = np.array(gap_scores, dtype=np.float64)
        def _norm(a: np.ndarray) -> np.ndarray:
            r = a.max() - a.min()
            return (a - a.min()) / (r + 1e-12)
        arr = 0.6 * _norm(arr) + 0.4 * _norm(g)

    return k_range[int(np.argmax(arr))]


# ---------------------------------------------------------------------------
# Public clustering API
# ---------------------------------------------------------------------------

def cluster_segments_gmm(
    seg_features: np.ndarray,
    segments: List[Dict],
    *,
    n_labels: int | None = None,
    n_labels_max: int = 12,
    n_labels_min: int = 2,
    covariance_type: str = "full",
    variance_threshold: float = 0.95,
    use_gap: bool = False,
    random_state: int = 0,
) -> List[Dict]:
    """GMM clustering with automatic k selection."""
    if not segments:
        return segments
    if seg_features.shape[0] != len(segments):
        raise ValueError("seg_features rows must match number of segments")

    Z = _preprocess(seg_features, variance_threshold=variance_threshold,
                    random_state=random_state)
    n = len(segments)

    k = (min(int(n_labels), n) if n_labels is not None
         else _select_k(Z, n_labels_min, n_labels_max, use_gap, random_state))

    try:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                              n_init=10, max_iter=300, random_state=random_state)
        labels = gmm.fit_predict(Z)
    except Exception:
        gmm = GaussianMixture(n_components=min(2, n), covariance_type="diag",
                              n_init=5, random_state=random_state)
        labels = gmm.fit_predict(Z)

    for s, lab in zip(segments, labels.tolist()):
        s["label"] = int(lab)
    return segments


def cluster_segments_hierarchical(
    seg_features: np.ndarray,
    segments: List[Dict],
    *,
    n_labels: int | None = None,
    n_labels_max: int = 12,
    n_labels_min: int = 2,
    linkage: str = "ward",
    variance_threshold: float = 0.95,
    use_gap: bool = False,
    random_state: int = 0,
) -> List[Dict]:
    """Ward hierarchical clustering with automatic k selection."""
    if not segments:
        return segments

    Z = _preprocess(seg_features, variance_threshold=variance_threshold,
                    random_state=random_state)
    n = len(segments)

    k = (min(int(n_labels), n) if n_labels is not None
         else _select_k(Z, n_labels_min, n_labels_max, use_gap, random_state))

    labels = AgglomerativeClustering(n_clusters=min(k, n), linkage=linkage).fit_predict(Z)
    for s, lab in zip(segments, labels.tolist()):
        s["label"] = int(lab)
    return segments


def cluster_segments_ensemble(
    seg_features: np.ndarray,
    segments: List[Dict],
    *,
    n_labels: int | None = None,
    n_labels_max: int = 12,
    n_labels_min: int = 2,
    variance_threshold: float = 0.95,
    use_gap: bool = False,
    random_state: int = 0,
) -> Tuple[List[Dict], str, Dict]:
    """Run GMM and Ward; keep the solution with the better composite score.

    Returns (segments, winner_name, diagnostics_dict).
    """
    n = len(segments)
    if n < 2:
        segs = cluster_segments_gmm(
            seg_features, [dict(s) for s in segments],
            n_labels=n_labels, n_labels_max=n_labels_max, n_labels_min=n_labels_min,
            variance_threshold=variance_threshold, use_gap=use_gap, random_state=random_state,
        )
        return segs, "gmm", {"winner": "gmm", "reason": "too few segments"}

    Z = _preprocess(seg_features, variance_threshold=variance_threshold,
                    random_state=random_state)

    segs_gmm  = cluster_segments_gmm(
        seg_features, [dict(s) for s in segments],
        n_labels=n_labels, n_labels_max=n_labels_max, n_labels_min=n_labels_min,
        variance_threshold=variance_threshold, use_gap=use_gap, random_state=random_state,
    )
    segs_hier = cluster_segments_hierarchical(
        seg_features, [dict(s) for s in segments],
        n_labels=n_labels, n_labels_max=n_labels_max, n_labels_min=n_labels_min,
        variance_threshold=variance_threshold, use_gap=use_gap, random_state=random_state,
    )

    lbl_gmm  = np.array([s["label"] for s in segs_gmm])
    lbl_hier = np.array([s["label"] for s in segs_hier])
    sc_gmm   = _score_partition(Z, lbl_gmm)
    sc_hier  = _score_partition(Z, lbl_hier)

    diag = {
        "gmm_k":    len(np.unique(lbl_gmm)),  "gmm_score":  round(sc_gmm, 4),
        "hier_k":   len(np.unique(lbl_hier)), "hier_score": round(sc_hier, 4),
        "pca_dims": Z.shape[1],
    }

    if sc_hier >= sc_gmm:
        diag["winner"] = "hierarchical"
        return segs_hier, "hierarchical", diag
    diag["winner"] = "gmm"
    return segs_gmm, "gmm", diag


def cluster_segments_kmeans(
    seg_features: np.ndarray,
    segments: List[Dict],
    *,
    n_labels: int,
    random_state: int = 0,
) -> List[Dict]:
    """K-means (legacy). Prefer cluster_segments_ensemble."""
    if n_labels <= 0:
        raise ValueError("n_labels must be > 0")
    if not segments:
        return segments
    n_labels = min(int(n_labels), len(segments))
    if seg_features.shape[0] != len(segments):
        raise ValueError("seg_features rows must match number of segments")

    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(np.nan_to_num(seg_features, nan=0.0))
    labels = KMeans(n_clusters=n_labels, n_init="auto", random_state=random_state).fit_predict(Z)
    for s, lab in zip(segments, labels.tolist()):
        s["label"] = int(lab)
    return segments


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_segments_csv(segments: List[Dict], path: str) -> None:
    pd.DataFrame(segments).to_csv(path, index=False)
