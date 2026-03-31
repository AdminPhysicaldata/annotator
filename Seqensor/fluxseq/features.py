from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------

def _derivative(x: np.ndarray, dt: float) -> np.ndarray:
    dx = np.empty_like(x)
    dx[:] = np.nan
    if len(x) < 2:
        return dx
    dx[1:] = (x[1:] - x[:-1]) / dt
    dx[0] = dx[1]
    return dx


def _smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    w = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(np.nan_to_num(x, nan=0.0), w, mode="same")


def _quat_angular_velocity(
    qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, dt: float
) -> np.ndarray:
    """Angular velocity (rad/s) from quaternion sequence — fully vectorised."""
    n = len(qw)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    Q = np.stack([qw, qx, qy, qz], axis=1).astype(np.float64)
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    Q /= np.where(norms < 1e-9, 1.0, norms)
    q0, q1 = Q[:-1], Q[1:]
    q0c = q0 * np.array([1., -1., -1., -1.])
    w0, x0, y0, z0 = q0c.T
    w1, x1, y1, z1 = q1.T
    rw = np.clip(np.abs(w0*w1 - x0*x1 - y0*y1 - z0*z1), 0., 1.)
    ang_vel = np.empty(n, dtype=np.float64)
    ang_vel[1:] = 2.0 * np.arccos(rw) / dt
    ang_vel[0] = ang_vel[1]
    return ang_vel


def _trajectory_curvature(v: np.ndarray) -> np.ndarray:
    """Curvature κ = angle between consecutive velocity vectors — vectorised.

    Related to the 2/3 power law: v ∝ κ^{-1/3}.
    High curvature → lower expected speed.
    """
    n = len(v)
    c = np.zeros(n, dtype=np.float64)
    if n < 3:
        return c
    vp, vn = v[:-2], v[2:]
    np_ = np.linalg.norm(vp, axis=1)
    nn  = np.linalg.norm(vn, axis=1)
    valid = (np_ > 1e-9) & (nn > 1e-9)
    dot = np.einsum("ij,ij->i", vp, vn)
    cos = np.where(valid, dot / (np_ * nn), 0.0)
    c[1:-1] = np.arccos(np.clip(cos, -1., 1.))
    return c


# ---------------------------------------------------------------------------
# Motor-control kinematic features (new)
# ---------------------------------------------------------------------------

def _minimum_jerk_residual(speed: np.ndarray) -> float:
    """Measure how closely a speed profile matches the minimum-jerk bell curve.

    Flash & Hogan (1985): for a point-to-point movement of duration T,
    the optimal speed profile is:
        v(τ) ∝ τ² (1 − τ)²    where τ = t/T ∈ [0,1]
    (derivative of the 5th-degree polynomial 10τ³ − 15τ⁴ + 6τ⁵)

    Returns 1 − Pearson correlation with the template.
    0 = perfect minimum-jerk, 1 = completely different shape.
    """
    n = len(speed)
    if n < 4:
        return 1.0
    tau = np.linspace(0., 1., n)
    template = tau**2 * (1.0 - tau)**2
    sp = np.nan_to_num(speed, nan=0.0)
    sp_std = sp.std()
    t_std  = template.std()
    if sp_std < 1e-12 or t_std < 1e-12:
        return 1.0
    corr = float(np.corrcoef(sp, template)[0, 1])
    return float(np.clip(1.0 - corr, 0.0, 2.0))


def _speed_skewness(speed: np.ndarray) -> float:
    """Skewness of the speed profile over a movement.

    Negative → peak speed in the first half (fast start, slow end = deceleration phase dominant).
    Positive → peak speed in the second half (slow start, fast end = rare, unexpected).
    Near zero → symmetric bell = canonical minimum-jerk.

    Human voluntary movements tend toward slight negative skew (Fitts-like
    deceleration phase is longer than acceleration phase for precise targets).
    """
    sp = np.nan_to_num(speed, nan=0.0)
    n  = len(sp)
    if n < 3:
        return 0.0
    mu  = sp.mean()
    std = sp.std()
    if std < 1e-12:
        return 0.0
    return float(np.mean(((sp - mu) / std) ** 3))


def _log_speed_kurtosis(speed: np.ndarray) -> float:
    """Excess kurtosis of log(1 + speed).

    Human speed peaks follow an approximately log-normal distribution across
    repetitions. Within one movement, high kurtosis means a sharp, isolated
    peak (ballistic), low kurtosis means a flat plateau (force-controlled).
    """
    sp = np.log1p(np.nan_to_num(np.abs(speed), nan=0.0))
    n  = len(sp)
    if n < 4:
        return 0.0
    mu  = sp.mean()
    std = sp.std()
    if std < 1e-12:
        return 0.0
    return float(np.mean(((sp - mu) / std) ** 4) - 3.0)  # excess kurtosis


def _power_law_residual(speed: np.ndarray, curvature: np.ndarray) -> float:
    """Residual from the 2/3 power law: v ∝ κ^{-1/3}.

    Lacquaniti et al. (1983): in curved movements, speed and curvature are
    linked by v = C · κ^{-1/3}. A low residual means the movement follows
    the power law — strongly suggesting voluntary, smooth motor control.
    A high residual suggests the movement is either noise, a straight-line
    movement (κ ≈ 0, law undefined), or a different motor primitive.

    Returns the coefficient of variation of the residuals (normalised).
    If curvature is near zero throughout (straight movement), returns NaN.
    """
    valid = curvature > 1e-4
    if valid.sum() < 4:
        return float("nan")
    kappa = curvature[valid]
    v     = np.nan_to_num(speed[valid], nan=0.0)
    v_pred = kappa ** (-1.0 / 3.0)
    # Scale v_pred to match v in mean
    scale  = v.mean() / (v_pred.mean() + 1e-12)
    resid  = v - scale * v_pred
    cv     = resid.std() / (v.mean() + 1e-12)
    return float(np.clip(cv, 0.0, 10.0))


def _normalised_jerk_score(speed: np.ndarray, duration: float, amplitude: float) -> float:
    """Dimensionless jerk score (Teulings et al., 1997 variant).

    NJS = sqrt(0.5 * ∫ j²(t) dt) * T^{5/2} / D

    where T = duration, D = amplitude (path length or speed range),
    j(t) = third derivative of position ≈ second derivative of speed.

    Lower = smoother (closer to minimum-jerk ideal).
    Computed from the speed profile directly (j ≈ d²v/dt²).
    """
    n = len(speed)
    if n < 4 or duration < 1e-6 or amplitude < 1e-9:
        return float("nan")
    sp = np.nan_to_num(speed, nan=0.0)
    dt = duration / (n - 1)
    # Second derivative of speed ≈ jerk of speed
    d1 = np.diff(sp) / dt
    d2 = np.diff(d1) / dt
    jerk_sq_integral = 0.5 * np.sum(d2**2) * dt
    njs = np.sqrt(jerk_sq_integral) * (duration ** 2.5) / amplitude
    return float(np.clip(njs, 0.0, 1e6))


def _speed_peak_time(speed: np.ndarray) -> float:
    """Normalised time of peak speed (0 = start, 1 = end).

    For minimum-jerk: peak at τ = 0.5 (symmetric).
    For Fitts-like movements toward small targets: peak at τ < 0.5
    (faster approach, longer deceleration).
    """
    sp = np.nan_to_num(speed, nan=-np.inf)
    if len(sp) < 2:
        return 0.5
    return float(np.argmax(sp)) / max(len(sp) - 1, 1)


# ---------------------------------------------------------------------------
# Frame-level feature extraction
# ---------------------------------------------------------------------------

def build_sensor_features(
    aligned_trackers: pd.DataFrame,
    aligned_pince1: pd.DataFrame | None = None,
    aligned_pince2: pd.DataFrame | None = None,
    *,
    fps: float,
    include_quat: bool = True,
    smooth_speed_ms: int = 60,
) -> pd.DataFrame:
    """Build frame-level features from aligned sensor streams.

    Standard kinematic features (per tracker i=1,2,3):
      position (x,y,z), directional velocity (vx,vy,vz),
      speed, acceleration, jerk, curvature, quaternion angular velocity.

    Motor-control features (per tracker):
      speed_log     — log(1 + speed), linearises log-normal distribution
      v_kappa_ratio — speed / κ^{-1/3}, residual from 2/3 power law
                      (near 0 → obeys power law = smooth voluntary motion)

    Cross-tracker:
      pairwise distance, distance velocity, velocity cosine correlation.

    Gripper features:
      angle_deg, signed ang_vel, abs ang_vel, state (+1/-1/0), angle_rel.

    Global aggregates:
      motion_speed_sum, motion_acc_sum, motion_jerk_sum.
    """
    dt       = 1.0 / float(fps)
    smooth_k = max(1, int(round((smooth_speed_ms / 1000.0) * fps)))

    feats = pd.DataFrame({"time_seconds": aligned_trackers["time_seconds"].to_numpy()})
    n = len(feats)
    trackers_present = []

    for i in [1, 2, 3]:
        pos_cols = [f"tracker_{i}_x", f"tracker_{i}_y", f"tracker_{i}_z"]
        if not all(c in aligned_trackers.columns for c in pos_cols):
            continue
        trackers_present.append(i)

        for ax in ["x", "y", "z"]:
            feats[f"tracker_{i}_{ax}"] = aligned_trackers[f"tracker_{i}_{ax}"].to_numpy(dtype=np.float64)

        p = feats[[f"tracker_{i}_x", f"tracker_{i}_y", f"tracker_{i}_z"]].to_numpy(dtype=np.float64)
        v = _derivative(p, dt)
        a = _derivative(v, dt)
        j = _derivative(a, dt)

        speed     = np.linalg.norm(v, axis=1)
        speed_sm  = _smooth_1d(speed, smooth_k)
        curvature = _trajectory_curvature(v)

        feats[f"tracker_{i}_vx"]        = v[:, 0]
        feats[f"tracker_{i}_vy"]        = v[:, 1]
        feats[f"tracker_{i}_vz"]        = v[:, 2]
        feats[f"tracker_{i}_speed"]     = speed_sm
        feats[f"tracker_{i}_acc"]       = _smooth_1d(np.linalg.norm(a, axis=1), smooth_k)
        feats[f"tracker_{i}_jerk"]      = np.linalg.norm(j, axis=1)
        feats[f"tracker_{i}_curvature"] = curvature

        # log-speed: linearises log-normal distribution of speed peaks
        feats[f"tracker_{i}_speed_log"] = np.log1p(speed_sm)

        # 2/3 power law residual per frame:
        # expected_v ∝ κ^{-1/3}; ratio > 1 = faster than expected for this curvature
        kappa_safe = np.where(curvature > 1e-4, curvature, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            v_pred = kappa_safe ** (-1.0 / 3.0)
        finite = np.isfinite(v_pred)
        if finite.sum() > 1:
            scale = np.nanmean(speed_sm[finite]) / (np.nanmean(v_pred[finite]) + 1e-12)
            ratio = speed_sm / (scale * np.nan_to_num(v_pred, nan=speed_sm.mean()) + 1e-12)
        else:
            ratio = np.ones(n, dtype=np.float64)
        feats[f"tracker_{i}_powerlaw_ratio"] = np.clip(ratio, 0.0, 10.0)

        # Quaternion angular velocity
        quat_cols = [f"tracker_{i}_qw", f"tracker_{i}_qx", f"tracker_{i}_qy", f"tracker_{i}_qz"]
        if all(c in aligned_trackers.columns for c in quat_cols):
            qw = aligned_trackers[f"tracker_{i}_qw"].to_numpy(dtype=np.float64)
            qx = aligned_trackers[f"tracker_{i}_qx"].to_numpy(dtype=np.float64)
            qy = aligned_trackers[f"tracker_{i}_qy"].to_numpy(dtype=np.float64)
            qz = aligned_trackers[f"tracker_{i}_qz"].to_numpy(dtype=np.float64)
            feats[f"tracker_{i}_ang_vel"] = _quat_angular_velocity(qw, qx, qy, qz, dt)
            if include_quat:
                for q, arr in zip(["qw", "qx", "qy", "qz"], [qw, qx, qy, qz]):
                    feats[f"tracker_{i}_{q}"] = arr

    # Cross-tracker features
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        if i not in trackers_present or j not in trackers_present:
            continue
        pi = feats[[f"tracker_{i}_x", f"tracker_{i}_y", f"tracker_{i}_z"]].to_numpy(dtype=np.float64)
        pj = feats[[f"tracker_{j}_x", f"tracker_{j}_y", f"tracker_{j}_z"]].to_numpy(dtype=np.float64)
        d  = np.linalg.norm(pi - pj, axis=1)
        feats[f"dist_{i}_{j}"]     = d
        feats[f"dist_vel_{i}_{j}"] = _derivative(d, dt)

    if len(trackers_present) >= 2:
        a_id, b_id = trackers_present[0], trackers_present[1]
        va = feats[[f"tracker_{a_id}_vx", f"tracker_{a_id}_vy", f"tracker_{a_id}_vz"]].to_numpy(dtype=np.float64)
        vb = feats[[f"tracker_{b_id}_vx", f"tracker_{b_id}_vy", f"tracker_{b_id}_vz"]].to_numpy(dtype=np.float64)
        dot   = np.einsum("ij,ij->i", va, vb)
        denom = np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            feats["vel_correlation_1_2"] = np.where(denom > 1e-9, dot / denom, 0.0)

    # Gripper features
    def _add_pince(prefix: str, df: pd.DataFrame | None) -> None:
        if df is None:
            return
        angle_col = "angle_deg" if "angle_deg" in df.columns else next(
            (c for c in df.columns if c != "time_seconds"), None
        )
        if angle_col is None:
            return

        ang = df[angle_col].to_numpy(dtype=np.float64)
        ang_vel = _derivative(ang, dt)

        feats[f"{prefix}_angle_deg"]   = ang
        feats[f"{prefix}_ang_vel"]     = ang_vel
        feats[f"{prefix}_ang_vel_abs"] = np.abs(ang_vel)

        state = np.zeros(len(ang), dtype=np.float64)
        state[ang_vel > 0.5]  =  1.0
        state[ang_vel < -0.5] = -1.0
        feats[f"{prefix}_state"] = state

        ang_clean = np.nan_to_num(ang,
            nan=float(np.nanmedian(ang)) if np.any(np.isfinite(ang)) else 0.0)
        feats[f"{prefix}_angle_rel"] = ang_clean - float(np.nanmedian(ang_clean))

        # Gripper also follows a bell-shaped velocity profile per open/close cycle
        # log-angular velocity: linearises the log-normal distribution
        feats[f"{prefix}_ang_vel_log"] = np.log1p(np.abs(ang_vel))

    _add_pince("pince1", aligned_pince1)
    _add_pince("pince2", aligned_pince2)

    # Global aggregates
    speed_cols = [c for c in feats.columns if c.endswith("_speed")]
    acc_cols   = [c for c in feats.columns if c.endswith("_acc")]
    jerk_cols  = [c for c in feats.columns if c.endswith("_jerk")]
    if speed_cols:
        feats["motion_speed_sum"] = feats[speed_cols].sum(axis=1)
    if acc_cols:
        feats["motion_acc_sum"]   = feats[acc_cols].sum(axis=1)
    if jerk_cols:
        feats["motion_jerk_sum"]  = feats[jerk_cols].sum(axis=1)

    return feats


# ---------------------------------------------------------------------------
# Video features
# ---------------------------------------------------------------------------

def build_video_features(
    video_path: str,
    timeline: "Timeline",
    *,
    resize: tuple[int, int] | None = (160, 90),
) -> pd.DataFrame:
    try:
        import cv2
    except ImportError as e:
        raise ImportError("pip install opencv-python") from e
    from pathlib import Path
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS: {video_fps}")
    frame_indices = np.clip(np.round(timeline.t * video_fps).astype(np.int64), 0, total_frames - 1)
    num = len(timeline.t)
    brightness = np.full(num, np.nan)
    blur       = np.full(num, np.nan)
    frame_diff = np.full(num, np.nan)
    prev_gray: np.ndarray | None = None
    cur = -1
    for out_idx, vid_idx in enumerate(frame_indices):
        if vid_idx != cur + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(vid_idx))
        ret, frame = cap.read()
        if not ret:
            prev_gray = None
            cur = vid_idx
            continue
        cur = vid_idx
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        brightness[out_idx] = float(gray.mean())
        blur[out_idx]       = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if prev_gray is not None:
            frame_diff[out_idx] = float(np.mean(np.abs(gray - prev_gray)))
        prev_gray = gray
    cap.release()
    return pd.DataFrame({
        "time_seconds":     timeline.t,
        "video_brightness": brightness,
        "video_blur":       blur,
        "video_frame_diff": frame_diff,
    })


# ---------------------------------------------------------------------------
# Segment-level feature extraction
# ---------------------------------------------------------------------------

def _seg_cols(
    frame_features: pd.DataFrame,
    exclude_cols: tuple[str, ...],
    exclude_raw_pos: bool,
    exclude_quats: bool,
) -> list[str]:
    def _skip(col: str) -> bool:
        if col in exclude_cols:
            return True
        if exclude_raw_pos and any(col.endswith(p) for p in ("_x", "_y", "_z")):
            return True
        if exclude_quats and any(col.endswith(p) for p in ("_qw", "_qx", "_qy", "_qz")):
            return True
        return False
    return [c for c in frame_features.columns if not _skip(c)]


def segment_level_features(
    frame_features: pd.DataFrame,
    segments: list[dict],
    *,
    fps: float = 30.0,
    exclude_cols: tuple[str, ...] = ("time_seconds",),
    exclude_raw_pos: bool = True,
    exclude_quats: bool = True,
) -> np.ndarray:
    """Build a rich descriptor vector per segment.

    Two layers of features:

    Layer 1 — Per-column statistics (frame → scalar, 9 stats each):
      mean, std, min, max — classical
      onset, offset       — mean of first/last 25% frames (trend direction)
      direction           — onset − offset
      slope               — linear regression coefficient on [0,1] time axis
      peak_pos            — when does the max occur (0=start, 1=end)

    Layer 2 — Motor-control shape descriptors (one per segment, 14 values):
      Computed over the primary speed signal (motion_speed_sum) and gripper:

      mj_residual         — minimum-jerk residual [0,2]; 0=perfect bell
      speed_skew          — skewness of speed profile; ~0=symmetric, <0=decelerative
      speed_log_kurtosis  — excess kurtosis of log(speed); high=sharp ballistic peak
      power_law_resid     — 2/3 power law residual; low=smooth voluntary curved motion
      njs                 — normalised jerk score; low=smooth, high=jerky/corrective
      speed_peak_t        — normalised time of peak speed [0,1]
      n_submovements      — number of velocity sub-peaks (1=simple, 2+=composite)
      grip_mj_residual    — same as mj_residual but for gripper angular velocity
      grip_speed_skew     — skewness of gripper velocity profile
      grip_peak_t         — normalised time of peak gripper velocity
      grip_n_submovements — sub-peaks in gripper velocity
      grip_max_angle      — peak opening angle (degrees)
      grip_net_change     — net angle offset−onset (+ = net open, − = net close)
      grip_frac_opening   — fraction of frames where gripper is opening

    These shape descriptors are *duration-invariant*: two identical actions
    at different speeds will have similar mj_residual, skewness, kurtosis,
    and peak_t, enabling the clustering to correctly group them.
    """
    cols   = _seg_cols(frame_features, exclude_cols, exclude_raw_pos, exclude_quats)
    n_cols = len(cols)
    X      = frame_features[cols].to_numpy(dtype=np.float64)

    def _get(name: str) -> np.ndarray | None:
        if name in frame_features.columns:
            return frame_features[name].to_numpy(dtype=np.float64)
        return None

    n_stats    = 9
    n_shape    = 14
    total_dims = n_cols * n_stats + n_shape
    out        = []

    speed_arr  = _get("motion_speed_sum")
    ang_arr    = _get("pince1_angle_deg")
    ang_vel_arr = _get("pince1_ang_vel_abs")
    state_arr  = _get("pince1_state")
    curv_arr   = None
    # Use tracker_1 curvature if available for power law
    for ti in [1, 2, 3]:
        c = _get(f"tracker_{ti}_curvature")
        if c is not None:
            curv_arr = c
            break

    for s in segments:
        a = max(int(s["start_idx"]), 0)
        b = min(int(s["end_idx"]), len(X) - 1)
        seg_len = b - a + 1

        if seg_len < 2:
            out.append(np.zeros(total_dims, dtype=np.float64))
            continue

        seg      = X[a : b + 1]
        duration = float(s.get("duration_s", seg_len / fps))

        # ── Layer 1: per-column statistics ──────────────────────────────
        m  = np.nanmean(seg, axis=0)
        sd = np.nanstd(seg, axis=0)
        mn = np.nanmin(seg, axis=0)
        mx = np.nanmax(seg, axis=0)

        q      = max(1, seg_len // 4)
        onset  = np.nanmean(seg[:q],  axis=0)
        offset = np.nanmean(seg[-q:], axis=0)
        direction = onset - offset

        t_n  = np.linspace(0., 1., seg_len)
        t_c  = t_n - t_n.mean()
        t_ss = (t_c ** 2).sum()
        if t_ss > 1e-12:
            y_c   = np.nan_to_num(seg - np.nanmean(seg, axis=0, keepdims=True), nan=0.)
            slope = (t_c[:, None] * y_c).sum(axis=0) / t_ss
        else:
            slope = np.zeros(n_cols, dtype=np.float64)

        peak_pos = np.argmax(np.nan_to_num(seg, nan=-np.inf), axis=0).astype(np.float64)
        peak_pos /= max(seg_len - 1, 1)

        layer1 = np.concatenate([m, sd, mn, mx, onset, offset, direction, slope, peak_pos])

        # ── Layer 2: motor-control shape descriptors ─────────────────────
        shape = np.zeros(n_shape, dtype=np.float64)

        # — Motion kinematic features —
        if speed_arr is not None:
            sp = speed_arr[a : b + 1]
            sp_clean = np.nan_to_num(sp, nan=0.0)
            amplitude = sp_clean.max() - sp_clean.min() + 1e-9

            shape[0] = _minimum_jerk_residual(sp_clean)
            shape[1] = _speed_skewness(sp_clean)
            shape[2] = _log_speed_kurtosis(sp_clean)
            shape[4] = _normalised_jerk_score(sp_clean, duration, amplitude)
            shape[5] = _speed_peak_time(sp_clean)
            shape[6] = float(_submovements_helper(sp_clean, fps))

        # — Power law (needs curvature) —
        if speed_arr is not None and curv_arr is not None:
            shape[3] = _power_law_residual(
                np.nan_to_num(speed_arr[a : b + 1], nan=0.0),
                np.nan_to_num(curv_arr[a : b + 1],  nan=0.0),
            )

        # — Gripper kinematic features —
        if ang_vel_arr is not None:
            gv = ang_vel_arr[a : b + 1]
            gv_clean = np.nan_to_num(gv, nan=0.0)
            shape[7]  = _minimum_jerk_residual(gv_clean)
            shape[8]  = _speed_skewness(gv_clean)
            shape[9]  = _speed_peak_time(gv_clean)
            shape[10] = float(_submovements_helper(gv_clean, fps))

        if ang_arr is not None:
            sa = ang_arr[a : b + 1]
            valid = np.isfinite(sa)
            if valid.sum() > 0:
                shape[11] = float(np.nanmax(sa))
                qa = max(1, seg_len // 4)
                shape[12] = float(np.nanmean(sa[-qa:]) - np.nanmean(sa[:qa]))

        if state_arr is not None:
            ss = state_arr[a : b + 1]
            valid = np.isfinite(ss)
            if valid.sum() > 0:
                shape[13] = float(np.nanmean(ss > 0))

        out.append(np.concatenate([layer1, shape]))

    return np.vstack(out) if out else np.empty((0, total_dims))


def _submovements_helper(speed: np.ndarray, fps: float) -> int:
    """Count velocity sub-peaks (submovements) in a speed profile."""
    sp = np.nan_to_num(speed, nan=0.0)
    if len(sp) < 3:
        return 1
    k = max(1, int(round(0.05 * fps)))
    sp_sm = _smooth_1d(sp, k)
    threshold = sp_sm.max() * 0.20
    peaks = 0
    for i in range(1, len(sp_sm) - 1):
        if sp_sm[i] > sp_sm[i-1] and sp_sm[i] > sp_sm[i+1] and sp_sm[i] > threshold:
            peaks += 1
    return max(1, peaks)


def segment_feature_names(
    frame_features: pd.DataFrame,
    *,
    exclude_cols: tuple[str, ...] = ("time_seconds",),
    exclude_raw_pos: bool = True,
    exclude_quats: bool = True,
) -> list[str]:
    cols  = _seg_cols(frame_features, exclude_cols, exclude_raw_pos, exclude_quats)
    stats = ["mean", "std", "min", "max", "onset", "offset", "direction", "slope", "peak_pos"]
    names = [f"{c}__{s}" for s in stats for c in cols]
    names += [
        # Motor-control shape descriptors
        "shape__mj_residual",        # minimum-jerk residual (motion)
        "shape__speed_skew",         # skewness of speed profile
        "shape__speed_log_kurtosis", # kurtosis of log(speed)
        "shape__powerlaw_resid",     # 2/3 power law residual
        "shape__njs",                # normalised jerk score
        "shape__speed_peak_t",       # time of peak speed
        "shape__n_submovements",     # number of velocity sub-peaks
        "shape__grip_mj_residual",   # minimum-jerk residual (gripper)
        "shape__grip_speed_skew",    # skewness of gripper velocity
        "shape__grip_peak_t",        # time of peak gripper velocity
        "shape__grip_n_submovements",# gripper sub-peaks
        "shape__grip_max_angle",     # peak opening angle
        "shape__grip_net_change",    # net angle change (+ open, - close)
        "shape__grip_frac_opening",  # fraction frames opening
    ]
    return names
