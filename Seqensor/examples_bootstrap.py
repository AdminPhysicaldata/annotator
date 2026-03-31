"""Bootstrap pipeline — full example with the upgraded fluxseq stack.

Usage
-----
    python examples_bootstrap.py \
        --trackers session/tracker_positions.csv \
        --pince1   session/pince1_data.csv \
        [--pince2  session/pince2_data.csv] \
        [--video   session/cam0/output.avi] \
        [--video   session/cam1/output.avi] \
        [--fps 30] \
        [--n_labels auto] \
        [--n_labels_min 2] \
        [--n_labels_max 12] \
        [--method ensemble] \
        [--use_gap] \
        [--out segments_bootstrap.csv]

Key improvements
----------------
Features:
  - Directional velocity per tracker (vx, vy, vz)
  - Angular velocity from quaternions (vectorised)
  - Trajectory curvature (vectorised)
  - Signed gripper delta + state enum (opening/closing/static)
  - Inter-tracker distance velocity
  - Motion jerk

Segment descriptors:
  - onset / offset means, direction, slope, peak position
  - 9 statistics per feature (was 4)

Segmentation:
  - Hysteresis thresholding (rising ≠ falling edge)
  - Jerk as third activation source

Clustering (automatic k):
  - Adaptive PCA with whitening (variance threshold)
  - k selected by composite score: silhouette + Calinski-Harabasz + Davies-Bouldin
  - Optional Gap statistic (--use_gap)
  - Ensemble: GMM vs Ward hierarchical, keeps best
"""

import argparse
import sys

from fluxseq import (
    load_trackers_csv,
    load_pince_csv,
    load_video,
    build_timeline,
    align_to_timeline,
    build_sensor_features,
    build_video_features,
    heuristic_segments,
    segment_level_features,
    segment_feature_names,
    cluster_segments_gmm,
    cluster_segments_ensemble,
    cluster_segments_kmeans,
    export_segments_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bootstrap temporal segmentation of multimodal sensor data."
    )
    ap.add_argument("--trackers", required=True, help="Path to tracker_positions.csv")
    ap.add_argument("--pince1",   required=True, help="Path to pince1_data.csv")
    ap.add_argument("--pince2",   default=None,  help="Path to pince2_data.csv (optional)")
    ap.add_argument(
        "--video", action="append", default=None, metavar="PATH",
        help="Video file. Repeat for multiple cameras.",
    )
    ap.add_argument(
        "--video_offset", type=float, default=0.0,
        help="Time offset (s) to align video clock to sensor clock",
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--n_labels", default="auto",
        help="Number of action classes. 'auto' = automatic selection (recommended). "
             "Integer = fixed.",
    )
    ap.add_argument("--n_labels_min", type=int, default=2,
                    help="Min clusters for auto selection (default: 2)")
    ap.add_argument("--n_labels_max", type=int, default=12,
                    help="Max clusters for auto selection (default: 12)")
    ap.add_argument(
        "--method", choices=["ensemble", "gmm", "kmeans"], default="ensemble",
        help="Clustering method (default: ensemble)",
    )
    ap.add_argument(
        "--use_gap", action="store_true", default=False,
        help="Also use Gap statistic during k selection (slower, more reliable on small datasets)",
    )
    ap.add_argument("--out", default="segments_bootstrap.csv")
    args = ap.parse_args()

    # Parse n_labels
    if args.n_labels == "auto":
        n_labels = None
    else:
        try:
            n_labels = int(args.n_labels)
        except ValueError:
            print(f"[ERROR] --n_labels must be 'auto' or an integer, got: {args.n_labels!r}")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("[1/5] Loading data…")
    trackers    = load_trackers_csv(args.trackers)
    p1          = load_pince_csv(args.pince1)
    p2          = load_pince_csv(args.pince2) if args.pince2 else None
    video_paths = args.video or []
    video_metas = [load_video(v, t_offset=args.video_offset) for v in video_paths]

    print(f"      Trackers : {len(trackers)} rows")
    print(f"      Pince 1  : {len(p1)} rows")
    if p2 is not None:
        print(f"      Pince 2  : {len(p2)} rows")
    for i, vm in enumerate(video_metas):
        print(f"      Video {i}  : {len(vm)} frames")

    # -------------------------------------------------------------------------
    # 2. Build timeline and align streams
    # -------------------------------------------------------------------------
    print("[2/5] Building timeline and aligning streams…")
    streams  = [df for df in [trackers, p1, p2, *video_metas] if df is not None]
    timeline = build_timeline(*streams, fps=args.fps)
    print(f"      Timeline : {len(timeline.t)} frames @ {timeline.fps} Hz "
          f"({timeline.t[0]:.3f}s – {timeline.t[-1]:.3f}s)")

    tr_al = align_to_timeline(trackers, timeline,
                               columns=[c for c in trackers.columns if c != "timestamp"])
    p1_al = align_to_timeline(p1, timeline)
    p2_al = align_to_timeline(p2, timeline) if p2 is not None else None

    # -------------------------------------------------------------------------
    # 3. Extract features
    # -------------------------------------------------------------------------
    print("[3/5] Extracting features…")
    feats = build_sensor_features(
        tr_al, aligned_pince1=p1_al, aligned_pince2=p2_al,
        fps=timeline.fps, include_quat=True, smooth_speed_ms=60,
    )
    for cam_idx, vid_path in enumerate(video_paths):
        vid_feats = build_video_features(vid_path, timeline)
        for col in ["video_brightness", "video_blur", "video_frame_diff"]:
            feats[f"cam{cam_idx}_{col}"] = vid_feats[col].to_numpy()

    print(f"      Frame features : {len(feats.columns) - 1} dims × {len(feats)} frames")

    # -------------------------------------------------------------------------
    # 4. Segmentation
    # -------------------------------------------------------------------------
    print("[4/5] Segmenting actions…")
    segs = heuristic_segments(feats, fps=timeline.fps)
    print(f"      Found {len(segs)} segments")
    if not segs:
        print("[WARN] No segments found. Check data quality or lower thresholds.")
        export_segments_csv([], args.out)
        return

    # -------------------------------------------------------------------------
    # 5. Segment features + clustering
    # -------------------------------------------------------------------------
    print("[5/5] Building segment descriptors and clustering…")
    X         = segment_level_features(feats, segs, fps=timeline.fps)
    feat_names = segment_feature_names(feats)
    print(f"      Segment descriptor : {X.shape[1]} dims × {X.shape[0]} segments")

    common_kw = dict(
        n_labels=n_labels,
        n_labels_max=args.n_labels_max,
        n_labels_min=args.n_labels_min,
        use_gap=args.use_gap,
    )

    if args.method == "ensemble":
        segs, winner, diag = cluster_segments_ensemble(X, segs, **common_kw)
        print(f"      Winner : {winner}")
        print(f"      GMM  → k={diag.get('gmm_k','?')}  score={diag.get('gmm_score','?')}")
        print(f"      Ward → k={diag.get('hier_k','?')}  score={diag.get('hier_score','?')}")
        print(f"      PCA dims kept : {diag.get('pca_dims','?')}")
    elif args.method == "gmm":
        segs = cluster_segments_gmm(X, segs, **common_kw)
        print("      Method : GMM")
    else:
        k = n_labels if n_labels is not None else 8
        segs = cluster_segments_kmeans(X, segs, n_labels=k)
        print(f"      Method : K-means (k={k})")

    # Summary
    labels        = [s["label"] for s in segs]
    unique_labels = sorted(set(labels))
    print(f"\n      Clusters found : {len(unique_labels)}")
    for lbl in unique_labels:
        members   = [s for s in segs if s["label"] == lbl]
        durations = [s["duration_s"] for s in members]
        print(
            f"        Label {lbl:2d} : {len(members):3d} segments, "
            f"dur {min(durations):.2f}–{max(durations):.2f}s "
            f"(mean {sum(durations)/len(durations):.2f}s)"
        )

    export_segments_csv(segs, args.out)
    print(f"\n[DONE] {len(segs)} segments → {args.out}")


if __name__ == "__main__":
    main()
