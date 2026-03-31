import argparse

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
    cluster_segments_kmeans,
    export_segments_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", required=True)
    ap.add_argument("--pince1", required=True)
    ap.add_argument("--pince2", default=None)
    ap.add_argument("--video", action="append", default=None, metavar="PATH",
                    help="Video file (mp4, avi, …). Repeat for multiple cameras: --video cam0.avi --video cam1.avi")
    ap.add_argument("--video_offset", type=float, default=0.0, help="Time offset in seconds to align video clock to sensor clock")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--n_labels", type=int, default=8)
    ap.add_argument("--out", default="segments_bootstrap.csv")
    args = ap.parse_args()

    trackers = load_trackers_csv(args.trackers)
    p1 = load_pince_csv(args.pince1)
    p2 = load_pince_csv(args.pince2) if args.pince2 else None
    video_paths = args.video or []
    video_metas = [load_video(v, t_offset=args.video_offset) for v in video_paths]

    streams = [df for df in [trackers, p1, p2, *video_metas] if df is not None]
    timeline = build_timeline(*streams, fps=args.fps)

    tr_al = align_to_timeline(trackers, timeline, columns=[c for c in trackers.columns if c != "timestamp"])
    p1_al = align_to_timeline(p1, timeline)
    p2_al = align_to_timeline(p2, timeline) if p2 is not None else None

    feats = build_sensor_features(tr_al, aligned_pince1=p1_al, aligned_pince2=p2_al, fps=timeline.fps)

    for cam_idx, vid_path in enumerate(video_paths):
        vid_feats = build_video_features(vid_path, timeline)
        prefix = f"cam{cam_idx}"
        for col in ["video_brightness", "video_blur", "video_frame_diff"]:
            feats[f"{prefix}_{col}"] = vid_feats[col].to_numpy()

    segs = heuristic_segments(feats, fps=timeline.fps)
    X = segment_level_features(feats, segs)
    segs = cluster_segments_kmeans(X, segs, n_labels=args.n_labels)

    export_segments_csv(segs, args.out)


if __name__ == "__main__":
    main()
