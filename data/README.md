# fluxseq

Library to synchronize Vive trackers + grippers (pinces) and bootstrap frame-level temporal segmentation.

## Install

From the folder containing this README:

```bash
pip install -e .
```

Or add the folder to `PYTHONPATH`.

## Quickstart

```python
from fluxseq import (
    load_trackers_csv,
    load_pince_csv,
    build_timeline,
    align_to_timeline,
    build_sensor_features,
    heuristic_segments,
    segment_level_features,
    cluster_segments_kmeans,
    export_segments_csv,
)

trackers = load_trackers_csv("tracker_positions.csv")
p1 = load_pince_csv("pince1_data.csv")
# p2 = load_pince_csv("pince2_data.csv")

# 1) timeline on overlap
timeline = build_timeline(trackers, p1, fps=30)

# 2) align sensors
tr_al = align_to_timeline(trackers, timeline, columns=[c for c in trackers.columns if c != "timestamp"])  # keep numeric
p1_al = align_to_timeline(p1, timeline)

# 3) features
feats = build_sensor_features(tr_al, aligned_pince1=p1_al, fps=timeline.fps)

# 4) heuristic segmentation
segs = heuristic_segments(feats, fps=timeline.fps)

# 5) assign X labels by clustering
Xseg = segment_level_features(feats, segs)
segs = cluster_segments_kmeans(Xseg, segs, n_labels=8)

# 6) export for manual correction
export_segments_csv(segs, "segments_bootstrap.csv")
```

The output CSV contains start/end indices + timestamps and a provisional `label`.

## Video

This library does not decode AVI by default. Use your own frame reader (OpenCV) and keep a timeline.
You can fuse video embeddings later by adding columns to the same `feats` dataframe.
