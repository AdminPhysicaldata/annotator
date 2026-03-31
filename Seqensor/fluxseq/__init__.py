"""fluxseq

Minimal library to synchronize multimodal streams (Vive trackers + grippers + videos)
and bootstrap temporal segmentation into action labels.

Core entrypoints:
- fluxseq.io.load_trackers_csv
- fluxseq.io.load_pince_csv
- fluxseq.sync.build_timeline
- fluxseq.features.build_sensor_features
- fluxseq.segment.heuristic_segments
- fluxseq.segment.cluster_segments_gmm        ← recommended
- fluxseq.segment.cluster_segments_ensemble   ← auto-selects best method
- fluxseq.segment.cluster_segments_hierarchical
- fluxseq.segment.cluster_segments_kmeans     ← legacy

"""

from .io import load_trackers_csv, load_pince_csv, load_video
from .sync import build_timeline, align_to_timeline
from .features import (
    build_sensor_features,
    build_video_features,
    segment_level_features,
    segment_feature_names,
)
from .segment import (
    heuristic_segments,
    cluster_segments_gmm,
    cluster_segments_ensemble,
    cluster_segments_hierarchical,
    cluster_segments_kmeans,
    export_segments_csv,
)

__all__ = [
    "load_trackers_csv",
    "load_pince_csv",
    "load_video",
    "build_timeline",
    "align_to_timeline",
    "build_sensor_features",
    "build_video_features",
    "segment_level_features",
    "segment_feature_names",
    "heuristic_segments",
    "cluster_segments_gmm",
    "cluster_segments_ensemble",
    "cluster_segments_hierarchical",
    "cluster_segments_kmeans",
    "export_segments_csv",
]
