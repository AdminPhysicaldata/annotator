"""fluxseq

Minimal library to synchronize multimodal streams (Vive trackers + grippers + videos)
and bootstrap temporal segmentation into X labels.

Core entrypoints:
- fluxseq.io.load_trackers_csv
- fluxseq.io.load_pince_csv
- fluxseq.sync.build_timeline
- fluxseq.features.build_sensor_features
- fluxseq.segment.heuristic_segments
- fluxseq.segment.cluster_segments_kmeans

"""

from .io import load_trackers_csv, load_pince_csv, load_video
from .sync import build_timeline, align_to_timeline
from .features import build_sensor_features, build_video_features, segment_level_features
from .segment import heuristic_segments, cluster_segments_kmeans, export_segments_csv

__all__ = [
    "load_trackers_csv",
    "load_pince_csv",
    "load_video",
    "build_timeline",
    "align_to_timeline",
    "build_sensor_features",
    "build_video_features",
    "segment_level_features",
    "heuristic_segments",
    "cluster_segments_kmeans",
    "export_segments_csv",
]
