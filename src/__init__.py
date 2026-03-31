"""VIVE Labeler - Advanced multimodal data labeling tool."""

__version__ = "0.1.0"
__author__ = "VIVE Labeler Team"
__license__ = "MIT"

from .core.dataset import ViVEDataset
from .core.data_loader import LeRobotDataLoader
from .labeling.label_manager import LabelManager, Label, Annotation
from .labeling.export import AnnotationExporter

__all__ = [
    "ViVEDataset",
    "LeRobotDataLoader",
    "LabelManager",
    "Label",
    "Annotation",
    "AnnotationExporter",
]
