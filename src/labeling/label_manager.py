"""Label management system for VIVE Labeler."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set, Any
from enum import Enum
import json
from pathlib import Path
import uuid


# Special label ID for unlabeled segments (covers the whole video initially)
UNLABELED_LABEL_ID = "__unlabeled__"
UNLABELED_LABEL_NAME = "Non labellisé"
UNLABELED_LABEL_COLOR = "#3d3d5c"

# Special label ID for idle segments (unlabeled segments that have been finalized)
IDLE_LABEL_ID = "__idle__"
IDLE_LABEL_NAME = "Idle"
IDLE_LABEL_COLOR = "#45475a"


class LabelType(Enum):
    """Type of label annotation."""
    FRAME = "frame"  # Single frame label
    INTERVAL = "interval"  # Time interval label


@dataclass
class Label:
    """Individual label definition."""
    id: str
    name: str
    color: str = "#FFFFFF"
    description: str = ""
    shortcut: Optional[str] = None  # Keyboard shortcut

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Label":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Annotation:
    """Individual annotation instance."""
    id: str
    label_id: str
    label_name: str
    annotation_type: LabelType
    frame_index: Optional[int] = None  # For frame annotations
    start_frame: Optional[int] = None  # For interval annotations
    end_frame: Optional[int] = None  # For interval annotations
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None  # ISO format timestamp

    def __post_init__(self):
        if isinstance(self.annotation_type, str):
            self.annotation_type = LabelType(self.annotation_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['annotation_type'] = self.annotation_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """Create from dictionary."""
        return cls(**data)


class LabelManager:
    """Manages labels and annotations for the dataset."""

    def __init__(self):
        """Initialize label manager."""
        self.labels: Dict[str, Label] = {}
        self.annotations: List[Annotation] = []

        # Index for quick lookup
        self._annotations_by_frame: Dict[int, List[Annotation]] = {}
        self._annotations_by_label: Dict[str, List[Annotation]] = {}

    def add_label(
        self,
        name: str,
        color: str = "#FFFFFF",
        description: str = "",
        shortcut: Optional[str] = None
    ) -> Label:
        """Add a new label definition.

        Args:
            name: Label name
            color: Color in hex format
            description: Label description
            shortcut: Keyboard shortcut

        Returns:
            Created Label object
        """
        label_id = str(uuid.uuid4())
        label = Label(
            id=label_id,
            name=name,
            color=color,
            description=description,
            shortcut=shortcut
        )
        self.labels[label_id] = label
        self._annotations_by_label[label_id] = []
        return label

    def remove_label(self, label_id: str) -> None:
        """Remove a label and all associated annotations.

        Args:
            label_id: ID of label to remove
        """
        if label_id not in self.labels:
            raise KeyError(f"Label {label_id} not found")

        # Remove all annotations with this label
        annotations_to_remove = self._annotations_by_label.get(label_id, [])
        for ann in annotations_to_remove:
            self.remove_annotation(ann.id)

        # Remove label
        del self.labels[label_id]
        if label_id in self._annotations_by_label:
            del self._annotations_by_label[label_id]

    def get_label_by_name(self, name: str) -> Optional[Label]:
        """Get label by name.

        Args:
            name: Label name

        Returns:
            Label object or None if not found
        """
        for label in self.labels.values():
            if label.name == name:
                return label
        return None

    def add_frame_annotation(
        self,
        frame_index: int,
        label_id: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Annotation:
        """Add a frame annotation.

        Args:
            frame_index: Frame index to annotate
            label_id: ID of label to apply
            confidence: Confidence score
            metadata: Additional metadata

        Returns:
            Created Annotation object
        """
        if label_id not in self.labels:
            raise KeyError(f"Label {label_id} not found")

        label = self.labels[label_id]
        annotation = Annotation(
            id=str(uuid.uuid4()),
            label_id=label_id,
            label_name=label.name,
            annotation_type=LabelType.FRAME,
            frame_index=frame_index,
            confidence=confidence,
            metadata=metadata or {}
        )

        self._add_annotation(annotation)
        return annotation

    def add_interval_annotation(
        self,
        start_frame: int,
        end_frame: int,
        label_id: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Annotation:
        """Add an interval annotation.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index
            label_id: ID of label to apply
            confidence: Confidence score
            metadata: Additional metadata

        Returns:
            Created Annotation object
        """
        if label_id not in self.labels:
            raise KeyError(f"Label {label_id} not found")

        if start_frame > end_frame:
            raise ValueError("start_frame must be <= end_frame")

        label = self.labels[label_id]
        annotation = Annotation(
            id=str(uuid.uuid4()),
            label_id=label_id,
            label_name=label.name,
            annotation_type=LabelType.INTERVAL,
            start_frame=start_frame,
            end_frame=end_frame,
            confidence=confidence,
            metadata=metadata or {}
        )

        self._add_annotation(annotation)
        return annotation

    # Intervals spanning more than this many frames are not indexed frame-by-frame
    # to avoid O(N) memory/time cost.  get_annotations_at_frame() falls back to a
    # linear scan for those.
    _FRAME_INDEX_MAX_SPAN = 10_000

    def _add_annotation(self, annotation: Annotation) -> None:
        """Internal method to add annotation and update indices."""
        self.annotations.append(annotation)

        # Update frame index
        if annotation.annotation_type == LabelType.FRAME:
            frame_idx = annotation.frame_index
            if frame_idx not in self._annotations_by_frame:
                self._annotations_by_frame[frame_idx] = []
            self._annotations_by_frame[frame_idx].append(annotation)
        elif annotation.annotation_type == LabelType.INTERVAL:
            span = annotation.end_frame - annotation.start_frame + 1
            if span <= self._FRAME_INDEX_MAX_SPAN:
                for frame_idx in range(annotation.start_frame, annotation.end_frame + 1):
                    if frame_idx not in self._annotations_by_frame:
                        self._annotations_by_frame[frame_idx] = []
                    self._annotations_by_frame[frame_idx].append(annotation)
            # else: too wide — will be found by linear scan in get_annotations_at_frame

        # Update label index
        if annotation.label_id not in self._annotations_by_label:
            self._annotations_by_label[annotation.label_id] = []
        self._annotations_by_label[annotation.label_id].append(annotation)

    def remove_annotation(self, annotation_id: str) -> None:
        """Remove an annotation.

        Args:
            annotation_id: ID of annotation to remove
        """
        # Find annotation
        annotation = None
        for ann in self.annotations:
            if ann.id == annotation_id:
                annotation = ann
                break

        if annotation is None:
            raise KeyError(f"Annotation {annotation_id} not found")

        # Remove from main list
        self.annotations.remove(annotation)

        # Remove from frame index
        if annotation.annotation_type == LabelType.FRAME:
            frame_idx = annotation.frame_index
            if frame_idx in self._annotations_by_frame:
                self._annotations_by_frame[frame_idx].remove(annotation)
        elif annotation.annotation_type == LabelType.INTERVAL:
            span = annotation.end_frame - annotation.start_frame + 1
            if span <= self._FRAME_INDEX_MAX_SPAN:
                for frame_idx in range(annotation.start_frame, annotation.end_frame + 1):
                    if frame_idx in self._annotations_by_frame:
                        if annotation in self._annotations_by_frame[frame_idx]:
                            self._annotations_by_frame[frame_idx].remove(annotation)
            # else: was never in the frame index, nothing to remove

        # Remove from label index
        if annotation.label_id in self._annotations_by_label:
            if annotation in self._annotations_by_label[annotation.label_id]:
                self._annotations_by_label[annotation.label_id].remove(annotation)

    def get_annotations_at_frame(self, frame_index: int) -> List[Annotation]:
        """Get all annotations active at a specific frame.

        Args:
            frame_index: Frame index

        Returns:
            List of Annotation objects
        """
        result = list(self._annotations_by_frame.get(frame_index, []))
        # Also include wide intervals that were not indexed frame-by-frame
        for ann in self.annotations:
            if ann.annotation_type == LabelType.INTERVAL:
                span = ann.end_frame - ann.start_frame + 1
                if span > self._FRAME_INDEX_MAX_SPAN:
                    if ann.start_frame <= frame_index <= ann.end_frame:
                        result.append(ann)
        return result

    def get_annotations_by_label(self, label_id: str) -> List[Annotation]:
        """Get all annotations for a specific label.

        Args:
            label_id: Label ID

        Returns:
            List of Annotation objects
        """
        return self._annotations_by_label.get(label_id, [])

    def get_labels_at_frame(self, frame_index: int) -> Set[str]:
        """Get set of label names active at a frame.

        Args:
            frame_index: Frame index

        Returns:
            Set of label names
        """
        annotations = self.get_annotations_at_frame(frame_index)
        return {ann.label_name for ann in annotations}

    def clear_annotations(self) -> None:
        """Remove all annotations (keep labels)."""
        self.annotations.clear()
        self._annotations_by_frame.clear()
        for label_id in self._annotations_by_label:
            self._annotations_by_label[label_id] = []

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary.

        Returns:
            Dictionary with labels and annotations
        """
        return {
            'labels': [label.to_dict() for label in self.labels.values()],
            'annotations': [ann.to_dict() for ann in self.annotations],
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Import from dictionary.

        Args:
            data: Dictionary with labels and annotations
        """
        # Clear existing data
        self.labels.clear()
        self.clear_annotations()

        # Load labels
        for label_data in data.get('labels', []):
            label = Label.from_dict(label_data)
            self.labels[label.id] = label
            self._annotations_by_label[label.id] = []

        # Load annotations
        for ann_data in data.get('annotations', []):
            annotation = Annotation.from_dict(ann_data)
            self._add_annotation(annotation)

    def save_to_file(self, file_path: Path) -> None:
        """Save to JSON file.

        Args:
            file_path: Path to output file
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_file(self, file_path: Path) -> None:
        """Load from JSON file.

        Args:
            file_path: Path to input file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.from_dict(data)

    def initialize_full_segment(self, frame_count: int, hand: str = "right") -> None:
        """Create a single unlabeled segment covering the entire video.

        Should be called after loading a session and clearing annotations.
        Creates the special __unlabeled__ label if it doesn't exist yet.
        """
        # Ensure the unlabeled label exists
        if UNLABELED_LABEL_ID not in self.labels:
            label = Label(
                id=UNLABELED_LABEL_ID,
                name=UNLABELED_LABEL_NAME,
                color=UNLABELED_LABEL_COLOR,
            )
            self.labels[UNLABELED_LABEL_ID] = label
            self._annotations_by_label[UNLABELED_LABEL_ID] = []

        if frame_count <= 0:
            return

        annotation = Annotation(
            id=str(uuid.uuid4()),
            label_id=UNLABELED_LABEL_ID,
            label_name=UNLABELED_LABEL_NAME,
            annotation_type=LabelType.INTERVAL,
            start_frame=0,
            end_frame=frame_count - 1,
            metadata={"hand": hand, "fail": False, "unlabeled": True},
        )
        self._add_annotation(annotation)

    def get_segment_at_frame(self, frame: int, hand: Optional[str] = None) -> Optional["Annotation"]:
        """Return the interval annotation that covers *frame* for a given hand.

        If *hand* is None, returns the first matching interval regardless of hand.
        Unlabeled segments are included.
        """
        for ann in self.annotations:
            if ann.annotation_type != LabelType.INTERVAL:
                continue
            if hand is not None:
                ann_hand = ann.metadata.get("hand")
                if ann_hand != hand:
                    continue
            if ann.start_frame <= frame <= ann.end_frame:
                return ann
        return None

    def split_annotation_at_frame(self, annotation_id: str, frame: int) -> Optional["Annotation"]:
        """Split an interval annotation at *frame*, creating two new segments.

        The original annotation is split into:
        - Left part : [original.start_frame, frame - 1]
        - Right part: [frame, original.end_frame]

        Both parts inherit the same label, metadata (unlabeled/labeled).
        Returns the right-part annotation (or None if the split is invalid).
        """
        # Find annotation
        annotation = None
        for ann in self.annotations:
            if ann.id == annotation_id:
                annotation = ann
                break

        if annotation is None or annotation.annotation_type != LabelType.INTERVAL:
            return None

        # Cannot split at the very start (no left part)
        if frame <= annotation.start_frame or frame > annotation.end_frame:
            return None

        original_end = annotation.end_frame
        original_metadata = dict(annotation.metadata)

        # Shorten the existing annotation to the left part
        # We need to rebuild the frame index for the affected range
        # Remove old annotation and re-add modified version
        self.remove_annotation(annotation.id)

        left = Annotation(
            id=str(uuid.uuid4()),
            label_id=annotation.label_id,
            label_name=annotation.label_name,
            annotation_type=LabelType.INTERVAL,
            start_frame=annotation.start_frame,
            end_frame=frame - 1,
            confidence=annotation.confidence,
            metadata=dict(original_metadata),
        )
        self._add_annotation(left)

        right = Annotation(
            id=str(uuid.uuid4()),
            label_id=annotation.label_id,
            label_name=annotation.label_name,
            annotation_type=LabelType.INTERVAL,
            start_frame=frame,
            end_frame=original_end,
            confidence=annotation.confidence,
            metadata=dict(original_metadata),
        )
        self._add_annotation(right)

        return right

    def assign_label_to_annotation(
        self,
        annotation_id: str,
        label_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reassign the label of an existing annotation (e.g. unlabeled → real label).

        Removes the annotation and re-adds it with the new label, preserving
        start/end frames and updating the indices correctly.
        """
        annotation = None
        for ann in self.annotations:
            if ann.id == annotation_id:
                annotation = ann
                break

        if annotation is None:
            raise KeyError(f"Annotation {annotation_id} not found")
        if label_id not in self.labels:
            raise KeyError(f"Label {label_id} not found")

        label = self.labels[label_id]
        new_meta = dict(annotation.metadata)
        new_meta.pop("unlabeled", None)
        if metadata is not None:
            new_meta.update(metadata)

        self.remove_annotation(annotation.id)

        new_ann = Annotation(
            id=str(uuid.uuid4()),
            label_id=label_id,
            label_name=label.name,
            annotation_type=annotation.annotation_type,
            frame_index=annotation.frame_index,
            start_frame=annotation.start_frame,
            end_frame=annotation.end_frame,
            confidence=annotation.confidence,
            metadata=new_meta,
        )
        self._add_annotation(new_ann)

    def convert_unlabeled_to_idle(self) -> int:
        """Convertit tous les segments non labellisés en segments 'idle'.

        Retourne le nombre de segments convertis.
        """
        # Ensure the idle label exists
        if IDLE_LABEL_ID not in self.labels:
            idle_label = Label(
                id=IDLE_LABEL_ID,
                name=IDLE_LABEL_NAME,
                color=IDLE_LABEL_COLOR,
            )
            self.labels[IDLE_LABEL_ID] = idle_label
            self._annotations_by_label[IDLE_LABEL_ID] = []

        targets = [
            ann for ann in self.annotations
            if ann.label_id == UNLABELED_LABEL_ID
        ]
        for ann in targets:
            new_meta = dict(ann.metadata)
            new_meta.pop("unlabeled", None)
            self.remove_annotation(ann.id)
            new_ann = Annotation(
                id=str(uuid.uuid4()),
                label_id=IDLE_LABEL_ID,
                label_name=IDLE_LABEL_NAME,
                annotation_type=ann.annotation_type,
                start_frame=ann.start_frame,
                end_frame=ann.end_frame,
                confidence=ann.confidence,
                metadata=new_meta,
            )
            self._add_annotation(new_ann)
        return len(targets)

    def get_statistics(self) -> Dict[str, Any]:
        """Get annotation statistics.

        Returns:
            Dictionary with statistics
        """
        total_annotations = len(self.annotations)
        frame_annotations = sum(1 for a in self.annotations if a.annotation_type == LabelType.FRAME)
        interval_annotations = sum(1 for a in self.annotations if a.annotation_type == LabelType.INTERVAL)

        label_counts = {}
        for label_id, label in self.labels.items():
            label_counts[label.name] = len(self._annotations_by_label.get(label_id, []))

        return {
            'total_labels': len(self.labels),
            'total_annotations': total_annotations,
            'frame_annotations': frame_annotations,
            'interval_annotations': interval_annotations,
            'annotations_by_label': label_counts,
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"LabelManager(\n"
            f"  labels={stats['total_labels']},\n"
            f"  annotations={stats['total_annotations']} "
            f"({stats['frame_annotations']} frame, {stats['interval_annotations']} interval)\n"
            f")"
        )
