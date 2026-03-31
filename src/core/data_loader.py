"""Data loader for LeRobot datasets.

Handles loading of LeRobotDataset v3.0 format including:
- Video files (MP4)
- Sensor data (Parquet files)
- Metadata (info.json, stats.json, tasks.parquet)
- Episode management
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: lerobot not installed. Install with: pip install lerobot>=0.4.0")


class LeRobotDataLoader:
    """Loader for LeRobotDataset v3.0 format."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        use_streaming: bool = False,
        cache_dir: Optional[Path] = None
    ):
        """Initialize data loader.

        Args:
            dataset_path: Path to local dataset or HuggingFace repo ID
            use_streaming: If True, stream data without downloading
            cache_dir: Directory for caching downloaded data
        """
        self.dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
        self.use_streaming = use_streaming
        self.cache_dir = cache_dir

        # Dataset metadata
        self.info: Optional[Dict] = None
        self.stats: Optional[Dict] = None
        self.tasks: Optional[pd.DataFrame] = None

        # LeRobot dataset object
        self.dataset: Optional[LeRobotDataset] = None

        # Episode information
        self.episodes: List[Dict] = []
        self.num_episodes: int = 0

        # Load dataset
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load the dataset and metadata."""
        if not LEROBOT_AVAILABLE:
            raise ImportError("lerobot library not available. Install with: pip install lerobot>=0.4.0")

        # Check if local path or HuggingFace repo
        is_local = self.dataset_path.exists() if isinstance(self.dataset_path, Path) else False

        if is_local:
            self._load_local_dataset()
        else:
            self._load_hub_dataset()

    def _load_local_dataset(self) -> None:
        """Load dataset from local filesystem."""
        dataset_root = self.dataset_path

        # Load metadata files
        info_path = dataset_root / "meta" / "info.json"
        stats_path = dataset_root / "meta" / "stats.json"
        tasks_path = dataset_root / "meta" / "tasks.parquet"

        if info_path.exists():
            with open(info_path, "r") as f:
                self.info = json.load(f)

        if stats_path.exists():
            with open(stats_path, "r") as f:
                self.stats = json.load(f)

        if tasks_path.exists():
            self.tasks = pd.read_parquet(tasks_path)

        # Load episode metadata from meta/episodes directory
        episodes_dir = dataset_root / "meta" / "episodes"
        if episodes_dir.exists():
            episode_files = sorted(episodes_dir.glob("*.parquet"))
            for ep_file in episode_files:
                ep_df = pd.read_parquet(ep_file)
                self.episodes.extend(ep_df.to_dict('records'))

        self.num_episodes = len(self.episodes)

        # Load LeRobot dataset
        try:
            self.dataset = LeRobotDataset(
                repo_id=str(dataset_root),
                root=self.cache_dir,
            )
        except Exception as e:
            print(f"Warning: Could not load LeRobot dataset object: {e}")
            print("Continuing with manual data loading...")

    def _load_hub_dataset(self) -> None:
        """Load dataset from HuggingFace Hub."""
        repo_id = str(self.dataset_path)

        try:
            if self.use_streaming:
                from lerobot.common.datasets.lerobot_dataset import StreamingLeRobotDataset
                self.dataset = StreamingLeRobotDataset(repo_id=repo_id)
            else:
                self.dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=self.cache_dir,
                )

            # Extract metadata from dataset
            if hasattr(self.dataset, 'meta'):
                self.info = self.dataset.meta
            if hasattr(self.dataset, 'stats'):
                self.stats = self.dataset.stats

            self.num_episodes = len(self.dataset.episodes) if hasattr(self.dataset, 'episodes') else 0
            self.episodes = self.dataset.episodes if hasattr(self.dataset, 'episodes') else []

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from HuggingFace Hub: {e}")

    def get_sample(self, index: int) -> Dict:
        """Get a single sample by index.

        Args:
            index: Sample index

        Returns:
            Dictionary with sample data (observation, action, timestamp, etc.)
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")

        return self.dataset[index]

    def get_episode(self, episode_idx: int) -> Dict:
        """Get all data for a specific episode.

        Args:
            episode_idx: Episode index

        Returns:
            Dictionary with episode data
        """
        if episode_idx >= self.num_episodes:
            raise IndexError(f"Episode {episode_idx} out of range (total: {self.num_episodes})")

        episode_info = self.episodes[episode_idx]

        # Get episode frames
        start_idx = episode_info.get('from', 0)
        end_idx = episode_info.get('to', start_idx + 1)

        samples = []
        for idx in range(start_idx, end_idx):
            samples.append(self.get_sample(idx))

        return {
            'episode_info': episode_info,
            'samples': samples,
            'episode_index': episode_idx,
        }

    def get_video_path(self, camera_key: str, episode_idx: int) -> Optional[Path]:
        """Get path to video file for specific camera and episode.

        Args:
            camera_key: Camera identifier (e.g., 'front_left', 'wrist')
            episode_idx: Episode index

        Returns:
            Path to video file, or None if not found
        """
        if not isinstance(self.dataset_path, Path):
            return None

        # Try to find video file
        video_dir = self.dataset_path / "videos" / camera_key
        if not video_dir.exists():
            return None

        # Look for episode video
        episode_info = self.episodes[episode_idx]
        episode_id = episode_info.get('episode_id', f'episode_{episode_idx:06d}')

        video_path = video_dir / f"{episode_id}.mp4"
        if video_path.exists():
            return video_path

        return None

    def get_camera_keys(self) -> List[str]:
        """Get list of available camera keys.

        Returns:
            List of camera identifiers
        """
        if self.info and 'camera_keys' in self.info:
            return self.info['camera_keys']

        # Try to infer from first sample
        if self.dataset and len(self.dataset) > 0:
            sample = self.dataset[0]
            camera_keys = [
                key.replace('observation.images.', '')
                for key in sample.keys()
                if key.startswith('observation.images.')
            ]
            return camera_keys

        # Try to infer from video directory
        if isinstance(self.dataset_path, Path):
            video_dir = self.dataset_path / "videos"
            if video_dir.exists():
                return [d.name for d in video_dir.iterdir() if d.is_dir()]

        return []

    def get_state_keys(self) -> List[str]:
        """Get list of available state observation keys.

        Returns:
            List of state keys (e.g., 'observation.state', 'observation.vive_position')
        """
        if self.dataset and len(self.dataset) > 0:
            sample = self.dataset[0]
            state_keys = [
                key for key in sample.keys()
                if key.startswith('observation.') and not key.startswith('observation.images.')
            ]
            return state_keys

        if self.info and 'features' in self.info:
            return [
                key for key in self.info['features'].keys()
                if key.startswith('observation.') and 'image' not in key.lower()
            ]

        return []

    def get_action_keys(self) -> List[str]:
        """Get list of available action keys.

        Returns:
            List of action keys
        """
        if self.dataset and len(self.dataset) > 0:
            sample = self.dataset[0]
            return [key for key in sample.keys() if key.startswith('action')]

        if self.info and 'features' in self.info:
            return [key for key in self.info['features'].keys() if key.startswith('action')]

        return ['action']  # Default

    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self.dataset) if self.dataset else 0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LeRobotDataLoader(\n"
            f"  path={self.dataset_path},\n"
            f"  episodes={self.num_episodes},\n"
            f"  samples={len(self)},\n"
            f"  cameras={self.get_camera_keys()},\n"
            f")"
        )
