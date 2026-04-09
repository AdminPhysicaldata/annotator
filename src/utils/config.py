"""Configuration management for VIVE Labeler."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    video_fps: int = 30
    timeline_height: int = 100
    overlay_opacity: float = 0.7
    trajectory_color: str = "#00FF00"
    trajectory_thickness: int = 2
    show_axes: bool = True
    axes_length: float = 0.1  # meters


@dataclass
class SynchronizationConfig:
    """Configuration for video/sensor synchronization."""
    interpolation_method: str = "linear"  # linear, cubic, nearest
    max_time_diff_ms: float = 50.0  # Maximum acceptable time difference
    tolerance_ms: float = 5.0  # Tolerance for synchronization warnings


@dataclass
class LabelingConfig:
    """Configuration for labeling system."""
    default_labels: List[str] = field(default_factory=lambda: [
        "grasping", "moving", "placing", "idle"
    ])
    enable_frame_labels: bool = True
    enable_interval_labels: bool = True
    auto_save_interval_sec: int = 300  # Auto-save every 5 minutes


@dataclass
class DataConfig:
    """Configuration for data loading."""
    cache_dir: Optional[Path] = None
    use_streaming: bool = False
    preload_videos: bool = False
    max_memory_gb: float = 8.0


@dataclass
class RabbitMQConfig:
    """Configuration for RabbitMQ connection."""
    host: str = "192.168.88.246"
    port: int = 5672
    username: str = ""
    password: str = ""
    queue_name: str = "ingestion_queue"
    virtual_host: str = "/"


@dataclass
class NASConfig:
    """Configuration for NAS SFTP access."""
    host: str = "192.168.88.248"
    port: int = 22
    username: str = ""
    password: str = ""
    key_path: Optional[str] = None   # Path to SSH private key (preferred over password)
    local_dir: Optional[str] = None  # Local cache directory for downloaded files
    silver_root: str = "/DB-EXORIA/lakehouse/silver/annotated"


@dataclass
class S3Config:
    """Configuration for AWS S3 access."""
    bucket: str = "physical-data-storage"
    region: str = "eu-west-3"
    bronze_prefix: str = "bronze"
    aws_access_key_id: Optional[str] = None      # use env vars AWS_ACCESS_KEY_ID or ~/.aws/credentials
    aws_secret_access_key: Optional[str] = None  # use env vars AWS_SECRET_ACCESS_KEY or ~/.aws/credentials
    presign_ttl: int = 900  # seconds (15 min)


@dataclass
class SpoolConfig:
    """Configuration for the SPOOL SFTP server (scenario inbox)."""
    host: str = "192.168.88.28"
    port: int = 22
    username: str = ""
    password: str = ""
    inbox_base: str = "/srv/exoria/"


@dataclass
class HDDConfig:
    """Configuration for the HDD SFTP archive server."""
    host: str = "192.168.88.82"
    port: int = 22
    username: str = "exoria"
    password: str = "Admin123456"
    inbox_base: str = "/mnt/inbox"
    bronze_base: str = "/mnt/storage/bronze"
    silver_base: str = "/mnt/storage/silver"
    gold_base: str = "/mnt/storage/gold"
    send_base: str = "/mnt/storage/send"
    retry_base: str = "/mnt/storage/retry"


@dataclass
class MongoDBConfig:
    """Configuration for MongoDB connection."""
    connection_string: str = "mongodb+srv://christoloisel:rose@cluster0.ppyauvl.mongodb.net/"
    database: str = "physical_data"
    collection: str = "annotators"


@dataclass
class AppConfig:
    """Main application configuration."""
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    synchronization: SynchronizationConfig = field(default_factory=SynchronizationConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rabbitmq: RabbitMQConfig = field(default_factory=RabbitMQConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    s3: S3Config = field(default_factory=S3Config)
    mongodb: MongoDBConfig = field(default_factory=MongoDBConfig)
    spool: SpoolConfig = field(default_factory=SpoolConfig)
    hdd: HDDConfig = field(default_factory=HDDConfig)

    # Window settings
    window_width: int = 1600
    window_height: int = 900
    theme: str = "dark"  # dark, light

    # Annotator identity
    annotator: str = ""

    @classmethod
    def load_from_file(cls, config_path: Path) -> "AppConfig":
        """Load configuration from YAML file.

        Never raises — returns defaults on any parse or I/O error.
        """
        if not config_path.exists():
            logger.info("Config file not found at %s — using defaults", config_path)
            return cls()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            logger.error(
                "Cannot read config %s: %s — using defaults", config_path, exc
            )
            return cls()

        if not isinstance(data, dict):
            logger.warning(
                "Config file %s is empty or not a mapping — using defaults", config_path
            )
            return cls()

        config = cls()

        # Each section is loaded independently so a bad section doesn't kill the rest.
        _section_map = [
            ("visualization",  VisualizationConfig,  "visualization"),
            ("synchronization", SynchronizationConfig, "synchronization"),
            ("labeling",       LabelingConfig,        "labeling"),
            ("data",           DataConfig,            "data"),
            ("rabbitmq",       RabbitMQConfig,        "rabbitmq"),
            ("nas",            NASConfig,             "nas"),
            ("s3",             S3Config,              "s3"),
            ("mongodb",        MongoDBConfig,         "mongodb"),
            ("spool",          SpoolConfig,           "spool"),
            ("hdd",            HDDConfig,             "hdd"),
        ]
        for key, klass, attr in _section_map:
            raw_section = data.get(key)
            if raw_section is None:
                continue
            if not isinstance(raw_section, dict):
                logger.warning(
                    "Config section [%s] is not a mapping (got %s) — using defaults",
                    key, type(raw_section).__name__,
                )
                continue
            try:
                setattr(config, attr, klass(**raw_section))
            except Exception as exc:
                logger.warning(
                    "Config section [%s] invalid (%s) — using defaults for this section",
                    key, exc,
                )

        # Top-level scalar settings
        try:
            config.window_width = int(data.get("window_width", config.window_width))
        except (TypeError, ValueError):
            pass
        try:
            config.window_height = int(data.get("window_height", config.window_height))
        except (TypeError, ValueError):
            pass
        theme = data.get("theme", config.theme)
        if isinstance(theme, str):
            config.theme = theme
        annotator = data.get("annotator", config.annotator)
        if isinstance(annotator, str):
            config.annotator = annotator

        return config

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file.

        Logs an error but never raises on I/O failure.
        """
        data = {
            "visualization": {
                "video_fps": self.visualization.video_fps,
                "timeline_height": self.visualization.timeline_height,
                "overlay_opacity": self.visualization.overlay_opacity,
                "trajectory_color": self.visualization.trajectory_color,
                "trajectory_thickness": self.visualization.trajectory_thickness,
                "show_axes": self.visualization.show_axes,
                "axes_length": self.visualization.axes_length,
            },
            "synchronization": {
                "interpolation_method": self.synchronization.interpolation_method,
                "max_time_diff_ms": self.synchronization.max_time_diff_ms,
                "tolerance_ms": self.synchronization.tolerance_ms,
            },
            "labeling": {
                "default_labels": self.labeling.default_labels,
                "enable_frame_labels": self.labeling.enable_frame_labels,
                "enable_interval_labels": self.labeling.enable_interval_labels,
                "auto_save_interval_sec": self.labeling.auto_save_interval_sec,
            },
            "data": {
                "cache_dir": str(self.data.cache_dir) if self.data.cache_dir else None,
                "use_streaming": self.data.use_streaming,
                "preload_videos": self.data.preload_videos,
                "max_memory_gb": self.data.max_memory_gb,
            },
            "rabbitmq": {
                "host": self.rabbitmq.host,
                "port": self.rabbitmq.port,
                "username": self.rabbitmq.username,
                "password": self.rabbitmq.password,
                "queue_name": self.rabbitmq.queue_name,
                "virtual_host": self.rabbitmq.virtual_host,
            },
            "nas": {
                "host": self.nas.host,
                "port": self.nas.port,
                "username": self.nas.username,
                "password": self.nas.password,
                "key_path": self.nas.key_path,
                "local_dir": self.nas.local_dir,
                "silver_root": self.nas.silver_root,
            },
            "s3": {
                "bucket": self.s3.bucket,
                "region": self.s3.region,
                "bronze_prefix": self.s3.bronze_prefix,
                "aws_access_key_id": self.s3.aws_access_key_id,
                "aws_secret_access_key": self.s3.aws_secret_access_key,
                "presign_ttl": self.s3.presign_ttl,
            },
            "mongodb": {
                "connection_string": self.mongodb.connection_string,
                "database": self.mongodb.database,
                "collection": self.mongodb.collection,
            },
            "spool": {
                "host": self.spool.host,
                "port": self.spool.port,
                "username": self.spool.username,
                "password": self.spool.password,
                "inbox_base": self.spool.inbox_base,
            },
            "hdd": {
                "host": self.hdd.host,
                "port": self.hdd.port,
                "username": self.hdd.username,
                "password": self.hdd.password,
                "inbox_base": self.hdd.inbox_base,
                "bronze_base": self.hdd.bronze_base,
                "silver_base": self.hdd.silver_base,
                "gold_base": self.hdd.gold_base,
                "send_base": self.hdd.send_base,
                "retry_base": self.hdd.retry_base,
            },
            "window_width": self.window_width,
            "window_height": self.window_height,
            "theme": self.theme,
            "annotator": self.annotator,
        }

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as exc:
            logger.error("Cannot save config to %s: %s", config_path, exc)


# Default configuration path
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "vive_labeler" / "config.yaml"
