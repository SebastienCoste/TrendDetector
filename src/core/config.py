from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import yaml
from pathlib import Path

class ModelConfig(BaseModel):
    type: str = Field(default="classification", description="Model type: 'classification' or 'regression'")
    n_trees: int = Field(default=10, description="Number of trees in random forest")
    drift_threshold: float = Field(default=0.01, description="ADWIN delta parameter")
    memory_size: int = Field(default=10000, description="Maximum trend history size")
    max_features: float = Field(default=0.6, description="Feature sampling ratio")
    update_frequency: int = Field(default=1000, description="Trend boundary update interval")
    embedding_dim: int = Field(default=512, description="Expected embedding dimension")
    max_clusters: int = Field(default=20, description="Maximum clusters per trend type")
    time_decay_hours: int = Field(default=24, description="Time-based similarity decay")
    output_range: List[float] = Field(default=[-1, 1], description="Output range for regression models")

class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    log_level: str = Field(default="INFO", description="Logging level")

class StorageConfig(BaseModel):
    model_path: str = Field(default="./models", description="Model storage path")
    checkpoint_interval: int = Field(default=1000, description="Checkpoint save interval")
    auto_save: bool = Field(default=True, description="Enable automatic saving")
    version_format: str = Field(default="v{}", description="Version format string")

class GPUConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable GPU acceleration")
    device_id: int = Field(default=0, description="GPU device ID")
    fallback_cpu: bool = Field(default=True, description="Fallback to CPU on errors")
    memory_limit: float = Field(default=0.8, description="GPU memory usage limit")

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", description="Log level")
    file_path: str = Field(default="./logs/trend_service.log", description="Log file path")
    log_requests: bool = Field(default=True, description="Log API requests")
    log_predictions: bool = Field(default=True, description="Log predictions")
    log_drift_events: bool = Field(default=True, description="Log drift events")
    max_file_size: str = Field(default="100MB", description="Max log file size")
    backup_count: int = Field(default=5, description="Number of backup files")

class AppConfig(BaseModel):
    model_settings: ModelConfig = Field(default_factory=ModelConfig)
    server_config: ServerConfig = Field(default_factory=ServerConfig)
    storage_config: StorageConfig = Field(default_factory=StorageConfig)
    gpu_config: GPUConfig = Field(default_factory=GPUConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "AppConfig":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)