"""
This module defines the configuration system for the circle detection task.

It includes:
- Validation of hyperparameters and experiment settings.
- Structured configuration using Python dataclasses for clarity and type safety.
"""

# Python modules
import json
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

# DL libraries
import torch


def load_json(file_path: str):
    """
    Loads a JSON configuration file from the specified absolute path.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config


# Structured configuration dataclasses
@dataclass
class CircleConfig:
    """Holds parameters related to the circle generation process."""
    img_size: int
    min_radius: int
    max_radius: int
    noise_level: float


@dataclass
class DataConfig:
    """Holds parameters for the dataset split and training epochs."""
    num_train: int
    num_val: int
    num_test: int
    num_epochs: int
    batch_size: int


@dataclass
class OptimizerConfig:
    """Holds optimizer type and its associated parameters."""
    type: str
    params: Dict[str, Any]


@dataclass
class SchedulerConfig:
    """Holds scheduler type and its associated parameters."""
    type: str
    params: Dict[str, Any]


@dataclass
class Config:
    """
    Main configuration dataclass combining all elements.
    Supports loading from JSON and validation of values.
    """
    input_shape: Tuple[int, int, int]
    channels: Tuple[int, ...]
    kernels: Tuple[int, ...]
    pools: Tuple[int, ...]
    strides: Optional[Tuple[int, ...]]
    FFN_dims: Tuple[int, ...]

    circle: CircleConfig
    data_config: DataConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig

    thresholds: List[float]
    log_dir: str
    checkpoint_dir: str
    model_name: str
    pretrained_model: str
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def from_json(path: str) -> "Config":
        """ Loads configuration from a JSON file and returns a validated Config object. """
        data = load_json(path)

        # Convert lists to tuples
        for key in ["input_shape", "channels", "kernels", "pools", "FFN_dims"]:
            data[key] = tuple(data[key])
        if data.get("strides") is not None:
            data["strides"] = tuple(data["strides"])

        # Parse nested configuration blocks
        data["circle"] = CircleConfig(**data["circle"])
        data["data_config"] = DataConfig(**data["data_config"])
        data["optimizer_config"] = OptimizerConfig(**data["optimizer_config"])
        data["scheduler_config"] = SchedulerConfig(**data["scheduler_config"])

        config = Config(**data)
        config.validate()
        return config

    def validate(self):
        """ Validates the loaded configuration to ensure consistency and reasonable values. """
        if not (
            self.circle.img_size > 0 and self.input_shape[1] and
            self.circle.min_radius > 0 and self.circle.max_radius > 0 and
            self.circle.min_radius <= self.circle.max_radius and
            self.circle.max_radius <= self.circle.img_size / 2 and
            0 <= self.circle.noise_level <= 1
        ):
            raise ValueError("Invalid circle configuration.")

        if not (
            self.data_config.num_train > 0 and
            self.data_config.num_val > 0 and
            self.data_config.num_test > 0 and
            self.data_config.batch_size > 0 and
            self.data_config.num_epochs > 0
        ):
            raise ValueError("Invalid data configuration.")

        if not (
            isinstance(self.thresholds, list) and len(self.thresholds) == 4 and
            all(0 < val < 1 for val in self.thresholds) and
            self.thresholds == sorted(self.thresholds)
        ):
            raise ValueError("Invalid thresholds configuration.")
