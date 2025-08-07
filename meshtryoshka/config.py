import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
from dacite import Config, from_dict
from omegaconf import OmegaConf

from .dataset import DatasetCfg
from .model import ModelCfg


@dataclass(frozen=True)
class LearningRateCfg:
    base: float
    schedule: dict[int, float]


@dataclass(frozen=True)
class TrainCfg:
    num_steps: int
    learning_rate: LearningRateCfg
    batch_size: dict[int, int]
    max_image_size: int | None
    checkpoint_interval: int
    visualization_interval: int | None
    val_interval: int
    heartbeat_interval_seconds: float
    num_dump: int | None
    dataset_device: Literal["cuda", "cpu"]


@dataclass(frozen=True)
class RootCfg:
    dataset: DatasetCfg
    model: ModelCfg
    train: TrainCfg
    seed: int
    overwrite: bool
    enable_viewer: bool
    eval_only: bool
    render_trajectory_only: bool
    max_checkpoint_step: int | None
    name: str | None  # dummy, only for Slurm jobs
    benchmark_rendering_speed: bool


def get_typed_config() -> RootCfg:
    # Read the configuration using Hydra.
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(config_name="main", overrides=sys.argv[1:])

    # Convert the configuration to a nested dataclass.
    return from_dict(
        RootCfg,
        OmegaConf.to_container(cfg),
        config=Config(cast=[Path, tuple]),
    )
