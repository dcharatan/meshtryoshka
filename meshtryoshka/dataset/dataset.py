from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor

Stage = Literal["train", "test", "val"]


@dataclass(frozen=True)
class LoadedDataset:
    id: str
    image_paths: list[Path]
    images: Float[Tensor, "image 3 height width"]
    extrinsics: Float[Tensor, "image 4 4"]  # OpenCV-style world-to-camera
    intrinsics: Float[Tensor, "image 3 3"]  # unnormalized
    point_cloud: Float[Tensor, "point 3"] | None = None
    alpha_masks: Float[Tensor, "image 1 height width"] | None = None

    @property
    def device(self) -> torch.device:
        return self.images.device


T = TypeVar("T")


class Dataset(ABC, Generic[T]):
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def load(self, stage: Stage) -> LoadedDataset:
        pass

    @abstractmethod
    def get_cache_key(self, stage: Stage) -> str:
        pass
