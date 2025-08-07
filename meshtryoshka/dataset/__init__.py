import logging
from time import time
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..components.typing_utils import to_device
from .dataset import Dataset, LoadedDataset, Stage
from .dataset_mip360 import DatasetMIP360, DatasetMIP360Cfg
from .dataset_nerf_synthetic import DatasetNeRFSynthetic, DatasetNeRFSyntheticCfg

DatasetCfg = DatasetNeRFSyntheticCfg | DatasetMIP360Cfg

DATASETS: dict[str, type[Dataset[Any]]] = {
    "nerf_synthetic": DatasetNeRFSynthetic,
    "mip360": DatasetMIP360,
}


def load_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    device: torch.device = torch.device("cpu"),
) -> LoadedDataset:
    logging.info(f"Loading dataset {cfg.name} for stage {stage}.")
    start = time()
    dataset = DATASETS[cfg.name](cfg)
    loaded = dataset.load(stage)
    logging.info(f"Loading dataset took {time() - start:.2f} seconds.")
    return to_device(loaded, device)


def resize_images(
    images: Float[Tensor, "batch 3 old_height old_width"],
    shape: tuple[int, int],
) -> Float[Tensor, "batch 3 new_height new_width"]:
    h, w = shape

    resized_images = []
    for image in images:
        image = rearrange(image, "c h w -> h w c")
        image = Image.fromarray((image.detach().cpu().numpy() * 255).astype(np.uint8))
        image = image.resize((w, h), Image.Resampling.LANCZOS)
        image = rearrange(np.array(image) / 255, "h w c -> c h w")
        image = torch.tensor(image, dtype=images.dtype, device=images.device)
        resized_images.append(image)

    return torch.stack(resized_images)


def resize_dataset(
    dataset: LoadedDataset, max_image_size: Optional[int]
) -> LoadedDataset:
    _, _, h, w = dataset.images.shape
    max_current_image_size = max(h, w)

    # Check if the dataset is fine to use unchanged.
    if max_image_size is None or max_current_image_size < max_image_size:
        return dataset

    # If necessary, resize the dataset.
    scale_factor = max_image_size / max_current_image_size
    new_h, new_w = round(scale_factor * h), round(scale_factor * w)
    images = resize_images(dataset.images, (new_h, new_w))
    intrinsics = dataset.intrinsics.clone()
    intrinsics[:, :2] *= scale_factor
    return LoadedDataset(dataset.id, images, dataset.extrinsics.clone(), intrinsics)
