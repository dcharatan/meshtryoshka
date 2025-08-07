import json
from dataclasses import dataclass
from math import tan
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import repeat
from PIL import Image
from tqdm import tqdm

from .dataset import Dataset, LoadedDataset, Stage


@dataclass(frozen=True)
class DatasetNeRFSyntheticCfg:
    name: Literal["nerf_synthetic"]
    scene: str
    directory: Path
    scale_factor: float
    background: str


class DatasetNeRFSynthetic(Dataset[DatasetNeRFSyntheticCfg]):
    def load(self, stage: Stage) -> LoadedDataset:
        path = self.cfg.directory / self.cfg.scene

        # Load the metadata.
        transforms = tf.ToTensor()
        with (path / f"transforms_{stage}.json").open("r") as f:
            metadata = json.load(f)

        # This converts the extrinsics to OpenCV style.
        conversion = torch.eye(4, dtype=torch.float32)
        conversion[1:3, 1:3] *= -1

        # Read the images and extrinsics.
        images = []
        extrinsics = []
        image_paths = []
        for frame in tqdm(metadata["frames"], f"Loading {stage} frames"):
            extrinsics.append(
                torch.tensor(frame["transform_matrix"], dtype=torch.float32)
                @ conversion
            )

            # Read the image.
            image = Image.open(path / f"{frame['file_path']}.png")
            image_paths.append(path / f"{frame['file_path']}.png")

            # Composite the image onto a white background.
            # background = Image.new("RGB", image.size, "white").convert("RGBA")
            background = Image.new("RGB", image.size, self.cfg.background).convert(
                "RGBA"
            )
            rgb = Image.alpha_composite(background, image)
            rgb = transforms(rgb.convert("RGB"))
            images.append(rgb)
        images = torch.stack(images)
        extrinsics = torch.stack(extrinsics)
        extrinsics[:, :3, 3] *= self.cfg.scale_factor
        extrinsics = extrinsics.inverse()

        # Convert the intrinsics.
        camera_angle_x = float(metadata["camera_angle_x"])
        focal_length = 0.5 / tan(0.5 * camera_angle_x)
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics[:2, :2] *= focal_length
        intrinsics[:2, 2] = 0.5
        n, _, h, w = images.shape
        wh = torch.tensor((w, h))
        intrinsics[:2] *= wh[:, None]
        intrinsics = repeat(intrinsics, "i j -> n i j", n=n)

        return LoadedDataset(
            self.get_cache_key(stage),
            image_paths,
            images,
            extrinsics,
            intrinsics,
        )

    def get_cache_key(self, stage: Stage) -> str:
        return f"nerf_synthetic_{self.cfg.directory.name}_{self.cfg.scene}_{stage}"
