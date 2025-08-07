from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Optional

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from PIL import Image

from ..components.ply_loader import fetchPly, read_points3D_binary, storePly
from ..components.projection import homogenize_points, transform_rigid
from .dataset import Dataset, LoadedDataset, Stage
from .normalization import auto_orient_and_center_poses

INDOOR = ["bonsai", "counter", "kitchen", "room"]
OUTDOOR = ["bicycle", "flowers", "garden", "stump", "treehill"]


@dataclass(frozen=True)
class DatasetMIP360Cfg:
    name: Literal["mip360"]
    scene: str
    directory: Path
    scale_factor: float
    llffhold: int


class DatasetMIP360(Dataset[DatasetMIP360Cfg]):
    # Static cache for the pose-normalization transform
    _pose_transform: ClassVar[Optional[torch.Tensor]] = None
    _conversion: ClassVar[torch.Tensor] = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    def load(self, stage: Stage) -> LoadedDataset:
        path = self.cfg.directory / self.cfg.scene

        # --- Point cloud loading ---
        ply_path = path / "sparse/0/points3D.ply"
        bin_path = path / "sparse/0/points3D.bin"
        if not ply_path.exists():
            print("Converting point3d.bin to .ply (first-time only)...")
            xyz, rgb, _ = read_points3D_binary(str(bin_path))
            storePly(str(ply_path), xyz, rgb)
        point_cloud = fetchPly(str(ply_path)).points  # (N, 3)

        # --- Metadata and splits ---
        metadata = torch.tensor(np.load(path / "poses_bounds.npy"), dtype=torch.float32)
        b = metadata.shape[0]
        if stage == "train":
            stage_indices = [i for i in range(b) if i % self.cfg.llffhold != 0]
        elif stage == "val":
            stage_indices = [i for i in range(b) if i % self.cfg.llffhold == 0]
        else:
            stage_indices = [i for i in range(b) if i % self.cfg.llffhold == 0]

        # --- Build raw camera extrinsics ---
        cams = rearrange(metadata[:, :-2], "b (i j) -> b i j", i=3, j=5)
        cams = cams[stage_indices]
        rot = cams[:, :3, :3]
        trans = cams[:, :3, 3]
        h, w, f = cams[:, :3, 4].unbind(-1)
        # assemble homogeneous c2w
        extrinsics = repeat(torch.eye(4), "i j -> b i j", b=len(stage_indices)).clone()
        extrinsics[:, :3, :3] = rot
        extrinsics[:, :3, 3] = trans
        # convert to internal CV style
        extrinsics = extrinsics @ self._conversion

        # --- Compute or reuse normalization transform ---
        if stage == "train":
            # orient + center
            extrinsics, base_transform = auto_orient_and_center_poses(
                extrinsics, method="vertical"
            )
            # compute scale
            positions = extrinsics[:, :3, 3]
            minima = positions.min(dim=0).values
            maxima = positions.max(dim=0).values
            factor = 2 / (maxima - minima).max()
            scale_mat = torch.eye(4)
            scale_mat[:3, :3] *= factor
            transform = scale_mat @ base_transform
            DatasetMIP360._pose_transform = transform
            extrinsics[:, :3, 3] *= factor
            point_cloud = transform_rigid(homogenize_points(point_cloud), transform)[
                ..., :3
            ]
        else:
            if DatasetMIP360._pose_transform is None:
                raise RuntimeError(
                    "Normalization transform undefined: load train stage first."
                )
            transform = DatasetMIP360._pose_transform
            # apply same transform to raw extrinsics and point cloud
            # new_pcd = transform @ pcd
            # new_w2c = old_w2c @ transform.inverse()
            # new_c2w = transform @ old_c2w
            extrinsics = transform @ extrinsics
            point_cloud = transform_rigid(homogenize_points(point_cloud), transform)[
                ..., :3
            ]

        # triangle rasterizer expects world->camera
        extrinsics = extrinsics.inverse()

        # --- Image loading ---
        if self.cfg.scene in INDOOR:
            image_folder = "images_2"
        elif self.cfg.scene in OUTDOOR:
            image_folder = "images_4"
        else:
            raise ValueError(f"Unknown scene {self.cfg.scene}.")
        image_paths = sorted((path / image_folder).iterdir())
        to_tensor = tf.ToTensor()
        images = [to_tensor(Image.open(str(p))) for p in image_paths]
        images = torch.stack([images[i] for i in stage_indices])
        image_paths = [image_paths[i] for i in stage_indices]

        # Load the intrinsics, normalize them.
        intrinsics = repeat(torch.eye(3), "i j -> b i j", b=len(stage_indices)).clone()
        intrinsics[:, :2, 2] = 0.5
        intrinsics[:, 0, 0] = f / w
        intrinsics[:, 1, 1] = f / h

        # Unnormalize intrinsics based on new image resolution.
        h, w = images.shape[-2:]
        wh = torch.tensor((w, h))
        intrinsics[:, :2] *= wh[None, :, None]

        return LoadedDataset(
            self.get_cache_key(stage),
            image_paths,
            images,
            extrinsics,
            intrinsics,
            point_cloud=point_cloud,
        )

    def get_cache_key(self, stage: Stage) -> str:
        return f"mip360_{self.cfg.scene}_{stage}"
