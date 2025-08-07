from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ..model.contraction import Contraction
from .initializer import Initializer


@dataclass(frozen=True)
class PointCloudInitializerCfg:
    name: Literal["point_cloud"]
    radius: float
    num_subsample_points: int
    interpolate_steps: int
    num_frozen_steps: int


class PointCloudInitializer(Initializer):
    point_cloud: Float[Tensor, "point 3"]

    def __init__(
        self,
        cfg: PointCloudInitializerCfg,
        contraction: Contraction,
        point_cloud: Float[Tensor, "point 3"],
    ) -> None:
        super().__init__()
        self.cfg = cfg

        point_cloud = self.subsample_point_cloud(point_cloud, cfg.num_subsample_points)
        self.register_buffer(
            "point_cloud",
            contraction.contract(point_cloud),
            persistent=False,
        )
        self.cache = None

    def subsample_point_cloud(
        self,
        point_cloud: Float[Tensor, "point 3"],
        num_samples: int,
    ) -> Float[Tensor, "subsampled_point 3"]:
        # Subsample with evenly spaced points.
        num_points, _ = point_cloud.shape
        indices = torch.linspace(
            0,
            num_points - 1,
            num_samples,
            dtype=torch.long,
            device=point_cloud.device,
        )
        return point_cloud[indices]

    def get_sdf_modify(
        self,
        sdf_in: Float[Tensor, " *batch"],
        xyz: Float[Tensor, "*batch xyz=3"],
        step: int,
    ) -> Float[Tensor, " *batch"] | float:
        """
        Computes the SDF modification by finding the minimum distance to the pcd
        and subtracting `self.cfg.radius`, effectively creating a union of spheres.
        """

        # It's assumed that the samples won't change after the first step, so we cache
        # the SDF modification (it's a bit expensive to compute).
        if step >= self.cfg.interpolate_steps + self.cfg.num_frozen_steps:
            self.cache = None
            self.point_cloud = None
            return sdf_in
        elif self.cache is None:
            with torch.no_grad():
                min_distances = torch.full_like(xyz[..., 0], torch.inf)

                # Chunk the calculation to avoid OOM.
                for chunk_points in self.point_cloud.split(500):
                    chunk_distances = torch.cdist(xyz, chunk_points, p=2)
                    chunk_min_distances, _ = chunk_distances.min(dim=-1)
                    min_distances = torch.minimum(min_distances, chunk_min_distances)

                # Find the minimum distance to any sampled point
                sdf_dist = min_distances - self.cfg.radius

                # Subtract radius to form the final SDF modification
                self.cache = sdf_dist

        # interpolate_scale = 1 - (step / self.cfg.interpolate_steps)
        if step < self.cfg.num_frozen_steps:
            interpolate_scale = 1
        else:
            interpolate_scale = 1 - (
                (step - self.cfg.num_frozen_steps) / self.cfg.interpolate_steps
            )
        return sdf_in * (1 - interpolate_scale) + self.cache * interpolate_scale
