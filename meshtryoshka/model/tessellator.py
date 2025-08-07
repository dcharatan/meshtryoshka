from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Float32, Int
from torch import Tensor, nn

from triangle_extraction import Voxels, tessellate_voxels

from ..model.contraction import Contraction
from ..utils import get_value_for_step
from .common import Scene


def cdf_to_sdf(
    cdf: Float[Tensor, "*batch"],
    sharpness: Float[Tensor, ""] | float,
) -> Float[Tensor, "*batch"]:
    return torch.log(cdf / (1 - cdf)) / sharpness


@dataclass(frozen=True)
class TessellatorCfg:
    shells: dict[int, tuple[float, ...]]
    sharpness: float


class Tessellator(nn.Module):
    step: Int[Tensor, ""]
    level_sets: Float[Tensor, " level_set"]

    def __init__(self, cfg: TessellatorCfg, contraction: Contraction) -> None:
        super().__init__()
        self.cfg = cfg
        self.contraction = contraction
        self.register_buffer("step", torch.zeros((1,), dtype=torch.int32))

    def tessellate(
        self,
        voxels: Voxels,
        signed_distances: Float32[Tensor, " vertex"],
        spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
        shell_mask: Bool[Tensor, " shell"] | None = None,
    ) -> Scene:
        triangles = tessellate_voxels(
            signed_distances,
            voxels.vertices,
            spherical_harmonics,
            voxels.neighbors,
            voxels.lower_corners,
            voxels.upper_corners,
            voxels.indices,
            self.level_sets if shell_mask is None else self.level_sets[shell_mask],
        )

        return Scene(
            self.contraction.uncontract(triangles.vertices),
            triangles.spherical_harmonics,
            triangles.signed_distances,
            triangles.faces,
            triangles.voxel_indices,
            self.level_sets if shell_mask is None else self.level_sets[shell_mask],
            triangles.face_boundaries,
        )

    def tessellate_surface(
        self,
        voxels: Voxels,
        signed_distances: Float32[Tensor, " vertex"],
        spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
    ) -> Scene:
        triangles = tessellate_voxels(
            signed_distances,
            voxels.vertices,
            spherical_harmonics,
            voxels.neighbors,
            voxels.lower_corners,
            voxels.upper_corners,
            voxels.indices,
            torch.zeros((1,), device=self.device, dtype=torch.float32),
        )

        return Scene(
            self.contraction.uncontract(triangles.vertices),
            triangles.spherical_harmonics,
            torch.full_like(triangles.signed_distances, -1e8),  # fully opaque
            triangles.faces,
            triangles.voxel_indices,
            torch.tensor([-1e8]).to(triangles.vertices.device),  # fully opaque
            triangles.face_boundaries,
        )

    def set_step(self, step: int) -> None:
        self.step.fill_(step)

    @property
    def device(self) -> torch.device:
        return self.step.device

    @property
    def level_sets(self) -> Float[Tensor, " level_set"]:
        with torch.no_grad():
            level_set_tensor = torch.tensor(
                get_value_for_step(self.step.item(), self.cfg.shells),
                dtype=torch.float32,
                device=self.device,
            )
            return cdf_to_sdf(level_set_tensor, self.cfg.sharpness)

    def get_sharpness(self) -> float:
        return self.cfg.sharpness
