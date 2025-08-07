from dataclasses import dataclass
from math import exp

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class ContractionCfg:
    # The radius of the center region, which is scaled linearly. The scene is assumed to
    # be scaled such that the cameras fit into a cube with bounds [-1, 1] on each axis.
    # Thus, a radius of 1 is guaranteed to keep the cameras in linearly scaled space.
    center_radius: float

    # The assumed ratio between the tesseract's arm resolution and the tesseract's
    # center resolution.
    arm_ratio: float

    # Whether to disable the exponential scaling and simply scale linearly between the
    # exponential scaling's minimum and maximum.
    disable_exponential: bool


@dataclass(frozen=True)
class Contraction:
    cfg: ContractionCfg
    center_extent: float

    @torch.no_grad()
    def contract(
        self,
        xyz: Float[Tensor, "*batch xyz=3"],
    ) -> Float[Tensor, "*batch xyz=3"]:
        # Determine which points are within the center.
        center_extent = self.center_extent
        center_mask = (xyz.abs() <= self.cfg.center_radius).all(dim=-1)

        def handle_center(xyz):
            return xyz * center_extent / self.cfg.center_radius

        def handle_other(xyz):
            # Compute the uncontracted parameter t_u.
            near_u = self.cfg.center_radius
            far_ratio = exp(2 * self.cfg.arm_ratio)
            far_u = far_ratio * self.cfg.center_radius
            t_u = (xyz.abs().max(dim=-1).values - near_u) / (far_u - near_u)

            # Compute the contracted parameter t_c.
            t_c = (t_u * (far_ratio - 1) + 1).log() / (2 * self.cfg.arm_ratio)

            n = t_c + self.center_extent * (1 - t_c)
            d = far_u * t_u + near_u * (1 - t_u)
            return xyz * (n / d)[..., None]

        if self.cfg.disable_exponential:
            # For the ablation, just scale linearly.
            result = xyz / exp(2 * self.cfg.arm_ratio)
        else:
            # Separately handle center and other points.
            result = torch.empty_like(xyz)
            result[center_mask] = handle_center(xyz[center_mask])
            result[~center_mask] = handle_other(xyz[~center_mask])

        # Map [-1, 1] to [0, 1].
        return result * 0.5 + 0.5

    @torch.no_grad()
    def uncontract(
        self,
        xyz: Float[Tensor, "*batch xyz=3"],
    ) -> Float[Tensor, "*batch xyz=3"]:
        # Map [0, 1] to [-1, 1].
        xyz = xyz.clip(min=0, max=1)
        xyz = xyz * 2 - 1

        # Determine which points are within the center.
        center_extent = self.center_extent
        center_mask = (xyz.abs() <= center_extent).all(dim=-1)

        def handle_center(xyz):
            return xyz * self.cfg.center_radius / center_extent

        def handle_other(xyz):
            # Compute the contracted parameter t_c.
            t_c = xyz.abs().max(dim=-1).values
            t_c = (t_c - center_extent) / (1 - center_extent)

            # Compute the uncontracted parameter t_u.
            far_ratio = exp(2 * self.cfg.arm_ratio)
            t_u = ((2 * t_c * self.cfg.arm_ratio).exp() - 1) / (far_ratio - 1)

            # Map from t_c to t_u.
            far_u = far_ratio * self.cfg.center_radius
            n = far_u * t_u + self.cfg.center_radius * (1 - t_u)
            d = t_c + self.center_extent * (1 - t_c)
            return xyz * (n / d)[..., None]

        if self.cfg.disable_exponential:
            # For the ablation, just scale linearly.
            result = xyz * exp(2 * self.cfg.arm_ratio)
        else:
            # Separately handle center and other points.
            result = torch.empty_like(xyz)
            result[center_mask] = handle_center(xyz[center_mask])
            result[~center_mask] = handle_other(xyz[~center_mask])

        return result
