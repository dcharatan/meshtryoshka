from pathlib import Path
from typing import NamedTuple

import slangtorch
import torch
from jaxtyping import Float32, Int32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

BLOCK_SIZE = 256

slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "compute_sdf_regularizers.slang"),
        verbose=True,
    )
)


class SDFRegularizerLosses(NamedTuple):
    eikonal_loss_pos: Float32[Tensor, " voxel"]
    eikonal_loss_neg: Float32[Tensor, " voxel"]
    curvature_loss: Float32[Tensor, " voxel"]


class ComputeSDFRegularizers(torch.autograd.Function):
    @record_function("compute_sdf_regularizers_forward")
    @staticmethod
    def forward(
        ctx,
        grid_signed_distances: Float32[Tensor, " sample"],
        voxel_neighbors: Int32[Tensor, "neighbor=7 voxel"],
        voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
        voxel_upper_corners: Int32[Tensor, "corner=4 voxel"],
        grid_size: tuple[int, int, int],
    ) -> tuple[
        Float32[Tensor, " voxel"],
        Float32[Tensor, " voxel"],
        Float32[Tensor, " voxel"],
    ]:
        device = grid_signed_distances.device

        # Allocate space for the output.
        _, num_voxels = voxel_upper_corners.shape
        eikonal_loss_pos = torch.empty(
            (num_voxels,), dtype=torch.float32, device=device
        )
        eikonal_loss_neg = torch.empty(
            (num_voxels,), dtype=torch.float32, device=device
        )
        curvature_loss = torch.empty((num_voxels,), dtype=torch.float32, device=device)

        # Call the Slang kernel.
        i, j, k = grid_size
        slang().compute_sdf_regularizers(
            grid_signed_distances=grid_signed_distances,
            voxel_neighbors=voxel_neighbors,
            voxel_lower_corners=voxel_lower_corners,
            voxel_upper_corners=voxel_upper_corners,
            voxel_x=k,
            voxel_y=j,
            voxel_z=i,
            eikonal_loss_pos=eikonal_loss_pos,
            eikonal_loss_neg=eikonal_loss_neg,
            curvature_loss=curvature_loss,
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
        )

        # Save tensors needed for the backward pass.
        ctx.save_for_backward(
            grid_signed_distances,
            voxel_neighbors,
            voxel_lower_corners,
            voxel_upper_corners,
            eikonal_loss_pos,
            eikonal_loss_neg,
            curvature_loss,
        )
        ctx.grid_size = grid_size

        return eikonal_loss_pos, eikonal_loss_neg, curvature_loss

    @record_function("compute_sdf_regularizers_backward")
    @staticmethod
    def backward(
        ctx,
        grad_eikonal_loss_pos,
        grad_eikonal_loss_neg,
        grad_curvature_loss,
    ):
        # Retrieve information from the context.
        (
            grid_signed_distances,
            voxel_neighbors,
            voxel_lower_corners,
            voxel_upper_corners,
            eikonal_loss_pos,
            eikonal_loss_neg,
            curvature_loss,
        ) = ctx.saved_tensors
        i, j, k = ctx.grid_size
        _, num_voxels = voxel_upper_corners.shape

        # Allocate space for the inputs' gradients.
        grad_grid_signed_distances = torch.zeros_like(grid_signed_distances)
        grad_eikonal_loss_pos = grad_eikonal_loss_pos.contiguous()
        grad_eikonal_loss_neg = grad_eikonal_loss_neg.contiguous()
        grad_curvature_loss = grad_curvature_loss.contiguous()

        # Call the Slang kernel.
        slang().compute_sdf_regularizers.bwd(
            grid_signed_distances=(grid_signed_distances, grad_grid_signed_distances),
            voxel_neighbors=voxel_neighbors,
            voxel_lower_corners=voxel_lower_corners,
            voxel_upper_corners=voxel_upper_corners,
            voxel_x=k,
            voxel_y=j,
            voxel_z=i,
            eikonal_loss_pos=(eikonal_loss_pos, grad_eikonal_loss_pos),
            eikonal_loss_neg=(eikonal_loss_neg, grad_eikonal_loss_neg),
            curvature_loss=(curvature_loss, grad_curvature_loss),
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
        )

        return grad_grid_signed_distances, None, None, None, None, None


def compute_sdf_regularizers(
    grid_signed_distances: Float32[Tensor, " sample"],
    voxel_neighbors: Int32[Tensor, "neighbor=7 voxel"],
    voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    voxel_upper_corners: Int32[Tensor, "corner=4 voxel"],
    grid_size: tuple[int, int, int],
) -> SDFRegularizerLosses:
    return SDFRegularizerLosses(
        *ComputeSDFRegularizers.apply(
            grid_signed_distances,
            voxel_neighbors,
            voxel_lower_corners,
            voxel_upper_corners,
            grid_size,
        )
    )
