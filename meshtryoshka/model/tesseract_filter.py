import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from triangle_extraction import dilate_occupancy, pack_occupancy

from ..components.projection import project
from ..model.contraction import Contraction
from .tesseract import Tesseract


def sample_voxel_corners(
    resolution: tuple[int, int, int],
    device: torch.device,
) -> Float[Tensor, "i j k xyz=3"]:
    i, j, k = resolution
    x = torch.linspace(0, 1, k + 1, device=device)
    y = torch.linspace(0, 1, j + 1, device=device)
    z = torch.linspace(0, 1, i + 1, device=device)
    return torch.stack(torch.meshgrid(x, y, z, indexing="ij")[::-1], dim=-1)


def get_create_filter(
    threshold: int,
    erosion: int,
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    image_shape: tuple[int, int],
    contraction: Contraction,
    batch_size: int = 4096,
):
    def create_filter(tesseract: Tesseract, step: int):
        device = extrinsics.device
        occupancy = []

        for component in tesseract.components:
            # Get mid-voxel sample points for the grid.
            component_shape = component.shape(step)
            xyz = sample_voxel_corners(component_shape, device)
            xyz = component.interpolate(xyz)
            xyz = contraction.uncontract(xyz)
            i, j, k, _ = xyz.shape
            split_size = max(1, (batch_size // (j * k)) - 1)

            # Iterate through batches of sample points.
            h, w = image_shape
            image_shape_xy = torch.tensor((w, h), dtype=torch.int32, device=device)
            mask = []
            for start in range(0, i, split_size):
                # Project the sample points onto each camera. A sample point is valid
                # for a particular camera if it's in front of the camera and falls
                # within bounds. A voxel is valid if all of its corners (the sample
                # points) are valid. If the number of cameras for which a particular
                # voxel is valid is grater than or equal to the threshold, then the
                # voxel is included in the filter grid.
                xyz_batch = xyz[start : start + split_size + 1]
                xy, valid = project(
                    xyz_batch,
                    rearrange(extrinsics, "b i j -> b () () () i j"),
                    rearrange(intrinsics, "b i j -> b () () () i j"),
                )
                in_bounds = ((xy >= 0) & (xy <= image_shape_xy)).all(dim=-1)
                valid = valid & in_bounds & ~xyz_batch.isinf().any(dim=-1)

                # Collapse samples to voxels.
                valid = (
                    valid[:, 1:, 1:, 1:]
                    & valid[:, 1:, 1:, :-1]
                    & valid[:, 1:, :-1, 1:]
                    & valid[:, 1:, :-1, :-1]
                    & valid[:, :-1, 1:, 1:]
                    & valid[:, :-1, 1:, :-1]
                    & valid[:, :-1, :-1, 1:]
                    & valid[:, :-1, :-1, :-1]
                )

                # Check against the visibility threshold.
                mask_batch = valid.sum(dim=0) >= threshold
                mask.append(mask_batch)

            # Reassemble the mask into a 3D grid.
            component_occupancy = pack_occupancy(torch.cat(mask))

            # Optionally erode the occupancy by dilating its inverse.
            if erosion > 0:
                all_bits_one = torch.full_like(component_occupancy, -1)
                component_occupancy = component_occupancy ^ all_bits_one
                component_occupancy = dilate_occupancy(component_occupancy, erosion)
                component_occupancy = component_occupancy ^ all_bits_one

            occupancy.append(component_occupancy.reshape(-1))

        return torch.cat(occupancy, dim=0)

    return create_filter
