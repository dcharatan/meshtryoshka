from pathlib import Path
from typing import NamedTuple

import slangtorch
import torch
from jaxtyping import Float32, Int32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation
from .compute_exclusive_cumsum import compute_exclusive_cumsum
from .compute_vertex_occupancy import compute_vertex_occupancy
from .upscale_occupancy import upscale_occupancy

# Since occupancies are packed along the last dimension, each block effectively operates
# on a (32, 32, 32) chunk of occupancies with a (32, 32, 1) block size.
BLOCK_SIZE = (32, 32, 1)


slang = wrap_compilation(
    lambda num_spherical_harmonics: slangtorch.loadModule(
        str(Path(__file__).parent / "upscale_parameters.slang"),
        defines={"NUM_SPHERICAL_HARMONICS": num_spherical_harmonics},
        verbose=True,
    )
)


class UpscaledParameters(NamedTuple):
    signed_distances: Float32[Tensor, " vertex"]
    spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"]
    occupancy: Int32[Tensor, "i j k_packed"]


@record_function("upscale_parameters")
@torch.no_grad()
def upscale_parameters(
    old_signed_distances: Float32[Tensor, " vertex"],
    old_spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
    old_occupancy: Int32[Tensor, "i j k_packed"],
    occupancy_to_upscale: Int32[Tensor, "i j k_packed"],
    new_resolution: tuple[int, int, int],
) -> UpscaledParameters:
    device = old_signed_distances.device

    # Get the upscaled occupancy grid. Note that because we need at least one value to
    # interpolate at each upscaled vertex, we have to ensure that the occupancy to
    # upscale is a subset of the old occupancy grid.
    occupancy_to_upscale = old_occupancy & occupancy_to_upscale
    new_occupancy = upscale_occupancy(occupancy_to_upscale, new_resolution)

    # Compute vertex occupancy and vertex offsets for both occupancy grids. These will
    # be needed to look up (i, j, k) -> vertex index during parameter upscaling.
    old_vertex_occupancy, old_vertex_offsets = compute_vertex_occupancy(old_occupancy)
    compute_exclusive_cumsum(old_vertex_offsets)
    old_vertex_offsets = old_vertex_offsets[:-1].view(old_vertex_occupancy.shape)

    new_vertex_occupancy, new_vertex_offsets = compute_vertex_occupancy(new_occupancy)
    compute_exclusive_cumsum(new_vertex_offsets)
    num_vertices = new_vertex_offsets[-1].item()
    new_vertex_offsets = new_vertex_offsets[:-1].view(new_vertex_occupancy.shape)

    # Allocate space for the upscaled parameters.
    new_signed_distances = torch.empty(
        (num_vertices,),
        dtype=torch.float32,
        device=device,
    )
    sh, _, _ = old_spherical_harmonics.shape
    new_spherical_harmonics = torch.empty(
        (sh, num_vertices, 3),
        dtype=torch.float32,
        device=device,
    )

    # Do the actual upscaling.
    if num_vertices > 0:
        slang(sh).upscale_parameters(
            old_vertex_occupancy=old_vertex_occupancy,
            old_vertex_offsets=old_vertex_offsets,
            old_signed_distances=old_signed_distances,
            old_spherical_harmonics=old_spherical_harmonics,
            new_vertex_occupancy=new_vertex_occupancy,
            new_vertex_offsets=new_vertex_offsets,
            new_signed_distances=new_signed_distances,
            new_spherical_harmonics=new_spherical_harmonics,
        ).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=tuple(
                ceildiv(dim, block)
                for dim, block in zip(new_vertex_occupancy.shape, BLOCK_SIZE)
            ),
        )

    return UpscaledParameters(
        new_signed_distances,
        new_spherical_harmonics,
        new_occupancy,
    )
