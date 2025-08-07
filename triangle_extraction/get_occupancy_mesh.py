from math import prod
from pathlib import Path
from typing import NamedTuple

import slangtorch
import torch
from jaxtyping import Float, Int32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation
from .compute_exclusive_cumsum import compute_exclusive_cumsum

# Since occupancies are packed along the last dimension, each block effectively operates
# on a (32, 32, 32) chunk of occupancies with a (32, 32, 1) block size.
BLOCK_SIZE = (32, 32, 1)


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "get_occupancy_mesh.slang"),
        verbose=True,
    )
)


class Mesh(NamedTuple):
    vertices: Float[Tensor, "triangle corner=3 xyz=3"]
    colors: Float[Tensor, "triangle rgb=3"]


@record_function("get_occupancy_mesh")
def get_occupancy_mesh(
    occupancy: Int32[Tensor, "i j k_packed"],
    colors: Float[Tensor, "axis=3 side=2 rgb=3"],
) -> Mesh:
    # First, count the number of triangles created by each voxel. Each voxel has 6
    # faces, of which 3 have a smaller coordinate value in one dimension. Each voxel
    # "owns" these 3 faces. Since a face consists of 2 triangles, each voxel can create
    # up to 6 triangles.
    i, j, k_packed = occupancy.shape
    shape = (i + 1, j + 1, k_packed + 1)
    triangle_counts = torch.zeros(
        (prod(shape) + 1,),
        dtype=torch.int32,
        device=occupancy.device,
    )

    # First, count the number of triangles created by each packed occupancy vector.
    slang().count_triangles(
        occupancy=occupancy,
        triangle_counts=triangle_counts[:-1].view(shape),
    ).launchRaw(
        blockSize=BLOCK_SIZE,
        gridSize=tuple(ceildiv(dim, block) for dim, block in zip(shape, BLOCK_SIZE)),
    )

    # Next, use a cumulative sum to convert the counts to offsets.
    compute_exclusive_cumsum(triangle_counts)
    num_triangles = triangle_counts[-1].item()

    # Finally, actually create the triangles.
    kwargs = dict(dtype=torch.float32, device=occupancy.device)
    triangle_vertices = torch.empty((num_triangles, 3, 3), **kwargs)
    triangle_colors = torch.empty((num_triangles, 3), **kwargs)
    if num_triangles > 0:
        slang().create_triangles(
            occupancy=occupancy,
            triangle_offsets=triangle_counts[:-1].view(shape),
            triangle_vertices=triangle_vertices,
            triangle_colors=triangle_colors,
            colors=colors,
        ).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=tuple(
                ceildiv(dim, block) for dim, block in zip(shape, BLOCK_SIZE)
            ),
        )

    return Mesh(triangle_vertices, triangle_colors)
