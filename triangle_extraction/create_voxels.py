from pathlib import Path
from typing import NamedTuple

import slangtorch
import torch
from jaxtyping import Float32, Int32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

# Since occupancies are packed along the last dimension, each block effectively operates
# on a (32, 32, 32) chunk of occupancies with a (32, 32, 1) block size.
BLOCK_SIZE = (32, 32, 1)


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "create_voxels.slang"),
        verbose=True,
    )
)


class Voxels(NamedTuple):
    vertices: Float32[Tensor, "vertex xyz=3"]
    neighbors: Int32[Tensor, "neighbor=7 voxel"]
    lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"]
    upper_corners: Int32[Tensor, "corner=4 voxel"]
    indices: Int32[Tensor, " voxel"]
    grid_shape: tuple[int, int, int]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_voxels(self) -> int:
        return self.upper_corners.shape[1]

    @property
    def num_subvoxels(self) -> int:
        return self.lower_corners.shape[1] - self.upper_corners.shape[1]


@record_function("create_voxels")
def create_voxels(
    occupancy: Int32[Tensor, "i j k_packed"],
    voxel_offsets: Int32[Tensor, "i+1 j+1 k_packed+1"],
    num_voxels: int,
    vertex_occupancy: Int32[Tensor, "i+1 j+1 k_packed+1"],
    vertex_offsets: Int32[Tensor, "i+1 j+1 k_packed+1"],
    num_vertices: int,
) -> Voxels:
    device = occupancy.device

    # There are three quantities we care about:
    # - Voxels (used to create triangle vertices and triangle faces)
    # - Subvoxels (only used to create triangle vertices)
    # - Vertices (used to sample the field)
    # The number of vertices is equal to the number of voxels plus the number of
    # subvoxels. Hence, the number of lower corners is the number of vertices.
    vertices = torch.empty((num_vertices, 3), dtype=torch.float32, device=device)
    neighbors = torch.empty((7, num_voxels), dtype=torch.int32, device=device)
    lower_corners = torch.empty((4, num_vertices), dtype=torch.int32, device=device)
    upper_corners = torch.empty((4, num_voxels), dtype=torch.int32, device=device)
    indices = torch.empty((num_voxels,), dtype=torch.int32, device=device)

    if num_voxels > 0:
        slang().create_voxels(
            voxel_occupancy=occupancy,
            voxel_offsets=voxel_offsets,
            vertex_occupancy=vertex_occupancy,
            vertex_offsets=vertex_offsets,
            vertices=vertices,
            neighbors=neighbors,
            lower_corners=lower_corners,
            upper_corners=upper_corners,
            indices=indices,
        ).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=tuple(
                ceildiv(dim, block)
                for dim, block in zip(vertex_occupancy.shape, BLOCK_SIZE)
            ),
        )

    i, j, k_packed = occupancy.shape
    return Voxels(
        vertices,
        neighbors,
        lower_corners,
        upper_corners,
        indices,
        (i, j, k_packed * 32),
    )
