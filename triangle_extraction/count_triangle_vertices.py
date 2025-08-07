from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float32, Int32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

BLOCK_SIZE = 256


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "count_triangle_vertices.slang"),
        verbose=True,
    )
)


@record_function("count_triangle_vertices")
def count_triangle_vertices(
    signed_distances: Float32[Tensor, " sample"],
    lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    level_sets: Float32[Tensor, " level_set"],
) -> Int32[Tensor, " level_set*voxel_and_subvoxel+1"]:
    device = signed_distances.device
    (num_level_sets,) = level_sets.shape
    _, num_voxels = lower_corners.shape
    vertex_counts_by_level_set = torch.empty(
        (num_level_sets * num_voxels + 1,),
        dtype=torch.int32,
        device=device,
    )
    vertex_counts_by_level_set[-1] = 0

    slang().countTriangleVertices(
        signed_distances=signed_distances,
        lower_corners=lower_corners,
        level_sets=level_sets,
        vertex_counts_by_level_set=vertex_counts_by_level_set[:-1].view(
            num_level_sets, -1
        ),
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
    )

    return vertex_counts_by_level_set
