from pathlib import Path

import slangtorch
import torch
from jaxtyping import Int32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

BLOCK_SIZE = 256


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "write_occupancy.slang"),
        verbose=True,
    )
)


@record_function("write_occupancy")
def write_occupancy(
    voxel_indices: Int32[Tensor, " triangle"],
    packed_occupancy_shape: tuple[int, ...],
    min_shell: int,
    max_shell: int,
) -> Int32[Tensor, "*rest packed"]:
    (num_triangles,) = voxel_indices.shape
    occupancy = torch.zeros(
        packed_occupancy_shape,
        dtype=torch.int32,
        device=voxel_indices.device,
    )

    slang().write_occupancy(
        occupancy=occupancy.view(-1),
        voxel_indices=voxel_indices,
        min_shell=min_shell,
        max_shell=max_shell,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(num_triangles, BLOCK_SIZE), 1, 1),
    )

    return occupancy
