from math import prod
from pathlib import Path

import slangtorch
import torch
from jaxtyping import Int32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

# Since occupancies are packed along the last dimension, each block effectively operates
# on a (32, 32, 32) chunk of occupancies with a (32, 32, 1) block size.
BLOCK_SIZE = (32, 32, 1)


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "count_occupancy.slang"),
        verbose=True,
    )
)


@record_function("count_occupancy")
def count_occupancy(
    occupancy: Int32[Tensor, "i j k_packed"],
) -> Int32[Tensor, " (i+1)*(j+1)*(k_packed+1)+1"]:
    device = occupancy.device
    i, j, k_packed = occupancy.shape
    shape = (i + 1, j + 1, k_packed + 1)
    voxel_counts = torch.zeros((prod(shape) + 1,), dtype=torch.int32, device=device)

    slang().count_occupancy(
        occupancy=occupancy,
        voxel_counts=voxel_counts[:-1].view(shape),
    ).launchRaw(
        blockSize=BLOCK_SIZE,
        gridSize=tuple(
            ceildiv(dim, block) for dim, block in zip(occupancy.shape, BLOCK_SIZE)
        ),
    )

    return voxel_counts
