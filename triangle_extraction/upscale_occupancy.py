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
        str(Path(__file__).parent / "upscale_occupancy.slang"),
        verbose=True,
    )
)


@record_function("upscale_occupancy")
def upscale_occupancy(
    occupancy: Int32[Tensor, "i j k_packed"],
    target_shape: tuple[int, int, int],
) -> Int32[Tensor, "i_target j_target k_packed_target"]:
    # This will only work if the new occupancy grid has a higher resolution than the old
    # occupancy grid.
    i, j, k_packed = occupancy.shape
    i_target, j_target, k_target = target_shape
    assert k_target % 32 == 0
    k_packed_target = k_target // 32
    assert i_target >= i and j_target >= j and k_packed_target >= k_packed

    new_occupancy = torch.empty(
        (i_target, j_target, k_packed_target),
        dtype=torch.int32,
        device=occupancy.device,
    )

    slang().upscale_occupancy(
        occupancy=occupancy,
        new_occupancy=new_occupancy,
    ).launchRaw(
        blockSize=BLOCK_SIZE,
        gridSize=tuple(
            ceildiv(dim, block) for dim, block in zip(new_occupancy.shape, BLOCK_SIZE)
        ),
    )

    return new_occupancy
