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
        str(Path(__file__).parent / "dilate_occupancy.slang"),
        verbose=True,
    )
)


@record_function("dilate_occupancy")
def dilate_occupancy(
    occupancy: Int32[Tensor, "i j k_packed"],
    dilation: int,
) -> Int32[Tensor, "i j k_packed"]:
    if dilation == 0:
        return occupancy.clone()

    # The implementation would have to change for larger dilation values.
    assert 0 < dilation < 32

    new_occupancy = torch.empty_like(occupancy)

    slang().dilate_occupancy(
        occupancy=occupancy,
        new_occupancy=new_occupancy,
        dilation=dilation,
    ).launchRaw(
        blockSize=BLOCK_SIZE,
        gridSize=tuple(
            ceildiv(dim, block) for dim, block in zip(occupancy.shape, BLOCK_SIZE)
        ),
    )

    return new_occupancy
