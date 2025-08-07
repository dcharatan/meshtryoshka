from math import prod
from pathlib import Path
from typing import NamedTuple

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
        str(Path(__file__).parent / "compute_vertex_occupancy.slang"),
        verbose=True,
    )
)


class VertexOccupancyFnResult(NamedTuple):
    vertex_occupancy: Int32[Tensor, "i+1 j+1 k_packed+1"]

    # The first n - 1 entries should be row-major to match the occupancy above.
    vertex_counts: Int32[Tensor, " (i+1)*(j+1)*(k_packed+1)+1"]


@record_function("compute_vertex_occupancy")
def compute_vertex_occupancy(
    occupancy: Int32[Tensor, "i j k_packed"],
) -> VertexOccupancyFnResult:
    device = occupancy.device
    i, j, k_packed = occupancy.shape
    kwargs = dict(dtype=torch.int32, device=device)

    vertex_occupancy_shape = (i + 1, j + 1, k_packed + 1)
    vertex_occupancy = torch.empty(vertex_occupancy_shape, **kwargs)
    vertex_counts = torch.empty((prod(vertex_occupancy_shape) + 1,), **kwargs)

    slang().compute_vertex_occupancy(
        occupancy=occupancy,
        vertex_occupancy=vertex_occupancy,
        vertex_counts=vertex_counts[:-1].view(vertex_occupancy_shape),
    ).launchRaw(
        blockSize=BLOCK_SIZE,
        gridSize=tuple(
            ceildiv(dim, block)
            for dim, block in zip(vertex_occupancy_shape, BLOCK_SIZE)
        ),
    )

    # The last (extra) vertex count isn't touched by the above kernel, so we zero it out
    # separately here.
    vertex_counts[-1] = 0

    return VertexOccupancyFnResult(vertex_occupancy, vertex_counts)
