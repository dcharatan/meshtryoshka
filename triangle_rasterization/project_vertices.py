from pathlib import Path
from typing import NamedTuple

import slangtorch
import torch
from jaxtyping import Float32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "project_vertices.slang"),
        verbose=True,
    )
)


BLOCK_SIZE = 256


class ProjectedVertices(NamedTuple):
    # Projected 2D vertex locations in pixel space (unnormalized).
    positions: Float32[Tensor, "batch vertex xy=2"]

    # Camera-space depths for each vertex.
    depths: Float32[Tensor, "batch vertex"]


@record_function("project_vertices")
def project_vertices(
    vertices: Float32[Tensor, "vertex xyz=3"],
    extrinsics: Float32[Tensor, "batch 4 4"],
    intrinsics: Float32[Tensor, "batch 3 3"],
) -> ProjectedVertices:
    device = vertices.device
    b, _, _ = extrinsics.shape
    v, _ = vertices.shape
    kwargs = {"device": device, "dtype": torch.float32}

    out_vertices = torch.empty((b, v, 2), **kwargs)
    out_depths = torch.empty((b, v), **kwargs)

    slang().project_vertices(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        vertices=vertices,
        out_vertices=out_vertices,
        out_depths=out_depths,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(v, BLOCK_SIZE), 1, 1),
    )

    return ProjectedVertices(out_vertices, out_depths)
