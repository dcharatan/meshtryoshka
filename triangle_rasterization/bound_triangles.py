from dataclasses import dataclass
from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float32, Int32
from torch import Tensor

from .common import TileGrid, ceildiv, record_function
from .compilation import wrap_compilation

slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "bound_triangles.slang"),
        verbose=True,
    )
)


BLOCK_SIZE = 256


@dataclass
class TriangleBounds:
    # Minimum tile indices (inclusive) that each triangle overlaps.
    tile_minima: Int32[Tensor, "batch triangle xy=2"]

    # Maximum tile indices (exclusive) that each triangle overlaps.
    tile_maxima: Int32[Tensor, "batch triangle xy=2"]

    # The number of tiles each triangle overlaps.
    num_tiles_overlapped: Int32[Tensor, "batch triangle"]


@record_function("bound_triangles")
def bound_triangles(
    vertices: Float32[Tensor, "batch vertex xy=2"],
    depths: Float32[Tensor, "batch vertex"],
    faces: Int32[Tensor, "triangle corner=3"],
    grid: TileGrid,
    near_plane: float,
) -> TriangleBounds:
    assert near_plane >= 0
    device = vertices.device
    c, _, _ = vertices.shape
    t, _ = faces.shape
    kwargs = {"device": device, "dtype": torch.int32}
    out_tile_minima = torch.empty((c, t, 2), **kwargs)
    out_tile_maxima = torch.empty((c, t, 2), **kwargs)
    out_num_tiles_touched = torch.empty((c, t), **kwargs)

    slang().bound_triangles(
        vertices=vertices,
        depths=depths,
        faces=faces,
        grid_tile_height=grid.tile_shape[0],
        grid_tile_width=grid.tile_shape[1],
        grid_row_minimum=0,
        grid_col_minimum=0,
        grid_row_maximum=grid.grid_shape[0],
        grid_col_maximum=grid.grid_shape[1],
        near_plane=near_plane,
        out_tile_minima=out_tile_minima,
        out_tile_maxima=out_tile_maxima,
        out_num_tiles_touched=out_num_tiles_touched,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(t, BLOCK_SIZE), 1, 1),
    )

    return TriangleBounds(out_tile_minima, out_tile_maxima, out_num_tiles_touched)
