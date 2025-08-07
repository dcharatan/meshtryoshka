from pathlib import Path
from typing import NamedTuple

import slangtorch
import torch
from jaxtyping import Float32, Int32, Int64
from torch import Tensor

from .common import TileGrid, record_function
from .compilation import wrap_compilation

slang = wrap_compilation(
    lambda tile_height, tile_width, key_defines: slangtorch.loadModule(
        str(Path(__file__).parent / "intersect.slang"),
        defines={
            "TILE_HEIGHT": tile_height,
            "TILE_WIDTH": tile_width,
            **dict(key_defines),
        },
        verbose=True,
    )
)


class RenderedImages(NamedTuple):
    # barycentric coordinates for the triangle that was hit
    uv: Float32[Tensor, "batch shell height width uv=2"]

    # the index of the triangle that was hit
    index: Int32[Tensor, "batch shell height width"]


@record_function("intersect")
def intersect(
    vertices: Float32[Tensor, "batch vertex xy=2"],
    faces: Int32[Tensor, "face corner=3"],
    sorted_triangle_indices: Int32[Tensor, " key"],
    tile_boundaries: Int64[Tensor, "batch shell tile boundary=2"],
    grid: TileGrid,
    key_defines: tuple[tuple[str, int], ...],
) -> RenderedImages:
    device = vertices.device
    h, w = grid.image_shape
    out_shape = (grid.num_cameras * grid.num_shells, h, w)

    out_uv = torch.empty((*out_shape, 2), dtype=torch.float32, device=device)
    out_index = torch.empty(out_shape, dtype=torch.int32, device=device)

    th, tw = grid.tile_shape
    slang(th, tw, key_defines).intersect(
        sorted_triangle_indices=sorted_triangle_indices,
        tile_boundaries=tile_boundaries,
        vertices=vertices,
        faces=faces,
        grid_num_rows=grid.grid_shape[0],
        grid_num_cols=grid.grid_shape[1],
        image_height=h,
        image_width=w,
        out_uv=out_uv,
        out_index=out_index,
    ).launchRaw(
        blockSize=(grid.tile_numel, 1, 1),
        gridSize=(grid.num_tiles_per_image * grid.num_cameras * grid.num_shells, 1, 1),
    )

    return_shape = (grid.num_cameras, grid.num_shells, h, w)
    return RenderedImages(
        uv=out_uv.view((*return_shape, 2)),
        index=out_index.view(return_shape),
    )
