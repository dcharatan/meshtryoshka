from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float32, Int32, Int64
from torch import Tensor

from .common import TileGrid, record_function
from .compilation import wrap_compilation

slang = wrap_compilation(
    lambda grid, key_defines: slangtorch.loadModule(
        str(Path(__file__).parent / "intersect_and_composite.slang"),
        defines={
            "TILE_HEIGHT": grid.tile_shape[0],
            "TILE_WIDTH": grid.tile_shape[1],
            **dict(key_defines),
        },
        verbose=True,
    )
)


class IntersectAndComposite(torch.autograd.Function):
    @record_function("intersect_and_composite_forward")
    @staticmethod
    def forward(
        ctx,
        vertices: Float32[Tensor, "batch vertex xy=2"],
        colors: Float32[Tensor, "batch vertex rgb=3"],
        signed_distances: Float32[Tensor, " vertex"],
        faces: Int32[Tensor, "triangle corner=3"],
        sharpness: Float32[Tensor, ""],
        sorted_keys: Int64[Tensor, " key"],
        sorted_triangle_indices: Int32[Tensor, " key"],
        tile_boundaries: Int64[Tensor, "batch tile boundary=2"],
        num_shells: int,
        grid: TileGrid,
        key_defines: tuple[tuple[str, int], ...],
    ) -> Float32[Tensor, "batch rgba=4 height width"]:
        device = vertices.device

        b, _, _ = vertices.shape
        h, w = grid.image_shape
        out_image = torch.empty((b, 4, h, w), dtype=torch.float32, device=device)

        slang(grid, key_defines).intersect_and_composite_forward(
            vertices=vertices,
            colors=colors,
            signed_distances=signed_distances,
            faces=faces,
            sharpness=sharpness,
            sorted_keys=sorted_keys,
            sorted_triangle_indices=sorted_triangle_indices,
            tile_boundaries=tile_boundaries,
            num_shells=num_shells,
            grid_num_rows=grid.grid_shape[0],
            grid_num_cols=grid.grid_shape[1],
            out_accumulators=out_image,
        ).launchRaw(
            blockSize=(grid.tile_numel, 1, 1),
            gridSize=(grid.num_tiles_per_image * grid.num_cameras, 1, 1),
        )

        ctx.save_for_backward(
            vertices,
            colors,
            signed_distances,
            faces,
            sharpness,
            sorted_keys,
            sorted_triangle_indices,
            tile_boundaries,
            out_image,
        )
        ctx.num_shells = num_shells
        ctx.grid = grid
        ctx.key_defines = key_defines

        return out_image

    @record_function("intersect_and_composite_backward")
    @staticmethod
    def backward(
        ctx,
        out_image_grad: Float32[Tensor, "batch rgba=4 height width"],
    ) -> tuple[
        None,
        Float32[Tensor, "batch vertex rgb=3"],
        Float32[Tensor, " vertex"],
        None,
        Float32[Tensor, ""],
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            vertices,
            colors,
            signed_distances,
            faces,
            sharpness,
            sorted_keys,
            sorted_triangle_indices,
            tile_boundaries,
            out_image,
        ) = ctx.saved_tensors
        grid: TileGrid = ctx.grid

        colors_grad = torch.zeros_like(colors)
        signed_distances_grad = torch.zeros_like(signed_distances)
        sharpness_grad = torch.zeros_like(sharpness)

        slang(grid, ctx.key_defines).intersect_and_composite_backward(
            vertices=vertices,
            colors=(colors, colors_grad),
            signed_distances=(signed_distances, signed_distances_grad),
            faces=faces,
            sharpness=(sharpness, sharpness_grad),
            sorted_keys=sorted_keys,
            sorted_triangle_indices=sorted_triangle_indices,
            tile_boundaries=tile_boundaries,
            num_shells=ctx.num_shells,
            grid_num_rows=grid.grid_shape[0],
            grid_num_cols=grid.grid_shape[1],
            out_accumulators=(out_image, out_image_grad),
        ).launchRaw(
            blockSize=(grid.tile_numel, 1, 1),
            gridSize=(grid.num_tiles_per_image * grid.num_cameras, 1, 1),
        )

        return (
            None,
            colors_grad,
            signed_distances_grad,
            None,
            sharpness_grad,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def intersect_and_composite(
    vertices: Float32[Tensor, "batch vertex xy=2"],
    colors: Float32[Tensor, "batch vertex rgb=3"],
    signed_distances: Float32[Tensor, " vertex"],
    faces: Int32[Tensor, "triangle corner=3"],
    sharpness: Float32[Tensor, ""],
    sorted_keys: Int64[Tensor, " key"],
    sorted_triangle_indices: Int32[Tensor, " key"],
    tile_boundaries: Int64[Tensor, "batch tile boundary=2"],
    num_shells: int,
    grid: TileGrid,
    key_defines: tuple[tuple[str, int], ...],
) -> Float32[Tensor, "batch rgba=4 height width"]:
    return IntersectAndComposite.apply(
        vertices,
        colors,
        signed_distances,
        faces,
        sharpness,
        sorted_keys,
        sorted_triangle_indices,
        tile_boundaries,
        num_shells,
        grid,
        key_defines,
    )
