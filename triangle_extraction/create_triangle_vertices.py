from pathlib import Path
from typing import NamedTuple

import slangtorch
import torch
from jaxtyping import Float32, Int32, UInt8
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

BLOCK_SIZE = 256

slang = wrap_compilation(
    lambda num_spherical_harmonics: slangtorch.loadModule(
        str(Path(__file__).parent / "create_triangle_vertices.slang"),
        defines={"NUM_SPHERICAL_HARMONICS": num_spherical_harmonics},
        verbose=True,
    )
)


class CreateTriangleVerticesResult(NamedTuple):
    vertices: Float32[Tensor, "triangle_vertex xyz=3"]
    signed_distances: Float32[Tensor, " triangle_vertex"]
    spherical_harmonics: Float32[Tensor, "sh triangle_vertex rgb=3"]
    vertex_types: UInt8[Tensor, " triangle_vertex"]


class CreateTriangleVertices(torch.autograd.Function):
    @record_function("create_triangle_vertices_forward")
    @staticmethod
    def forward(
        ctx,
        grid_vertices: Float32[Tensor, "sample xyz=3"],
        grid_signed_distances: Float32[Tensor, " sample"],
        grid_spherical_harmonics: Float32[Tensor, "sh sample rgb=3"],
        voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
        level_sets: Float32[Tensor, " level_set"],
        vertex_offsets: Int32[Tensor, " level_set*voxel_and_subvoxel+1"],
    ):
        device = grid_vertices.device

        # Determine number of voxels and total vertices.
        (num_level_sets,) = level_sets.shape
        _, num_voxels = voxel_lower_corners.shape
        num_spherical_harmonics, _, _ = grid_spherical_harmonics.shape
        num_vertices = vertex_offsets[-1].item()

        # Allocate outputs.
        float_kwargs = dict(dtype=torch.float32, device=device)
        triangle_vertices = torch.empty((num_vertices, 3), **float_kwargs)
        triangle_signed_distances = torch.empty((num_vertices,), **float_kwargs)
        triangle_vertex_types = torch.empty(
            (num_vertices,),
            dtype=torch.uint8,
            device=device,
        )

        triangle_spherical_harmonics = torch.empty(
            (num_spherical_harmonics, num_vertices, 3),
            dtype=torch.float32,
            device=device,
        )

        # Call the forward kernel (the actual GPU launch)
        slang(num_spherical_harmonics).create_triangle_vertices_forward(
            grid_vertices=grid_vertices,
            grid_signed_distances=grid_signed_distances,
            grid_spherical_harmonics=grid_spherical_harmonics,
            voxel_lower_corners=voxel_lower_corners,
            vertex_offsets=vertex_offsets[:-1].view(num_level_sets, -1),
            level_sets=level_sets,
            triangle_vertices=triangle_vertices,
            triangle_signed_distances=triangle_signed_distances,
            triangle_spherical_harmonics=triangle_spherical_harmonics,
            triangle_vertex_types=triangle_vertex_types,
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
        )

        # Save tensors needed for the backward pass.
        ctx.save_for_backward(
            grid_vertices,
            grid_signed_distances,
            grid_spherical_harmonics,
            voxel_lower_corners,
            vertex_offsets,
            level_sets,
            triangle_vertices,
            triangle_signed_distances,
            triangle_spherical_harmonics,
            triangle_vertex_types,
        )

        return (
            triangle_vertices,
            triangle_signed_distances,
            triangle_spherical_harmonics,
            triangle_vertex_types,
        )

    @record_function("create_triangle_vertices_backward")
    @staticmethod
    def backward(
        ctx,
        grad_triangle_vertices,
        grad_triangle_signed_distances,
        grad_triangle_spherical_harmonics,
        grad_triangle_vertex_types,
    ):
        # Retrieve saved tensors.
        (
            grid_vertices,
            grid_signed_distances,
            grid_spherical_harmonics,
            voxel_lower_corners,
            vertex_offsets,
            level_sets,
            triangle_vertices,
            triangle_signed_distances,
            triangle_spherical_harmonics,
            triangle_vertex_types,
        ) = ctx.saved_tensors
        _, num_voxels = voxel_lower_corners.shape
        (num_level_sets,) = level_sets.shape

        # Allocate gradients for the differentiable inputs. Note that using empty_like
        # would not work here!
        grad_grid_vertices = torch.zeros_like(grid_vertices)
        grad_grid_signed_distances = torch.zeros_like(grid_signed_distances)
        grad_grid_spherical_harmonics = torch.zeros_like(grid_spherical_harmonics)

        # Call the backward kernel. Here the kernel expects tuples (primal,
        # differential) for the inputs that need gradients.
        num_spherical_harmonics, _, _ = grid_spherical_harmonics.shape
        slang(num_spherical_harmonics).create_triangle_vertices_backward(
            grid_vertices=(grid_vertices, grad_grid_vertices),
            grid_signed_distances=(grid_signed_distances, grad_grid_signed_distances),
            grid_spherical_harmonics=(
                grid_spherical_harmonics,
                grad_grid_spherical_harmonics,
            ),
            voxel_lower_corners=voxel_lower_corners,
            vertex_offsets=vertex_offsets[:-1].view(num_level_sets, -1),
            level_sets=level_sets,
            triangle_vertices=(triangle_vertices, grad_triangle_vertices),
            triangle_signed_distances=(
                triangle_signed_distances,
                grad_triangle_signed_distances,
            ),
            triangle_spherical_harmonics=(
                triangle_spherical_harmonics,
                grad_triangle_spherical_harmonics,
            ),
            triangle_vertex_types=triangle_vertex_types,
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
        )

        return (
            grad_grid_vertices,
            grad_grid_signed_distances,
            grad_grid_spherical_harmonics,
            None,
            None,
            None,
        )


def create_triangle_vertices(
    grid_vertices: Float32[Tensor, "sample xyz=3"],
    grid_signed_distances: Float32[Tensor, " sample"],
    grid_spherical_harmonics: Float32[Tensor, "sh sample rgb=3"],
    voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    level_sets: Float32[Tensor, " level_set"],
    vertex_offsets: Int32[Tensor, " level_set*voxel_and_subvoxel+1"],
) -> CreateTriangleVerticesResult:
    result = CreateTriangleVertices.apply(
        grid_vertices,
        grid_signed_distances,
        grid_spherical_harmonics,
        voxel_lower_corners,
        level_sets,
        vertex_offsets,
    )
    return CreateTriangleVerticesResult(*result)
