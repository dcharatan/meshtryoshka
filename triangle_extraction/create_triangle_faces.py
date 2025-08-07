from pathlib import Path
from typing import NamedTuple

import slangtorch
import torch
from jaxtyping import Float, Int32, UInt8
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

BLOCK_SIZE = 256


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "create_triangle_faces.slang"),
        verbose=True,
    )
)


class CreateTriangleFacesResult(NamedTuple):
    faces: Int32[Tensor, "triangle_face corner=3"]
    voxel_indices: Int32[Tensor, " triangle_face"] | None


@record_function("create_triangle_faces")
def create_triangle_faces(
    signed_distances: Float[Tensor, " sample"],
    neighbors: Int32[Tensor, "neighbor=7 voxel"],
    indices: Int32[Tensor, " voxel"],
    vertex_offsets: Int32[Tensor, " _"],  # (level_set * voxel_and_subvoxel) + 1
    triangle_vertex_types: UInt8[Tensor, " triangle_vertex"],
    voxel_cell_codes: UInt8[Tensor, "level_set voxel"],
    level_sets: Float[Tensor, " level_set"],
    face_offsets: Int32[Tensor, " level_set*voxel+1"],
) -> CreateTriangleFacesResult:
    device = signed_distances.device
    _, num_voxels = neighbors.shape
    num_faces = face_offsets[-1].item() // 3
    (num_level_sets,) = level_sets.shape
    num_voxels_and_subvoxels = (vertex_offsets.shape[0] - 1) // num_level_sets

    triangle_faces = torch.empty((num_faces, 3), dtype=torch.int32, device=device)
    voxel_indices = torch.empty((num_faces,), dtype=torch.int32, device=device)

    slang().create_triangle_faces(
        signed_distances=signed_distances,
        neighbors=neighbors,
        indices=indices,
        vertex_offsets=vertex_offsets,
        triangle_vertex_types=triangle_vertex_types,
        voxel_cell_codes=voxel_cell_codes,
        level_sets=level_sets,
        triangle_faces=triangle_faces,
        voxel_indices=voxel_indices,
        num_voxels_and_subvoxels=num_voxels_and_subvoxels,
        face_offsets=face_offsets,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
    )

    return CreateTriangleFacesResult(triangle_faces, voxel_indices)
