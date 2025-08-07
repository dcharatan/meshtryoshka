import logging
from typing import NamedTuple

import torch
from jaxtyping import Float32, Int32
from torch import Tensor
from torch.profiler import record_function

from .common import pack_occupancy as pack_occupancy
from .common import unpack_occupancy as unpack_occupancy
from .compute_exclusive_cumsum import compute_exclusive_cumsum
from .compute_sdf_regularizers import (
    compute_sdf_regularizers as compute_sdf_regularizers,
)
from .compute_vertex_occupancy import compute_vertex_occupancy
from .count_occupancy import count_occupancy
from .count_triangle_faces import count_triangle_faces
from .count_triangle_vertices import count_triangle_vertices
from .create_triangle_faces import create_triangle_faces
from .create_triangle_vertices import create_triangle_vertices
from .create_voxels import Voxels, create_voxels
from .dilate_occupancy import dilate_occupancy as dilate_occupancy
from .get_occupancy_mesh import get_occupancy_mesh as get_occupancy_mesh
from .index_vertices import index_vertices as index_vertices
from .upscale_occupancy import upscale_occupancy as upscale_occupancy
from .upscale_parameters import UpscaledParameters as UpscaledParameters
from .upscale_parameters import upscale_parameters as upscale_parameters
from .write_occupancy import write_occupancy as write_occupancy


@record_function("extract_voxels")
def extract_voxels(occupancy: Int32[Tensor, "i j k_packed"]) -> Voxels:
    i, j, k_packed = occupancy.shape
    logging.debug(f"Extracting voxels for grid with shape ({i}, {j}, {k_packed * 32}).")

    # Compute vertex occupancy. A voxel is considered vertex-occupied if it or any
    # of its 7 lower neighbors (one index lower in each direction) are occupied.
    # Also compute the number of vertices owned by each voxel. Each vertex-occupied
    # voxel owns one vertex.
    vertex_occupancy, vertex_offsets = compute_vertex_occupancy(occupancy)

    # Compute vertex offsets.
    compute_exclusive_cumsum(vertex_offsets)
    num_vertices = vertex_offsets[-1].item()
    vertex_offsets = vertex_offsets[:-1].view(vertex_occupancy.shape)

    # Compute voxel offsets.
    voxel_offsets = count_occupancy(occupancy)
    compute_exclusive_cumsum(voxel_offsets)
    num_voxels = voxel_offsets[-1].item()
    voxel_offsets = voxel_offsets[:-1].view(vertex_occupancy.shape)

    # Actually create the samples.
    logging.debug(f"Sampled {num_vertices} vertices and {num_voxels} voxels.")
    return create_voxels(
        occupancy,
        voxel_offsets,
        num_voxels,
        vertex_occupancy,
        vertex_offsets,
        num_vertices,
    )


class Triangles(NamedTuple):
    vertices: Float32[Tensor, "vertex xyz=3"]
    signed_distances: Float32[Tensor, " vertex"]
    spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"]
    faces: Int32[Tensor, "face corner=3"]
    voxel_indices: Int32[Tensor, " face"]
    vertex_boundaries: Int32[Tensor, " level_set"]
    face_boundaries: Int32[Tensor, " level_set"]


@record_function("tessellate_voxels")
def tessellate_voxels(
    signed_distances: Float32[Tensor, " sample"],
    vertices: Float32[Tensor, "sample xyz=3"],
    spherical_harmonics: Float32[Tensor, "sh sample rgb=3"],
    neighbors: Int32[Tensor, "neighbor=7 voxel"],
    lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    upper_corners: Int32[Tensor, "corner=4 voxel"],
    indices: Int32[Tensor, " voxel"],
    level_sets: Float32[Tensor, " level_set"],
) -> Triangles:
    (num_level_sets,) = level_sets.shape
    logging.debug(
        f"Tessellating {neighbors.shape[1]} voxels with {num_level_sets} level sets."
    )

    # Non-differentiable operations:
    # Count the number of vertices created by each voxel.
    with torch.no_grad():
        vertex_offsets = count_triangle_vertices(
            signed_distances,
            lower_corners,
            level_sets,
        )
        compute_exclusive_cumsum(vertex_offsets)
        num_vertices = vertex_offsets[-1].item()

    # Differentiable operation:
    # Use the custom differentiable function to compute triangle vertices.
    # triangle_vertices, triangle_vertex_types = CreateTriangleVertices.apply(
    (
        triangle_vertices,
        triangle_signed_distances,
        triangle_spherical_harmonics,
        triangle_vertex_types,
    ) = create_triangle_vertices(
        vertices,
        signed_distances,
        spherical_harmonics,
        lower_corners,
        level_sets,
        vertex_offsets,
    )

    # Continue with non-differentiable operations:
    # Count the number of faces created by each voxel.
    with torch.no_grad():
        face_offsets, voxel_cell_codes = count_triangle_faces(
            signed_distances,
            lower_corners,
            upper_corners,
            level_sets,
        )

        # Convert the face counts to face offsets in place.
        compute_exclusive_cumsum(face_offsets)
        num_faces = face_offsets[-1].item() // 3

        faces, voxel_indices = create_triangle_faces(
            signed_distances,
            neighbors,
            indices,
            vertex_offsets,
            triangle_vertex_types,
            voxel_cell_codes,
            level_sets,
            face_offsets,
        )

    # Get the boundaries between the meshes.
    vertex_boundaries = vertex_offsets[1:].view(num_level_sets, -1)[:, -1].clone()
    face_boundaries = face_offsets[1:].view(num_level_sets, -1)[:, -1].clone() // 3

    logging.debug(f"Created {num_vertices} vertices and {num_faces} faces.")
    return Triangles(
        triangle_vertices,
        triangle_signed_distances,
        triangle_spherical_harmonics,
        faces,
        voxel_indices,
        vertex_boundaries,
        face_boundaries,
    )
