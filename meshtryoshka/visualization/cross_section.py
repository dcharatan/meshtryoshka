from dataclasses import replace

from ..model.common import Scene, SectionParams
from ..model.contraction import Contraction


def apply_cross_section(
    scene: Scene,
    section_params: SectionParams,
    contraction: Contraction,
) -> Scene:
    # We want to use contracted version of the scene for consistency.
    vertices_contracted = contraction.contract(scene.vertices)
    faces = scene.faces

    # For each face, extract the coordinate along the specified view_axis.
    # vertices[faces] gives a tensor of shape (num_faces, 3, 3);
    # indexing with section_params.view_axis reduces this to (num_faces, 3).
    face_coords = vertices_contracted[faces, section_params.view_axis]

    # Create a mask: if any vertex is outside the bounds
    # then the face is completely outside the range and should be pruned.
    mask = (
        (face_coords >= section_params.min_value)
        & (face_coords <= section_params.max_value)
    ).all(dim=1)

    # Use the mask to filter the faces (and shell_indices accordingly).
    return replace(
        scene,
        faces=scene.faces[mask],
        voxel_indices=scene.voxel_indices[mask],
    )
