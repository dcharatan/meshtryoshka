from dataclasses import replace

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ..model.common import Scene


def unbuffer_scene_with_random_colors(scene: Scene) -> Scene:
    """
    Returns a new Scene where vertices are unbuffered (each face has its own vertices)
    and the vertex colors are set to a random color for each triangle. All vertices
    of a triangle share the same color, and the output is deterministic.
    """
    device = scene.vertices.device
    dtype = scene.vertices.dtype

    # Unbuffer vertices: For each face, gather its own vertices.
    new_vertices = scene.vertices[scene.faces].reshape(-1, 3)

    # Unbuffer signed distances in the same way.
    new_signed_distances = scene.signed_distances[scene.faces].reshape(-1)

    # New faces: since each face's vertices are now unique, they are consecutive.
    num_faces = scene.faces.shape[0]
    new_faces = torch.arange(
        num_faces * 3, device=device, dtype=scene.faces.dtype
    ).reshape(num_faces, 3)

    # Create random colors for each face deterministically.
    # Using a fixed seed ensures reproducibility given the same number of faces.
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    face_colors = torch.rand(num_faces, 3, generator=gen, dtype=dtype, device=device)
    # Expand each face's color to its 3 vertices.
    new_colors = (
        face_colors.unsqueeze(1).expand(num_faces, 3, 3).reshape(1, num_faces * 3, 3)
    )  # sh = 1 here

    return replace(
        scene,
        vertices=new_vertices,
        spherical_harmonics=new_colors,
        signed_distances=torch.full_like(new_signed_distances, -1e8),  # fully opaque
        faces=new_faces,
    )


def vertices_to_scene(vertices: Float[Tensor, "triangle corner=3 xyz=3"]) -> Scene:
    device = vertices.device
    num_faces, _, _ = vertices.shape
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    colors = torch.rand(
        num_faces // 2,
        3,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    colors = repeat(colors, "f rgb -> (f t c) rgb", t=2, c=3)
    faces = torch.arange(num_faces * 3, device=device, dtype=torch.int32)
    faces = rearrange(faces, "(t c) -> t c", c=3)
    return Scene(
        rearrange(vertices, "t c xyz -> (t c) xyz"),
        colors[None],
        torch.full_like(colors[:, 0], -1e8),
        faces,
        torch.zeros((num_faces,), dtype=torch.int32, device=device),
        torch.tensor((-1e8,), dtype=torch.float32, device=device),
        torch.tensor((num_faces,), dtype=torch.int32, device=device),
    )
