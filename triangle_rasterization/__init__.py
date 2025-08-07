from typing import Literal

import torch
from einops import rearrange, reduce, repeat
from jaxtyping import Float32, Int32
from torch import Tensor

from .bound_triangles import bound_triangles
from .common import TileGrid, record_function
from .composite import composite
from .delineate_tiles import delineate_tiles
from .evaluate_spherical_harmonics import evaluate_spherical_harmonics
from .generate_keys import generate_keys
from .interpolate import interpolate
from .intersect import RenderedImages, intersect
from .intersect_and_composite import intersect_and_composite
from .keys import create_key_defines
from .project_vertices import project_vertices
from .sort_keys import sort_keys


@record_function("rasterize")
def rasterize(
    vertices: Float32[Tensor, "vertex xyz=3"],
    faces: Int32[Tensor, "triangle corner=3"],
    shell_face_boundaries: Int32[Tensor, " shell"],
    extrinsics: Float32[Tensor, "batch 4 4"],
    intrinsics: Float32[Tensor, "batch 3 3"],
    image_shape: tuple[int, int],
    tile_shape: tuple[int, int] = (16, 16),
    near_plane: float = 0.2,
    num_camera_bits: int = 6,
    num_shell_bits: int = 6,
    num_tile_bits: int = 20,
) -> RenderedImages:
    """Rasterize the specified triangles from the specified camera poses.

    Extrinsics are OpenCV-style world-to-camera matrices (+Z look vector, -Y up vector,
    +X right vector). Intrinsics are unnormalized.
    """

    # Guard against empty scenes.
    h, w = image_shape
    device = vertices.device
    c, _, _ = extrinsics.shape
    (s,) = shell_face_boundaries.shape
    if vertices.numel() == 0 or faces.numel() == 0:
        uvw = torch.zeros((c, s, h, w, 3), device=device, dtype=torch.fsoat32)
        index = torch.zeros((c, s, h, w), device=device, dtype=torch.int32)
        return RenderedImages(uvw, index)

    grid = TileGrid(tile_shape, image_shape, c, s)
    key_defines = create_key_defines(
        ("camera", num_camera_bits),
        ("shell", num_shell_bits),
        ("tile", num_tile_bits),
        ("depth", 32),
    )

    projected_positions, depths = project_vertices(vertices, extrinsics, intrinsics)
    triangle_bounds = bound_triangles(
        projected_positions,
        depths,
        faces,
        grid,
        near_plane,
    )
    paired_keys = generate_keys(
        depths,
        faces,
        shell_face_boundaries,
        triangle_bounds.tile_minima,
        triangle_bounds.tile_maxima,
        triangle_bounds.num_tiles_overlapped,
        grid,
        key_defines,
    )
    del triangle_bounds
    del depths
    paired_keys = sort_keys(paired_keys, grid)
    tile_boundaries = delineate_tiles(
        paired_keys,
        grid,
        key_defines,
        True,
    )
    return intersect(
        projected_positions,
        faces,
        paired_keys.triangle_indices,
        tile_boundaries,
        grid,
        key_defines,
    )


@record_function("render_explicit")
def render_explicit(
    extrinsics: Float32[Tensor, "batch 4 4"],
    intrinsics: Float32[Tensor, "batch 3 3"],
    image_shape: tuple[int, int],
    vertices: Float32[Tensor, "vertex xyz=3"],
    faces: Int32[Tensor, "triangle corner=3"],
    shell_face_boundaries: Int32[Tensor, " shell"],
    signed_distances: Float32[Tensor, " vertex"],
    spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
    sh_degree: int,
    sharpness: Float32[Tensor, ""],
    background: Float32[Tensor, "rgb=3"],
    tile_shape: tuple[int, int] = (16, 16),
    near_plane: float = 0.2,
    num_camera_bits: int = 6,
    num_shell_bits: int = 6,
    num_tile_bits: int = 20,
    ssaa: int = 1,
) -> Float32[Tensor, "batch rgb=3 height width"]:
    # Modify the image shape and intrinsics if super-sampling anti-aliasing is desired.
    if ssaa > 1:
        h, w = image_shape
        image_shape = (h * ssaa, w * ssaa)
        multiplier = torch.tensor((ssaa, ssaa, 1), device=intrinsics.device)
        intrinsics = intrinsics * multiplier[:, None]

    # Non-differentiably rasterize each of the shells.
    uv, index = rasterize(
        vertices,
        faces,
        shell_face_boundaries,
        extrinsics,
        intrinsics,
        image_shape,
        tile_shape=tile_shape,
        near_plane=near_plane,
        num_camera_bits=num_camera_bits,
        num_shell_bits=num_shell_bits,
        num_tile_bits=num_tile_bits,
    )

    # Differentiably evaluate the spherical harmonics.
    colors = evaluate_spherical_harmonics(
        vertices,
        spherical_harmonics,
        extrinsics,
        sh_degree,
    )

    # Differentiably interpolate the SDF and colors at the rasterized pixels.
    signed_distances, colors = interpolate(uv, index, faces, signed_distances, colors)
    mask = index != -1

    # Flatten the rendered pixels and then composite them.
    signed_distances = rearrange(signed_distances, "b l h w -> (b h w) l").contiguous()
    colors = rearrange(colors, "b l h w c -> (b h w) l c").contiguous()
    mask = rearrange(mask, "b l h w -> (b h w) l").contiguous()
    image = composite(signed_distances, colors, mask, background, sharpness)
    h, w = image_shape
    b, _, _ = extrinsics.shape
    image = rearrange(image, "(b h w) c -> b c h w", b=b, h=h, w=w)

    # Scale the image if SSAA is desired.
    if ssaa > 1:
        image = reduce(image, "b c (h mh) (w mw) -> b c h w", "mean", mh=ssaa, mw=ssaa)

    return image


@record_function("render_fused")
def render_fused(
    extrinsics: Float32[Tensor, "batch 4 4"],
    intrinsics: Float32[Tensor, "batch 3 3"],
    image_shape: tuple[int, int],
    vertices: Float32[Tensor, "vertex xyz=3"],
    faces: Int32[Tensor, "triangle corner=3"],
    shell_face_boundaries: Int32[Tensor, " shell"],
    signed_distances: Float32[Tensor, " vertex"],
    spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
    sh_degree: int,
    sharpness: Float32[Tensor, ""],
    background: Float32[Tensor, "rgb=3"],
    tile_shape: tuple[int, int] = (16, 16),
    near_plane: float = 0.2,
    ssaa: int = 1,
    num_camera_bits: int = 6,
    num_shell_bits: int = 6,
    num_tile_bits: int = 20,
) -> Float32[Tensor, "batch rgb=3 height width"]:
    """Render the specified triangles from the specified camera poses.

    Extrinsics are OpenCV-style world-to-camera matrices (+Z look vector, -Y up vector,
    +X right vector). Intrinsics are unnormalized.
    """

    # Guard against empty scenes.
    b, _, _ = extrinsics.shape
    h, w = image_shape
    if vertices.numel() == 0 or faces.numel() == 0:
        return repeat(background, "c -> b c h w", b=b, h=h, w=w)

    # Modify the image shape and intrinsics if super-sampling anti-aliasing is desired.
    if ssaa > 1:
        image_shape = (h * ssaa, w * ssaa)
        multiplier = torch.tensor((ssaa, ssaa, 1), device=intrinsics.device)
        intrinsics = intrinsics * multiplier[:, None]

    (s,) = shell_face_boundaries.shape
    grid = TileGrid(tile_shape, image_shape, b, s)
    key_defines = create_key_defines(
        ("camera", num_camera_bits),
        ("tile", num_tile_bits),
        ("depth", 32),
        ("shell", num_shell_bits),
    )

    # Project the vertices to image space.
    projected_positions, depths = project_vertices(vertices, extrinsics, intrinsics)

    # Compute triangle tile membership.
    triangle_bounds = bound_triangles(
        projected_positions,
        depths,
        faces,
        grid,
        near_plane,
    )

    # Generate keys (one per triangle-tile overlap).
    paired_keys = generate_keys(
        depths,
        faces,
        shell_face_boundaries,
        triangle_bounds.tile_minima,
        triangle_bounds.tile_maxima,
        triangle_bounds.num_tiles_overlapped,
        grid,
        key_defines,
    )

    # Save a bit of VRAM since these things are no longer needed.
    del triangle_bounds
    del depths

    # Sort the keys. After sorting, keys from the same tile will be
    # contiguous and ordered by ascending depth.
    paired_keys = sort_keys(paired_keys, grid)

    # Delineate tile boundaries.
    tile_boundaries = delineate_tiles(paired_keys, grid, key_defines, False)

    # Evaluate the spherical harmonics.
    colors = evaluate_spherical_harmonics(
        vertices,
        spherical_harmonics,
        extrinsics,
        sh_degree,
    )

    # Alpha-composite triangles within each tile.
    _, _, t, _ = tile_boundaries.shape
    images = intersect_and_composite(
        projected_positions,
        colors,
        signed_distances,
        faces,
        sharpness,
        paired_keys.keys,
        paired_keys.triangle_indices,
        tile_boundaries.view(b, t, 2),
        s,
        grid,
        key_defines,
    )

    # Composite the images onto the background.
    images = images[:, :3] + images[:, 3:4] * background[:, None, None]

    # Scale the images if SSAA is desired.
    if ssaa > 1:
        images = reduce(
            images,
            "b c (h mh) (w mw) -> b c h w",
            "mean",
            mh=ssaa,
            mw=ssaa,
        )

    # Ensure that the images are within a valid range.
    return images.clip(min=0, max=1)


def render(
    extrinsics: Float32[Tensor, "batch 4 4"],
    intrinsics: Float32[Tensor, "batch 3 3"],
    image_shape: tuple[int, int],
    vertices: Float32[Tensor, "vertex xyz=3"],
    faces: Int32[Tensor, "triangle corner=3"],
    shell_face_boundaries: Int32[Tensor, " shell"],
    signed_distances: Float32[Tensor, " vertex"],
    spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
    sh_degree: int,
    sharpness: Float32[Tensor, ""],
    background: Float32[Tensor, "rgb=3"],
    tile_shape: tuple[int, int] = (16, 16),
    near_plane: float = 0.2,
    num_camera_bits: int = 6,
    num_shell_bits: int = 6,
    num_tile_bits: int = 20,
    ssaa: int = 1,
    mode: Literal["explicit", "fused"] = "fused",
) -> Float32[Tensor, "batch rgb=3 height width"]:
    _render = render_fused if mode == "fused" else render_explicit
    return _render(
        extrinsics,
        intrinsics,
        image_shape,
        vertices,
        faces,
        shell_face_boundaries,
        signed_distances,
        spherical_harmonics,
        sh_degree,
        sharpness,
        background,
        tile_shape=tile_shape,
        near_plane=near_plane,
        num_camera_bits=num_camera_bits,
        num_shell_bits=num_shell_bits,
        num_tile_bits=num_tile_bits,
        ssaa=ssaa,
    )
