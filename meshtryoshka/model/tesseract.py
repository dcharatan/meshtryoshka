import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Callable, Generator, Literal

import numpy as np
import torch
from einops import einsum
from jaxtyping import Float32, Int, Int32
from torch import Tensor, nn

from triangle_extraction import (
    UpscaledParameters,
    Voxels,
    dilate_occupancy,
    extract_voxels,
    get_occupancy_mesh,
    index_vertices,
    upscale_parameters,
    write_occupancy,
)

from ..initializer import Initializer
from ..utils import get_value_for_step
from .common import Scene


def get_corners(device: torch.device) -> Int[Tensor, "i=2 j=2 k=2 xyz=3"]:
    xyz = torch.arange(2, dtype=torch.int32, device=device)
    return torch.stack(torch.meshgrid((xyz, xyz, xyz), indexing="ij")[::-1], dim=-1)


def compute_corner_barycentric_coordinates(
    xyz: Float32[Tensor, "*batch xyz=3"],
) -> Float32[Tensor, "*batch i=2 j=2 k=2"]:
    """Convert XYZ coordinates in [0, 1] to barycentric corner coordinates."""
    xyz = xyz[..., None, None, None, :]
    corners = get_corners(xyz.device)
    return (corners * xyz + (1 - xyz) * (1 - corners)).prod(dim=-1)


@dataclass(frozen=True)
class TesseractCfg:
    # A schedule for the resolution of the tesseract's center.
    center_resolution: dict[int, int]

    # A schedule for the resolution of the tesseract's arms.
    arm_resolution: dict[int, int]

    # The fraction of the contracted field's volume allocated to the center.
    center_extent: float

    # The seed used to initialize the model's parameters.
    parameter_init_seed: int

    # The number of spherical harmonics used to represent the tesseract's surface. Maps
    # the current step to the active spherical harmonics degree.
    active_sh_degree: dict[int, int]

    # The learning rate multiplier for the spherical harmonics coefficients.
    sh_lr_scale: float

    # Because of how Hydra's dictionary merging works, it's easiest to completely
    # disable spherical harmonics this way.
    disable_spherical_harmonics: bool

    # Whether to entirely disable sparsity, including the filter grid.
    disable_sparsity: bool

    # Whether to disable the tesseract and extend the cubic grid to the entire volume.
    disable_tesseract: bool


class TesseractComponent(ABC):
    @abstractmethod
    def shape(self, step: int) -> tuple[int, int, int]:
        pass

    @property
    @abstractmethod
    def corners(self) -> Float32[Tensor, "i=2 j=2 k=2 xyz=3"]:
        pass

    def interpolate(
        self,
        xyz: Float32[Tensor, "*batch xyz=3"],
    ) -> Float32[Tensor, "*batch xyz=3"]:
        barycentric = compute_corner_barycentric_coordinates(xyz)
        return einsum(
            barycentric,
            self.corners.to(xyz.device),
            "... i j k, i j k xyz -> ... xyz",
        )


@dataclass(frozen=True)
class TesseractCenter(TesseractComponent):
    cfg: TesseractCfg

    def shape(self, step: int) -> tuple[int, int, int]:
        center = get_value_for_step(step, self.cfg.center_resolution)
        if self.cfg.disable_tesseract:
            center += 2 * get_value_for_step(step, self.cfg.arm_resolution)
        return (center, center, center)

    @property
    def corners(self) -> Float32[Tensor, "i=2 j=2 k=2 xyz=3"]:
        center_extent = torch.tensor(
            1.0 if self.cfg.disable_tesseract else self.cfg.center_extent
        )
        low = 0.5 - 0.5 * center_extent
        high = 0.5 + 0.5 * center_extent
        return low + (high - low) * get_corners(center_extent.device)


@dataclass(frozen=True)
class TesseractArm(TesseractComponent):
    cfg: TesseractCfg
    dimension: Literal[0, 1, 2]
    direction: Literal[0, 1]

    def shape(self, step: int) -> tuple[int, int, int]:
        center = get_value_for_step(step, self.cfg.center_resolution)
        shape = [center, center, center]
        shape[self.dimension] = get_value_for_step(step, self.cfg.arm_resolution)
        return tuple(shape)

    @property
    def corners(self) -> Float32[Tensor, "i=2 j=2 k=2 xyz=3"]:
        # Start with the center's corners.
        corners = TesseractCenter(self.cfg).corners

        # Flip the center's corners along the arm's dimension, then replace four of the
        # corners with the corresponding outside corners to create a truncated frustum.
        outside_corners = get_corners(corners.device)
        index = [slice(None), slice(None), slice(None)]
        index[self.dimension] = self.direction
        corners = corners.flip(self.dimension)
        corners[tuple(index)] = outside_corners[tuple(index)]

        return corners


FilterFn = Callable[["Tesseract", int], Int32[Tensor, " _"]]
GetSceneFn = Callable[
    [Voxels, Float32[Tensor, " vertex"], Float32[Tensor, "sh vertex rgb=3"]], Scene
]


class Tesseract(nn.Module):
    cfg: TesseractCfg
    components: tuple[TesseractComponent, ...]
    occupancy: Int32[Tensor, " _"]
    _voxels: Voxels | None
    step: Int[Tensor, ""]
    get_scene: GetSceneFn
    filter_fn: FilterFn
    initializer: Initializer
    vertex_boundaries: tuple[int, ...] | None

    # These are the model's learnable parameters.
    signed_distances_raw: Float32[Tensor, " vertex"]
    sh_rgb: Float32[Tensor, "sh vertex rgb=3"]
    sh_rest: Float32[Tensor, "sh1 vertex rgb=3"]

    def __init__(
        self,
        cfg: TesseractCfg,
        get_scene: GetSceneFn,
        filter_fn: FilterFn,
        initializer: Initializer,
    ) -> None:
        super().__init__()
        self.register_buffer("step", torch.tensor(1, dtype=torch.int32))
        self.cfg = cfg
        self.get_scene = get_scene
        self.filter_fn = filter_fn
        self.initializer = initializer
        self.vertex_boundaries = None
        self._voxels = None

        # Define the tesseract's components: a central cube and an arm for each face.
        components = [TesseractCenter(cfg)]
        if cfg.arm_resolution[0] > 0 and not cfg.disable_tesseract:
            components += [
                TesseractArm(cfg, 0, 0),
                TesseractArm(cfg, 0, 1),
                TesseractArm(cfg, 1, 0),
                TesseractArm(cfg, 1, 1),
                TesseractArm(cfg, 2, 0),
                TesseractArm(cfg, 2, 1),
            ]
        self.components = tuple(components)

        # Allocate memory for the occupancy grid.
        self.register_buffer("occupancy", filter_fn(self, 0))
        if cfg.disable_sparsity:
            # Mark every voxel as occupied.
            self.occupancy = torch.full_like(self.occupancy, -1)

        # Initialize the model's parameters.
        generator = torch.Generator()
        generator.manual_seed(cfg.parameter_init_seed)
        num_vertices, _ = self.to("cuda").voxels.vertices.shape
        self.signed_distances_raw = nn.Parameter(
            torch.randn((num_vertices,), dtype=torch.float32) * 0.01
        )

        self.sh_rgb = nn.Parameter(
            torch.randn((1, num_vertices, 3), dtype=torch.float32) * 0.1
        )
        self.sh_rest = nn.Parameter(torch.empty([], dtype=torch.float32))
        self.sh_rest.custom_lr_scale = self.cfg.sh_lr_scale
        if 0 in cfg.active_sh_degree and self.get_sh_degree_for_step(0) > 0:
            self.active_sh_degree = self.get_sh_degree_for_step(0)
            n_features = (1 + self.active_sh_degree) ** 2
            if n_features > 1:
                self.register_spherical_harmonics(1, n_features, self.sh_rest)

        else:
            self.active_sh_degree = 0

    def get_sh_degree_for_step(self, step: int) -> int:
        if self.cfg.disable_spherical_harmonics:
            return 0
        else:
            return get_value_for_step(step, self.cfg.active_sh_degree)

    @property
    def signed_distances(self) -> Float32[Tensor, " vertex"]:
        return self.initializer.get_sdf_modify(
            self.signed_distances_raw,
            self.voxels.vertices,
            self.step.item(),
        )

    @property
    def spherical_harmonics(self) -> Float32[Tensor, "sh vertex rgb=3"]:
        if self.active_sh_degree == 0:
            return self.sh_rgb
        else:
            return torch.cat(
                [
                    self.sh_rgb,
                    self.sh_rest,
                ],
                dim=0,
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the cuboid that would contain the tesseract."""
        center = get_value_for_step(self.step.item(), self.cfg.center_resolution)
        arm = get_value_for_step(self.step.item(), self.cfg.arm_resolution)
        extent = center + 2 * arm
        return (extent, extent, extent)

    @property
    def occupancy_boundaries(self) -> tuple[int, ...]:
        boundaries = [0] + [prod(c.shape(self.step.item())) for c in self.components]
        return tuple(np.cumsum(boundaries).tolist())

    @staticmethod
    def concatenate_voxels(voxels: list[Voxels]) -> Voxels:
        """Reindex and concatenate the input voxels."""
        num_vertices = 0
        num_voxels = 0
        num_subvoxels = 0

        # Count the number of vertices, voxels, and subvoxels.
        for v in voxels:
            num_vertices += v.num_vertices
            num_voxels += v.num_voxels
            num_subvoxels += v.num_subvoxels

        # Reindex the voxels.
        vertex_offset = 0
        voxel_offset = 0
        subvoxel_offset = 0
        grid_offset = 0
        neighbors = []
        voxel_lower_corners = []
        subvoxel_lower_corners = []
        voxel_upper_corners = []
        indices = []

        for v in voxels:
            # The neighbors are voxel/subvoxel indices, so they need to be reindexed.
            # All voxels appear before all subvoxels, so we need to reindex the voxels
            # and the subvoxels separately.
            subvoxel_mask = v.neighbors >= v.num_voxels
            v_neighbors = v.neighbors.clone()
            v_neighbors[subvoxel_mask] += subvoxel_offset + num_voxels - v.num_voxels
            v_neighbors[~subvoxel_mask] += voxel_offset
            neighbors.append(v_neighbors)

            # The corners refer to vertex indices.
            voxel_lower_corners.append(
                v.lower_corners[:, : v.num_voxels] + vertex_offset
            )
            subvoxel_lower_corners.append(
                v.lower_corners[:, v.num_voxels :] + vertex_offset
            )
            voxel_upper_corners.append(v.upper_corners + vertex_offset)

            # The indices are voxel indices.
            indices.append(v.indices + grid_offset)

            # Update the offsets.
            vertex_offset += v.num_vertices
            voxel_offset += v.num_voxels
            subvoxel_offset += v.num_subvoxels
            grid_offset += prod(v.grid_shape)

        # The vertices are XYZ coordinates, so they can simply be concatenated.
        vertices = torch.cat([v.vertices for v in voxels], dim=0)
        return Voxels(
            vertices,
            torch.cat(neighbors, dim=1),
            torch.cat((*voxel_lower_corners, *subvoxel_lower_corners), dim=1),
            torch.cat(voxel_upper_corners, dim=1),
            torch.cat(indices, dim=0),
            (0, 0, 0),
        )

    def iterate_components(
        self,
        occupancy: Int32[Tensor, " _"] | None = None,
    ) -> Generator[
        tuple[TesseractComponent, Int32[Tensor, "i j k_packed"]],
        None,
        None,
    ]:
        """Iterate through the tesseract's components and their occupancy grids."""
        occupancy = self.occupancy if occupancy is None else occupancy
        for index, component in enumerate(self.components):
            start = self.occupancy_boundaries[index]
            end = self.occupancy_boundaries[index + 1]

            # Extract the component's slice of the tesseract's occupancy grid.
            i, j, k = component.shape(self.step.item())
            component_occupancy = occupancy[start // 32 : end // 32]
            component_occupancy = component_occupancy.view((i, j, k // 32))
            yield component, component_occupancy

    def extract_voxels(
        self,
        occupancy: Int32[Tensor, " _"] | None = None,
    ) -> tuple[Voxels, tuple[int, ...]]:
        voxels = []
        vertex_counts = [0]

        for component, occupancy in self.iterate_components(occupancy):
            # Extract voxels for the component.
            c_voxels = extract_voxels(occupancy)

            # Deform the component's vertices according to its corners.
            vertices = component.interpolate(c_voxels.vertices)
            c_voxels = c_voxels._replace(vertices=vertices)

            vertex_counts.append(c_voxels.num_vertices)
            voxels.append(c_voxels)

        # Compute vertex boundaries from the individual components' vertex counts.
        vertex_boundaries = tuple(np.cumsum(vertex_counts).tolist())

        # Concatenate the voxels and set the grid shape to the overall shape.
        voxels = Tesseract.concatenate_voxels(voxels)
        voxels = voxels._replace(grid_shape=self.shape)

        # De-duplicate the vertices.
        voxels = Tesseract.reindex_duplicate_vertices(voxels)
        return voxels, vertex_boundaries

    @property
    def device(self) -> torch.device:
        return self.occupancy.device

    @property
    def voxels(self) -> Voxels:
        # Extract voxels if they haven't been cached or if their size has changed.
        if self._voxels is None or self._voxels.grid_shape != self.shape:
            self._voxels, self.vertex_boundaries = self.extract_voxels()
        return self._voxels

    def get_mesh(self) -> Float32[Tensor, "triangle corner=3 xyz=3"]:
        """Get an uncolored mesh of the tesseract."""
        dummy_colors = torch.zeros((3, 2, 3), dtype=torch.float32, device=self.device)
        vertices = [
            component.interpolate(get_occupancy_mesh(occupancy, dummy_colors)[0])
            for component, occupancy in self.iterate_components()
        ]
        return torch.cat(vertices, dim=0)

    @torch.no_grad()
    def set_step(self, step: int) -> bool:
        subdivide_center = step in self.cfg.center_resolution
        subdivide_arms = subdivide_center or step in self.cfg.arm_resolution

        if_change_optimizer = False

        prev_num_features = self.spherical_harmonics.shape[0]
        self.active_sh_degree = self.get_sh_degree_for_step(step)
        new_num_features = new_num_features = (1 + self.active_sh_degree) ** 2

        assert new_num_features >= prev_num_features

        if new_num_features > prev_num_features:
            # Expand the spherical harmonics to the new size.
            self.register_spherical_harmonics(
                prev_num_features, new_num_features, self.sh_rest
            )
            if_change_optimizer = True

        # Outside of subdivision steps, do nothing.
        if not (subdivide_center or subdivide_arms) or step == 0:
            self.step.fill_(step)
            return if_change_optimizer

        logging.info(f"Subdividing tesseract at step {step}.")
        assert self.get_scene is not None

        # Copy parameter values from the "canonical" duplicates to all other duplicates.
        index = index_vertices(self.voxels.vertices, 1e-6)
        self.signed_distances_raw.data = self.signed_distances_raw[index]
        self.spherical_harmonics.data = self.spherical_harmonics[:, index, :]

        scene = self.get_scene(
            self.voxels,
            self.signed_distances,
            self.spherical_harmonics,
        )

        # Extract an occupancy grid at the zero level set.
        occupancy_to_upscale = write_occupancy(
            scene.voxel_indices,
            self.occupancy.shape,
            0,
            1,
        )
        if self.cfg.disable_sparsity:
            occupancy_to_upscale = torch.full_like(occupancy_to_upscale, -1)

        # Handle subdividing the individual components.
        upscaled: list[UpscaledParameters] = []
        bundle = enumerate(self.iterate_components())
        for index, (component, occupancy) in bundle:
            vertex_start = self.vertex_boundaries[index]
            vertex_end = self.vertex_boundaries[index + 1]
            occupancy_start = self.occupancy_boundaries[index] // 32
            occupancy_end = self.occupancy_boundaries[index + 1] // 32

            # Dilate the occupancy grid slightly (to account for other level sets).
            i, j, k = component.shape(step - 1)
            c_occupancy_to_upscale = occupancy_to_upscale[occupancy_start:occupancy_end]
            c_occupancy_to_upscale = c_occupancy_to_upscale.view((i, j, k // 32))
            c_occupancy_to_upscale = dilate_occupancy(c_occupancy_to_upscale, 4)

            # Upscale the parameters and the occupancy grid.
            c_upscaled = upscale_parameters(
                self.signed_distances[vertex_start:vertex_end],
                self.spherical_harmonics[:, vertex_start:vertex_end, :],
                occupancy,
                c_occupancy_to_upscale,
                component.shape(step),
            )
            upscaled.append(c_upscaled)

        # Update the occupancy grid and reset the extracted voxels.
        self.occupancy = torch.cat([c.occupancy.view(-1) for c in upscaled], dim=0)
        self.signed_distances_raw = nn.Parameter(
            torch.cat([c.signed_distances for c in upscaled])
        )

        spherical_harmonics = torch.cat(
            [c.spherical_harmonics for c in upscaled], dim=1
        )
        self.sh_rgb = nn.Parameter(spherical_harmonics[:1])
        self.sh_rest = nn.Parameter(spherical_harmonics[1:])
        self.sh_rest.custom_lr_scale = self.cfg.sh_lr_scale

        # Filter the voxels at the current resolution.
        self.step.fill_(step)
        self._voxels = None
        return True

    def register_spherical_harmonics(
        self,
        old_n_features: int,
        new_n_features: int,
        old_sh_rest: Float32[Tensor, "sh vertex rgb=3"] | Float32[Tensor, ""],
    ) -> None:
        num_vertices, _ = self.voxels.vertices.shape
        if old_n_features == 1:
            self.sh_rest = nn.Parameter(
                torch.zeros(
                    (new_n_features - 1, num_vertices, 3),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        else:
            assert new_n_features > old_n_features
            self.sh_rest = nn.Parameter(
                torch.cat(
                    [
                        old_sh_rest,
                        torch.zeros(
                            (new_n_features - old_n_features, num_vertices, 3),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    ],
                    dim=0,
                )
            )
        self.sh_rest.custom_lr_scale = self.cfg.sh_lr_scale

    @staticmethod
    def reindex_duplicate_vertices(voxels: Voxels) -> Voxels:
        index = index_vertices(voxels.vertices, 1e-6)
        return voxels._replace(
            lower_corners=index[voxels.lower_corners],
            upper_corners=index[voxels.upper_corners],
        )
