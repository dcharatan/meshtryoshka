from dataclasses import dataclass, replace
from typing import Literal

import torch
from jaxtyping import Float, Float32, Int
from torch import Tensor

from triangle_extraction import compute_sdf_regularizers
from triangle_rasterization import render

from ..components.visualization import (
    unbuffer_scene_with_random_colors,
    vertices_to_scene,
)
from ..initializer import Initializer, InitializerCfg, get_initializer
from ..visualization.cross_section import apply_cross_section
from .common import Scene, SectionParams
from .contraction import Contraction, ContractionCfg
from .model import Model
from .tessellator import Tessellator, TessellatorCfg
from .tesseract import Tesseract, TesseractCfg
from .tesseract_filter import get_create_filter


@dataclass(frozen=True)
class ModelMeshtryoshkaCfg:
    name: Literal["meshtryoshka"]
    background: tuple[float, float, float]
    initializer: InitializerCfg
    contraction: ContractionCfg
    tesseract: TesseractCfg
    tessellator: TessellatorCfg
    loss_weights: dict[
        Literal["rgb", "eikonal_pos", "eikonal_neg", "curvature", "sdf_binary", "sh"],
        float,
    ]
    render_mode: Literal["fused", "explicit"]
    train_ssaa: int
    render_ssaa: int
    train_near_plane: float
    eval_near_plane: float


class ModelMeshtryoshka(Model):
    step: Int[Tensor, ""]
    background: Float[Tensor, "rgb=3"]
    tesseract: Tesseract
    contraction: Contraction
    initializer: Initializer

    # These are used to filter out occluded voxels.
    extrinsics: Float[Tensor, "camera 4 4"]
    intrinsics: Float[Tensor, "camera 3 3"]
    image_shape: tuple[int, int]

    def __init__(
        self,
        cfg: ModelMeshtryoshkaCfg,
        extrinsics: Float[Tensor, "*batch 4 4"],
        intrinsics: Float[Tensor, "*batch 3 3"],
        image_shape: tuple[int, int],
        point_cloud: Float[Tensor, "point 3"] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Set up the filter tesseract, which permanently disables voxels that aren't
        # seen by any training views.
        self.contraction = Contraction(cfg.contraction, cfg.tesseract.center_extent)

        # Initialize the model's components.
        self.initializer = get_initializer(
            cfg.initializer,
            self.contraction,
            point_cloud,
        )
        self.tessellator = Tessellator(cfg.tessellator, self.contraction)
        self.tesseract = Tesseract(
            cfg.tesseract,
            self.tessellator.tessellate_surface,
            get_create_filter(
                1,
                0,
                extrinsics,
                intrinsics,
                image_shape,
                self.contraction,
            ),
            self.initializer,
        )

        self.register_buffer(
            "background",
            torch.tensor(cfg.background, dtype=torch.float32),
        )
        self.register_buffer("step", torch.zeros((1,), dtype=torch.int32))

        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.image_shape = image_shape

    def render_scene(
        self,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        image_shape: tuple[int, int],
        scene: Scene,
        background: Float[Tensor, "rgb=3"] | None = None,
        near_plane: float = 0.2,
        ssaa: int = 1,
    ) -> Float[Tensor, "batch rgb=3 height width"]:
        active_sh_degree = (
            0
            if scene.spherical_harmonics.shape[0] == 1
            else self.tesseract.active_sh_degree
        )

        return render(
            extrinsics,
            intrinsics,
            image_shape,
            scene.vertices,
            scene.faces,
            scene.face_boundaries,
            scene.signed_distances,
            scene.spherical_harmonics,
            active_sh_degree,
            self.sharpness,
            self.background if background is None else background,
            near_plane=near_plane,
            ssaa=ssaa,
            mode=self.cfg.render_mode,
        )

    def get_eval_render_modes(self) -> tuple[str, ...]:
        return ("layered", "surface")

    def get_viewer_render_modes(self) -> tuple[str, ...]:
        shell_modes = [
            f"shell_{i:0>2}" for i, _ in enumerate(self.tessellator.level_sets)
        ]
        shell_random_color_modes = [
            f"shell_random_color_{i:0>2}"
            for i, _ in enumerate(self.tessellator.level_sets)
        ]
        return (
            "layered",
            "surface",
            "occupancy",
            *shell_modes,
            *shell_random_color_modes,
        )

    def render(
        self,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        image_shape: tuple[int, int],
        mode: str = "surface",
        eval: bool = False,
        section_params: SectionParams | None = None,
        contract: bool = False,
    ) -> Float[Tensor, "batch rgb=3 height width"]:
        near_plane = self.cfg.eval_near_plane if eval else self.cfg.train_near_plane

        if mode == "layered":
            scene = self.tessellator.tessellate(
                self.tesseract.voxels,
                self.tesseract.signed_distances,
                self.tesseract.spherical_harmonics,
            )
        elif mode == "surface":
            scene = self.tessellator.tessellate_surface(
                self.tesseract.voxels,
                self.tesseract.signed_distances,
                self.tesseract.spherical_harmonics,
            )
        elif mode == "occupancy":
            vertices = self.tesseract.get_mesh()
            vertices = self.contraction.uncontract(vertices)
            scene = vertices_to_scene(vertices)
        elif "shell_random_color" in mode:
            shell_index = int(mode.split("_")[3])
            mask = torch.zeros_like(self.tessellator.level_sets, dtype=torch.bool)
            mask[shell_index] = True
            scene = self.tessellator.tessellate(
                self.tesseract.voxels,
                self.tesseract.signed_distances,
                self.tesseract.spherical_harmonics,
                mask,
            )
            scene = unbuffer_scene_with_random_colors(scene)
        elif "shell" in mode:
            shell_index = int(mode.split("_")[1])
            mask = torch.zeros_like(self.tessellator.level_sets, dtype=torch.bool)
            mask[shell_index] = True
            scene = self.tessellator.tessellate(
                self.tesseract.voxels,
                self.tesseract.signed_distances,
                self.tesseract.spherical_harmonics,
                mask,
            )
            scene = replace(
                scene,
                signed_distances=torch.full_like(scene.signed_distances, -1e8),
            )
        else:
            raise ValueError(f"Unrecognized rendering mode {mode}.")

        # Apply options for the interactive viewer.
        if section_params is not None and (
            section_params.min_value != 0 or section_params.max_value != 1
        ):
            scene = apply_cross_section(scene, section_params, self.contraction)
        if contract:
            scene = replace(
                scene,
                vertices=2 * self.tessellator.contraction.contract(scene.vertices) - 1,
            )

        return self.render_scene(
            extrinsics,
            intrinsics,
            image_shape,
            scene,
            near_plane=near_plane,
            ssaa=self.cfg.render_ssaa,
        )

    def fit(
        self,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        images: Float[Tensor, "batch rgb=3 height width"],
        background: Float[Tensor, "rgb=3"] | None = None,
    ) -> tuple[
        dict[str, Float[Tensor, ""]],  # losses
        dict[str, float | int],  # metrics
    ]:
        scene = self.tessellator.tessellate(
            self.tesseract.voxels,
            self.tesseract.signed_distances,
            self.tesseract.spherical_harmonics,
        )

        _, _, h, w = images.shape
        images_hat = self.render_scene(
            extrinsics,
            intrinsics,
            (h, w),
            scene,
            self.background if background is None else background,
            near_plane=self.cfg.train_near_plane,
            ssaa=self.cfg.train_ssaa,
        )
        l2_rgb_loss = ((images - images_hat) ** 2).mean(dim=(1, 2)).mean()
        l1_rgb_loss = (images - images_hat).abs().mean(dim=(1, 2)).mean()
        rgb_loss = 0.5 * (l2_rgb_loss + l1_rgb_loss)

        # Compute the regularizers.
        eikonal_loss_pos, eikonal_loss_neg, curvature_loss = compute_sdf_regularizers(
            self.tesseract.signed_distances,
            self.tesseract.voxels.neighbors,
            self.tesseract.voxels.lower_corners,
            self.tesseract.voxels.upper_corners,
            self.tesseract.shape,
        )

        losses = {
            "rgb": rgb_loss * self.cfg.loss_weights["rgb"],
            "eikonal_pos": eikonal_loss_pos.mean()
            * self.cfg.loss_weights["eikonal_pos"],
            "eikonal_neg": eikonal_loss_neg.mean()
            * self.cfg.loss_weights["eikonal_neg"],
            "curvature": curvature_loss.mean() * self.cfg.loss_weights["curvature"],
        }

        # Regularize the non-zero spherical harmonics.
        sh_coeffs = scene.spherical_harmonics[1:]
        losses["sh"] = sh_coeffs.pow(2).mean() * self.cfg.loss_weights["sh"]

        # This loss encourages the SDF to represent either solids or empty space.
        if self.cfg.loss_weights["sdf_binary"] > 0:
            weight = self.cfg.loss_weights["sdf_binary"]
            losses["sdf_binary"] = -scene.signed_distances.abs().mean() * weight

        # Collect metrics.
        num_samples, _ = self.tesseract.voxels.vertices.shape
        _, num_occupied_voxels_and_subvoxels = self.tesseract.voxels.lower_corners.shape
        _, num_occupied_voxels = self.tesseract.voxels.upper_corners.shape
        metrics = {
            "sharpness": self.sharpness.item(),
            "num_vertices": scene.num_vertices,
            "num_faces": scene.num_faces,
            "num_samples": num_samples,
            "num_occupied_voxels_and_subvoxels": num_occupied_voxels_and_subvoxels,
            "num_occupied_voxels": num_occupied_voxels,
            "num_subvoxels": num_occupied_voxels_and_subvoxels - num_occupied_voxels,
        }

        return losses, metrics

    @torch.no_grad()
    def set_step(self, step: int) -> bool:
        self.tessellator.set_step(step)
        self.step.fill_(step)
        return self.tesseract.set_step(step)

    @property
    def sharpness(self) -> Float32[Tensor, ""]:
        return torch.tensor(self.tessellator.get_sharpness(), device=self.device)
