from abc import ABC, abstractmethod
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..components.trajectory import generate_spin_wave_loop
from ..dataset import LoadedDataset
from ..image_io import save_image
from ..utils import get_metrics


class Model(ABC, nn.Module):
    @abstractmethod
    def render(
        self,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        image_shape: tuple[int, int],
        mode: str = "default",
    ) -> Float[Tensor, "batch rgb=3 height width"]:
        """Render full images."""
        pass

    @abstractmethod
    def fit(
        self,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        images: Float[Tensor, "batch rgb=3 height width"],
    ) -> tuple[
        dict[str, Float[Tensor, ""]],  # losses
        dict[str, float | int],  # metrics
    ]:
        """Fit the model to the given images. Return losses and metrics."""
        pass

    @abstractmethod
    def get_render_modes(self) -> tuple[str, ...]:
        """Get the supported rendering modes for the viewer and visualizations."""
        pass

    def set_step(self, step: int) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def metrics(
        self,
        dataset: LoadedDataset,
        compute_metrics: bool = True,
        output_path: Path | None = None,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
        # Initialize metrics storage
        metrics_accumulated = {}
        metrics_individual = {}
        num_images = dataset.images.shape[0]
        _, _, h, w = dataset.images.shape

        eval_render_modes = self.get_eval_render_modes()

        # Iterate over images one by one
        for i in range(num_images):
            ground_truth = dataset.images[i : i + 1].to(self.device)
            extrinsics = dataset.extrinsics[i : i + 1].to(self.device)
            intrinsics = dataset.intrinsics[i : i + 1].to(self.device)
            image_name = dataset.image_paths[i].name

            for near_plane in self.cfg.eval_near_planes:
                for render_type in eval_render_modes:
                    metric_type = f"{render_type}_{near_plane}"
                    with torch.no_grad():
                        image_hat = self.render(
                            extrinsics,
                            intrinsics,
                            (h, w),
                            mode=render_type,
                            near_plane=near_plane,
                        )
                        if output_path is not None:
                            out_dir = Path(output_path) / metric_type
                            save_image(image_hat[0], out_dir / image_name)

                    if compute_metrics:
                        metrics = get_metrics(ground_truth, image_hat)

                        # Accumulate metrics
                        if metric_type not in metrics_accumulated:
                            metrics_accumulated[metric_type] = metrics
                        else:
                            for metric in metrics:
                                metrics_accumulated[metric_type][metric] += metrics[
                                    metric
                                ]

                        if metric_type not in metrics_individual:
                            metrics_individual[metric_type] = {}
                        metrics_individual[metric_type][image_name] = metrics.copy()

        # Average metrics over all images
        for metric_type, metrics in metrics_accumulated.items():
            metrics_accumulated[metric_type] = {
                key: value / num_images for key, value in metrics.items()
            }

        return metrics_accumulated, metrics_individual

    def dump_images(
        self,
        dataset: LoadedDataset,
        num_dump: int | None = None,
        output_path: Path | None = None,
    ) -> None:
        num_val_frames = dataset.images.shape[0]
        _, _, h, w = dataset.images.shape
        if num_dump is None:
            # Full list of indices
            indices = list(range(num_val_frames))
        else:
            # Evenly spaced indices, cast to Python ints
            indices = torch.linspace(
                0,
                num_val_frames,
                steps=num_dump,
                dtype=torch.int32,
            ).tolist()[:-1]

        for near_plane in self.cfg.eval_near_planes:
            metric_type = f"layered_{near_plane}"
            with torch.no_grad():
                images = []
                for index in indices:
                    image = self.render(
                        dataset.extrinsics[index : index + 1].to(self.device),
                        dataset.intrinsics[index : index + 1].to(self.device),
                        (h, w),
                        mode="layered",
                    )
                    images.append(image)
                images = torch.cat(images)

                if output_path is not None:
                    out_dir = Path(output_path) / metric_type
                    for i, image_hat in zip(indices, images):
                        image_name = dataset.image_paths[i].name
                        save_image(image_hat, out_dir / image_name)

    def render_trajectory(self, dataset: LoadedDataset, tag: str, num_dump=None):
        num_images = dataset.images.shape[0]
        _, _, h, w = dataset.images.shape

        if num_dump is None:
            # Full list of indices
            vis_indices = list(range(num_images))
        else:
            # Evenly spaced indices, cast to Python ints
            vis_indices = torch.linspace(
                0,
                num_images,
                steps=num_dump,
                dtype=torch.int32,
            ).tolist()[:-1]

        output = {}
        # Iterate over images one by one
        # for i in range(num_images):
        for i in vis_indices:
            extrinsics = dataset.extrinsics[i : i + 1].to(self.device)
            intrinsics = dataset.intrinsics[i : i + 1].to(self.device)
            image_name = dataset.image_paths[i].name

            # full spin loop for this single view
            spin_loop = generate_spin_wave_loop(extrinsics, 384).squeeze(0)
            # expand intrinsics to match number of frames
            intrinsics_expanded = intrinsics.expand(spin_loop.shape[0], -1, -1)

            for near_plane in self.cfg.eval_near_planes:
                metric_type = f"surface_{near_plane}"
                frames = []

                # render one frame at a time
                for t in range(spin_loop.shape[0]):
                    with torch.no_grad():
                        # single-frame batch
                        frame_tensor = self.render(
                            spin_loop[t : t + 1],
                            intrinsics_expanded[t : t + 1],
                            (h, w),
                            mode="surface",
                            near_plane=near_plane,
                        )

                        # clamp & convert to uint8 on CPU
                        frame = (frame_tensor.clip(0, 1) * 255).type(torch.uint8)
                        frame = frame.squeeze(0)  # [3, H, W]
                        frame = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                        frames.append(frame)

                        # optionally free up any lingering GPU memory
                        del frame_tensor

                key = f"{metric_type}_{image_name}"
                output[key] = frames

        return output
