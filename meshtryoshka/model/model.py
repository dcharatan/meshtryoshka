import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
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
        render_mode: str,
        eval: bool = False,
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
    def get_eval_render_modes(self) -> tuple[str, ...]:
        """Get the rendering modes used for evaluation."""
        pass

    @abstractmethod
    def get_viewer_render_modes(self) -> tuple[str, ...]:
        """Get the rendering modes used for the viewer."""
        pass

    def set_step(self, step: int) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def log_images_and_metrics(self, dataset: LoadedDataset, log_path: Path) -> None:
        num_images, _, h, w = dataset.images.shape
        for render_type in self.get_eval_render_modes():
            accumulated_metrics = defaultdict(list)
            individual_metrics = {}

            for i in range(num_images):
                ground_truth = dataset.images[i : i + 1].to(self.device)
                extrinsics = dataset.extrinsics[i : i + 1].to(self.device)
                intrinsics = dataset.intrinsics[i : i + 1].to(self.device)
                image_name = dataset.image_paths[i].name

                # Render and save the image.
                with torch.no_grad():
                    image_hat = self.render(
                        extrinsics,
                        intrinsics,
                        (h, w),
                        mode=render_type,
                        eval=True,
                    )
                image_name_png = f"{Path(image_name).stem}.png"
                save_image(image_hat[0], log_path / render_type / image_name_png)

                # Compute metrics.
                metrics = get_metrics(ground_truth, image_hat)
                for key, value in metrics.items():
                    accumulated_metrics[key].append(value)
                individual_metrics[image_name] = metrics

            # Save the metrics.
            average_metrics = {
                k: np.mean(v).item() for k, v in accumulated_metrics.items()
            }
            with (log_path / f"{render_type}_average.json").open("w") as f:
                json.dump(average_metrics, f, indent=4)
            with (log_path / f"{render_type}_individual.json").open("w") as f:
                json.dump(individual_metrics, f, indent=4)

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
                        eval=True,
                    )

                    # clamp & convert to uint8 on CPU
                    frame = (frame_tensor.clip(0, 1) * 255).type(torch.uint8)
                    frame = frame.squeeze(0)  # [3, H, W]
                    frame = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                    frames.append(frame)

                    # optionally free up any lingering GPU memory
                    del frame_tensor

            key = f"surface_{image_name}"
            output[key] = frames

        return output
