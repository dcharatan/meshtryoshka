import json
import logging
import os
import time
from dataclasses import asdict
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import yaml
from einops import rearrange
from jaxtyping import Float
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from .benchmarking import TimeRecorder
from .config import RootCfg
from .dataset import LoadedDataset, Stage, load_dataset, resize_dataset
from .image_io import save_image, save_video
from .model import Model, get_model
from .model.common import SectionParams
from .model.model_meshtryoshka import ModelMeshtryoshka
from .preemption import register_preemption_handler
from .utils import composite_background, get_value_for_step
from .viewer.camera import ViewerCamera
from .viewer.viewer import Viewer
from .visualization.layout import (
    add_border,
    add_label,
    hcat,
    vcat,
    write_lossless_video,
)


class Trainer:
    model: Model

    def __init__(self, workspace: Path, cfg: RootCfg) -> None:
        self.workspace = workspace
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rng = torch.Generator(device=torch.device(self.cfg.train.dataset_device))
        self.rng.manual_seed(cfg.seed)

        _, _, h, w = self.train_dataset.images.shape
        self.model = get_model(
            cfg.model,
            self.train_dataset.extrinsics,
            self.train_dataset.intrinsics,
            (h, w),
            self.train_dataset.point_cloud,
        ).to(self.device)

        self.step = 0
        self.model.set_step(self.step)
        self.configure_optimizer()

        self.writer = SummaryWriter(self.workspace / "tensorboard")
        self.log_dir = self.workspace / "tensorboard"

        config_dict = asdict(self.cfg)
        if not self.cfg.eval_only:
            config_path = os.path.join(self.workspace, "config.yml")
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)

        self.time_recorder = TimeRecorder()

    def configure_optimizer(self) -> None:
        # Separate parameters into different groups automatically
        base_lr = self.cfg.train.learning_rate.base

        params_high_lr = []
        params_low_lr = []

        for name, param in self.model.named_parameters():
            if hasattr(param, "custom_lr_scale"):
                params_low_lr.append(
                    {"params": param, "lr": base_lr * param.custom_lr_scale}
                )
            else:
                params_high_lr.append({"params": param, "lr": base_lr})

        # Define optimizer
        self.optimizer = Adam(params_high_lr + params_low_lr)

        # Create the learning rate scheduler
        self.lr_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: get_value_for_step(
                step,
                self.cfg.train.learning_rate.schedule,
            ),
        )

    @cached_property
    def train_dataset(self) -> LoadedDataset:
        return self.load_dataset("train")

    @cached_property
    def test_dataset(self) -> LoadedDataset:
        return self.load_dataset("test")

    @cached_property
    def val_dataset(self) -> LoadedDataset:
        return self.load_dataset("val")

    def save_checkpoint(self, path: Path) -> None:
        logging.info(f"Saving checkpoint to {path}")
        content = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "benchmarks": self.time_recorder.state_dict(),
        }
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(content, path)

    def load_checkpoint(self, path: Path) -> None:
        logging.info(f"Loading checkpoint from {path}")
        content = torch.load(path, weights_only=True)

        # Update the model's parameters to have the right shape.
        for key, value in content["model"].items():
            parameter = self.model
            for sub_key in key.split("."):
                parameter = getattr(parameter, sub_key)
            parameter.data = torch.empty_like(value)

        self.model.set_step(self.step)
        self.model.load_state_dict(content["model"])
        self.optimizer.load_state_dict(content["optimizer"])
        self.time_recorder.load_state_dict(content.get("benchmarks", {}))
        self.step = content["step"]

    def save_checkpoint_to_workspace(self) -> None:
        logging.info(f"Saving checkpoint at step {self.step}.")
        self.save_checkpoint(self.workspace / f"checkpoints/{self.step}.ckpt")

    def load_latest_checkpoint(self) -> None:
        logging.info("Attempting to load the latest checkpoint.")
        checkpoints = (self.workspace / "checkpoints").iterdir()
        steps = [int(path.stem) for path in checkpoints if path.suffix == ".ckpt"]
        if self.cfg.max_checkpoint_step is not None:
            steps = [step for step in steps if step <= self.cfg.max_checkpoint_step]
        if not steps:
            raise FileNotFoundError()

        for step in sorted(steps, reverse=True):
            logging.info(f"Attempting to load checkpoint for step {step}.")
            try:
                self.load_checkpoint(self.workspace / f"checkpoints/{step}.ckpt")
                return
            except Exception as e:
                logging.warning(f"Failed to load checkpoint for step {step}.")
                logging.warning(e)

        self.model.set_step(0)
        logging.warning("Could not load any checkpoints.")

    def load_dataset(self, stage: Stage) -> LoadedDataset:
        # Ensure train normalization is computed before val/test
        if stage != "train":
            _ = self.train_dataset  # triggers train load and defines static transform
        dataset = load_dataset(
            self.cfg.dataset,
            stage,
            device=torch.device(self.cfg.train.dataset_device),
        )
        return resize_dataset(dataset, self.cfg.train.max_image_size)

    def train(self) -> None:
        logging.info("Starting training.")

        # Load the workspace's latest checkpoint if it exists.
        try:
            logging.info("Attempting to load latest checkpoint.")
            self.load_latest_checkpoint()
        except FileNotFoundError:
            logging.info("No checkpoint found.")

        # Set up a handler for preemption.
        maybe_exit = register_preemption_handler(self.save_checkpoint_to_workspace)
        heartbeat = time.time()

        def train_work_fn() -> bool:
            nonlocal heartbeat

            # Return False if all work is complete and we should exit.
            if self.step >= self.cfg.train.num_steps:
                return False

            # Otherwise, run as usual.
            maybe_exit()
            if time.time() - heartbeat > self.cfg.train.heartbeat_interval_seconds:
                logging.info(
                    f"Entering training step {self.step} of {self.cfg.train.num_steps}."
                )
                heartbeat = time.time()

            # Render a random image.
            with self.time_recorder.record("train"):
                n, _, _, _ = self.train_dataset.images.shape
                permutation = torch.randperm(
                    n,
                    generator=self.rng,
                    device=self.train_dataset.device,
                )
                batch_size = get_value_for_step(self.step, self.cfg.train.batch_size)
                indices = permutation[:batch_size]

                if self.train_dataset.alpha_masks is not None:
                    background = torch.rand(3, device=self.train_dataset.images.device)
                    batch_images = composite_background(
                        self.train_dataset.images[indices],
                        self.train_dataset.alpha_masks[indices],
                        background,
                    )
                else:
                    batch_images = self.train_dataset.images[indices]
                    background = None

                losses, metrics = self.model.fit(
                    self.train_dataset.extrinsics[indices].to(self.device),
                    self.train_dataset.intrinsics[indices].to(self.device),
                    batch_images.to(self.device),
                    background,
                )

                # Compute the loss on that image and backpropagate it.
                self.optimizer.zero_grad()
                loss = sum(losses.values())
                loss.backward()
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0)
                self.optimizer.step()
                if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.step += 1
                if self.model.set_step(self.step):
                    # When the number of parameters changes, we need a new optimizer.
                    self.configure_optimizer()

            # Log the losses and ensure that the autograd graph is deleted.
            losses = {k: v.item() for k, v in losses.items()}
            self.writer.add_scalar("loss/total", loss.item(), global_step=self.step)
            for key, value in losses.items():
                self.writer.add_scalar(f"loss/{key}", value, global_step=self.step)
            del loss

            # Log the metrics
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, global_step=self.step)

            # Handle visualization.
            interval = self.cfg.train.visualization_interval
            if interval is not None and self.step % interval == 0:
                with torch.no_grad():
                    self.visualize(self.val_dataset, "val")
                    self.visualize(self.train_dataset, "train")

            # Handle checkpointing.
            if self.step % self.cfg.train.checkpoint_interval == 0:
                self.save_checkpoint_to_workspace()

            # Handle validation.
            if self.step % self.cfg.train.val_interval == 0:
                with torch.no_grad():
                    self.log_images_and_metrics(self.test_dataset, "test")
            if self.step == self.cfg.train.num_steps:
                with torch.no_grad():
                    self.log_images_and_metrics(self.test_dataset, "test")

            # Log system information.
            free, total = torch.cuda.mem_get_info(self.device)
            used = total - free
            gb = 1 / (1024**3)
            self.writer.add_scalar("vram/free_gb", free * gb, global_step=self.step)
            self.writer.add_scalar("vram/used_gb", used * gb, global_step=self.step)

            # Return whether to keep training.
            return self.step < self.cfg.train.num_steps

        # If only running evaluation, log images and metrics.
        if self.cfg.eval_only:
            self.log_images_and_metrics(self.test_dataset, "test")

        # If only rendering a trajectory, do that.
        elif self.cfg.render_trajectory_only:
            self.render_trajectory(self.test_dataset, "test")

        # If the viewer is enabled, run training in the asyncio event loop.
        elif self.cfg.enable_viewer:

            @torch.no_grad
            def render_fn(
                camera: ViewerCamera,
                render_mode: str,
                section_params: SectionParams,
                contract: bool,
            ) -> Float[np.ndarray, "height width rgb=3"]:
                image = self.model.render(
                    torch.tensor(camera.extrinsics, device=self.device)[None],
                    torch.tensor(camera.intrinsics, device=self.device)[None],
                    camera.image_shape,
                    mode=render_mode,
                    section_params=section_params,
                    contract=contract,
                )
                return rearrange(image, "() c h w -> h w c").cpu().numpy()

            viewer = Viewer(
                render_fn,
                self.model.get_viewer_render_modes(),
                self.train_dataset,
                work_fn=train_work_fn,
            )
            viewer.run()

        # Otherwise, run training in the main thread.
        else:
            while train_work_fn():
                pass

        if not (self.cfg.eval_only or self.cfg.render_trajectory_only):
            logging.info(f"Saving final checkpoint at step {self.step}.")
            self.save_checkpoint_to_workspace()

    def benchmark_rendering_speed(
        self,
        image_shape: tuple[int, int] = (1080, 1920),
    ) -> None:
        logging.info("Benchmarking rendering speed.")

        # Load the workspace's latest checkpoint if it exists.
        try:
            logging.info("Attempting to load latest checkpoint.")
            self.load_latest_checkpoint()
        except FileNotFoundError:
            logging.info("No checkpoint found.")

        # Rescale the intrinsics to 1080p.
        th, tw = image_shape
        n, _, h, w = self.test_dataset.images.shape
        scale_factor = max(th / h, tw / w)
        intrinsics = self.test_dataset.intrinsics.clone()
        intrinsics[:, :2] *= scale_factor
        intrinsics[:, 0, 2] -= (w * scale_factor - tw) / 2
        intrinsics[:, 1, 2] -= (h * scale_factor - th) / 2

        # This is kind of hacky, but whatever.
        assert isinstance(self.model, ModelMeshtryoshka)

        # Trigger JIT for everything.
        logging.info("Triggering JIT before benchmarking.")
        self.model.tessellator.tessellate(
            self.model.tesseract.voxels,
            self.model.tesseract.signed_distances,
            self.model.tesseract.spherical_harmonics,
        )
        scene = self.model.tessellator.tessellate_surface(
            self.model.tesseract.voxels,
            self.model.tesseract.signed_distances,
            self.model.tesseract.spherical_harmonics,
        )
        self.model.render_scene(
            self.test_dataset.extrinsics[:1],
            intrinsics[:1],
            (th, tw),
            scene,
        )

        recorder = TimeRecorder()

        # Measure layered rendering speed.
        logging.info("Measuring layered rendering speed.")
        dump_path = self.workspace / "benchmark_layered_images"
        dump_path.mkdir(exist_ok=True, parents=True)
        with recorder.record("tessellate_layered"):
            scene = self.model.tessellator.tessellate(
                self.model.tesseract.voxels,
                self.model.tesseract.signed_distances,
                self.model.tesseract.spherical_harmonics,
            )
        for i in range(n):
            with recorder.record("render_layered"):
                image = self.model.render_scene(
                    self.test_dataset.extrinsics[i : i + 1],
                    intrinsics[i : i + 1],
                    (th, tw),
                    scene,
                )
            save_image(image[0], dump_path / f"image_{i:0>3}.png")

        # Measure surface rendering speed.
        logging.info("Measuring surface rendering speed.")
        dump_path = self.workspace / "benchmark_surface_images"
        dump_path.mkdir(exist_ok=True, parents=True)
        with recorder.record("tessellate_surface"):
            scene = self.model.tessellator.tessellate_surface(
                self.model.tesseract.voxels,
                self.model.tesseract.signed_distances,
                self.model.tesseract.spherical_harmonics,
            )
        for i in range(n):
            with recorder.record("render_surface"):
                image = self.model.render_scene(
                    self.test_dataset.extrinsics[i : i + 1],
                    intrinsics[i : i + 1],
                    (th, tw),
                    scene,
                )
            save_image(image[0], dump_path / f"image_{i:0>3}.png")

        dump_path = self.workspace / "rendering_benchmark.json"
        logging.info(f"Dumping benchmark results to {dump_path}")
        with dump_path.open("w") as f:
            json.dump(recorder.state_dict(), f)

    @torch.no_grad()
    def visualize(self, dataset: LoadedDataset, stage) -> None:
        logging.info(f"Logging visualization at step {self.step}.")
        log_dir = self.log_dir / stage

        # Render comparison images for each rendering mode.
        num_val_frames, _, _, _ = dataset.images.shape
        visualization_indices = torch.linspace(
            0,
            num_val_frames - 1,
            steps=6,
            dtype=torch.int32,
            device=dataset.device,
        )
        _, _, h, w = dataset.images.shape
        images = {
            mode: self.model.render(
                dataset.extrinsics[visualization_indices].to(self.device),
                dataset.intrinsics[visualization_indices].to(self.device),
                (h, w),
                mode=mode,
            )
            for mode in self.model.get_eval_render_modes()
        }
        images["ground_truth"] = dataset.images[visualization_indices]
        images = {k: add_label(vcat(v.cpu().numpy()), k) for k, v in images.items()}

        # Save individual images.
        for key, image in images.items():
            save_image(add_border(image), log_dir / f"{key}/{self.step:0>6}.png")

        # Compile the images into a single big image and save it.
        ground_truth = images["ground_truth"]
        del images["ground_truth"]
        images = hcat([ground_truth, *images.values()], border=8)
        save_image(images, log_dir / f"collage/{self.step:0>6}.png")

        videos = {}
        for key, video in videos.items():
            save_video(video, log_dir / f"{key}/{self.step:0>6}.mp4")

        logging.info(f"Done logging visualization at step {self.step}.")

    def render_trajectory(self, dataset: LoadedDataset, tag: str) -> None:
        logging.info(f"Rendering trajectory at step {self.step}.")
        videos = self.model.render_trajectory(dataset, tag, self.cfg.train.num_dump)
        output_path = self.workspace / f"videos/{tag}_trajectory_{self.step:0>6}"
        output_path.mkdir(exist_ok=True, parents=True)
        for key, video in videos.items():
            write_lossless_video(video, output_path / f"{key}.mp4", fps=24)

    def log_images_and_metrics(
        self,
        dataset: LoadedDataset,
        tag: str,
    ) -> None:
        logging.info(f"Logging Rendered Images at step {self.step}.")
        log_path = self.workspace / f"renders/{tag}_{self.step:0>6}"
        self.model.log_images_and_metrics(dataset, log_path)
