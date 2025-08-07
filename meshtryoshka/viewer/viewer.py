import logging
import threading
import time
from multiprocessing.synchronize import Lock
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch
import torchvision
import viser
import viser.transforms as vtf
from jaxtyping import Float
from viser import ClientHandle, GuiButtonHandle, GuiDropdownHandle, ViserServer

from ..model.common import SectionParams
from .camera import ViewerCamera
from .client import ClientThread

WorkFn = Callable[[], None]


@runtime_checkable
class RenderFn(Protocol):
    def __call__(
        self,
        camera: ViewerCamera,
        render_mode: str,
    ) -> Float[np.ndarray, "height width rgb=3"]:
        pass


class Viewer:
    client_threads: dict[int, ClientThread]
    render_fn: RenderFn
    work_fn: WorkFn | None
    lock: Lock
    server: ViserServer
    is_training: bool
    button: GuiButtonHandle
    render_mode_dropdown: GuiDropdownHandle

    def __init__(
        self,
        render_fn: RenderFn,
        render_modes: tuple[str, ...],
        train_dataset,
        work_fn: WorkFn | None = None,
    ) -> None:
        self.is_training = False
        self.render_fn = render_fn
        self.work_fn = work_fn
        self.client_threads = {}
        self.lock = threading.Lock()
        self.server = ViserServer()
        self.server.on_client_connect(self.on_client_connect)
        self.server.on_client_disconnect(self.on_client_disconnect)
        self.button = self.server.gui.add_button("Toggle Training")
        self.render_mode_dropdown = self.server.gui.add_dropdown(
            "Render Mode",
            render_modes,
            "surface",
        )
        self.cross_section_axis = self.server.gui.add_dropdown(
            "Cross Section Axis",
            ("x", "y", "z"),
            "x",
        )
        self.cross_section_max = self.server.gui.add_slider(
            "Cross Section Max",
            0.0,
            1.0,
            0.05,
            1.0,
        )
        self.cross_section_min = self.server.gui.add_slider(
            "Cross Section Min",
            0.0,
            1.0,
            0.05,
            0.0,
        )
        self.render_mode_dropdown.on_update(lambda _: self.set_needs_render())
        self.cross_section_axis.on_update(lambda _: self.set_needs_render())
        self.cross_section_max.on_update(lambda _: self.set_needs_render())
        self.cross_section_min.on_update(lambda _: self.set_needs_render())
        self.server.gui.set_panel_label(f"Training: {self.is_training}")

        self.camera_handles, self.fov = self.init_scene(train_dataset)

        # Add a single toggle button for camera visibility.
        self.show_cameras_checkbox = self.server.gui.add_checkbox("Show Cameras", True)
        self.show_cameras_checkbox.on_update(lambda _: self.update_camera_visibility())

        # Initialize the contract state
        self.contract = False
        self.toggle_contract_btn = self.server.gui.add_button(label="Contract")
        self.toggle_contract_btn.on_click(lambda _: self.toggle_contract())

    def init_scene(self, train_dataset):
        camera_handles: dict[int, viser.CameraFrustumHandle] = {}
        train_extrinsics = (
            train_dataset.extrinsics.cpu().numpy()
        )  # B, 4, 4, format is w2c
        train_intrinsics = train_dataset.intrinsics.cpu().numpy()
        train_images = train_dataset.images

        _, _, h, w = train_images.shape
        intrinsics = train_intrinsics[0]
        fx = intrinsics[0, 0].item()
        fov = float(2 * np.arctan((w / 2) / fx))
        # Aspect ratio is simply width/height.
        aspect = float(w) / float(h)

        # TODO: Maybe only select some of the images?
        # NVM, nerfstudio only filters if above 512 lmfao

        for idx in range(train_extrinsics.shape[0]):
            image = train_images[idx]
            _, h, w = image.shape
            image_uint8 = (image * 255).detach().type(torch.uint8)  # c, h, w
            w2c = train_extrinsics[idx]  # TODO: Do we need permuting?
            c2w = np.linalg.inv(w2c)
            # intrinsics = train_intrinsics[idx]

            image_uint8 = torchvision.transforms.functional.resize(
                image_uint8, 100, antialias=None
            )  # type: ignore
            image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()

            R = vtf.SO3.from_matrix(c2w[:3, :3])

            # fx = intrinsics[0, 0].item()  # focal length in pixels (horizontal)
            # # Compute horizontal field-of-view:
            # fov = float(2 * np.arctan((w / 2) / fx))
            # # Aspect ratio is simply width/height.
            # aspect = float(w) / float(h)

            camera_handle = self.server.scene.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=fov,
                scale=0.02,  # frustum size
                aspect=aspect,
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2w[:3, 3],
            )

            def on_click_callback(
                event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle],
            ) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            camera_handle.on_click(on_click_callback)

            camera_handles[idx] = camera_handle

        return camera_handles, fov

    def set_camera_visibility(self, visible: bool) -> None:
        """Toggle the visibility of the training cameras."""
        with self.server.atomic():
            for idx in self.camera_handles:
                self.camera_handles[idx].visible = visible

    def update_camera_visibility(self) -> None:
        self.set_camera_visibility(self.show_cameras_checkbox.value)
        self.set_needs_render()

    def toggle_contract(self) -> None:
        self.contract = not self.contract
        if self.contract:
            self.toggle_contract_btn.label = "Uncontract"
            self.show_cameras_checkbox.disabled = True
            self.set_camera_visibility(False)
        else:
            self.toggle_contract_btn.label = "Contract"
            self.show_cameras_checkbox.disabled = False
            self.set_camera_visibility(self.show_cameras_checkbox.value)
        self.set_needs_render()

    def on_client_connect(self, client: ClientHandle) -> None:
        logging.info(f"Client {client.client_id} connected.")

        def client_render_fn(camera: ViewerCamera):
            section_params = SectionParams(
                view_axis={"x": 0, "y": 1, "z": 2}[self.cross_section_axis.value],
                max_value=self.cross_section_max.value,
                min_value=self.cross_section_min.value,
            )
            return self.render_fn(
                camera,
                self.render_mode_dropdown.value,
                section_params,
                self.contract,
            )

        thread = ClientThread(client, self.lock, self.fov, client_render_fn)
        self.client_threads[client.client_id] = thread
        thread.start()

    def on_client_disconnect(self, client: ClientHandle) -> None:
        logging.info(f"Client {client.client_id} disconnected.")
        self.client_threads[client.client_id].stop()
        del self.client_threads[client.client_id]

    def run(self) -> None:
        while True:
            if self.button.value:
                self.is_training = not self.is_training
                self.server.gui.set_panel_label(f"Training: {self.is_training}")
                self.button.value = False

            if self.is_training:
                with self.lock:
                    self.work_fn()

            # Update the render mode.
            with self.lock:
                pass

            # This thread will eat the entire CPU without this.
            time.sleep(0.01)

    def set_needs_render(self) -> None:
        for thread in self.client_threads.values():
            thread.state = "needs_render"
