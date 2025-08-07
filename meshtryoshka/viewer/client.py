import logging
import time
from threading import Thread
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float
from viser import CameraHandle, ClientHandle

from .camera import ViewerCamera, convert_camera

# Using threading.Lock breaks beartype, since threading.Lock isn't actually a type.
Lock = Any

State = Literal["needs_render", "preview", "waiting", "static"]
TRANSITIONS: dict[State, State] = {
    "needs_render": "preview",
    "preview": "waiting",
    "waiting": "static",
    "static": "static",
}


@runtime_checkable
class ClientRenderFn(Protocol):
    def __call__(self, camera: ViewerCamera) -> Float[np.ndarray, "height width rgb=3"]:
        pass


class ClientThread(Thread):
    client: ClientHandle
    lock: Lock
    should_stop: bool
    state: State
    render_fn: ClientRenderFn
    camera: CameraHandle

    def __init__(
        self,
        client: ClientHandle,
        lock: Lock,
        fov: float,
        render_fn: ClientRenderFn,
    ) -> None:
        super().__init__()
        self.client = client
        self.lock = lock
        self.should_stop = False
        self.state = "needs_render"
        self.render_fn = render_fn
        self.camera = client.camera
        self.client.camera.on_update(self.on_camera_update)
        self.fov = fov

    def on_camera_update(self, camera: CameraHandle) -> None:
        # Acquire the client lock in order to read and write to the state.
        with self.client.atomic():
            self.state = "needs_render"
            self.camera = camera

    def run(self) -> None:
        last_render = time.time()
        while True:
            # Handle client disconnections.
            if self.should_stop:
                break

            # Render at least once per second.
            if time.time() - last_render > 5:
                self.state = "waiting"

            # Handle state transitions.
            if self.state == "static":
                time.sleep(0.01)
                continue
            if self.state == "waiting" and time.time() - last_render < 0.5:
                time.sleep(0.01)
                continue
            self.state = TRANSITIONS[self.state]
            last_render = time.time()

            # Render and display the latest image.
            start_time = time.time()
            height = 1440 if self.state == "static" else 512
            with self.lock:
                with torch.no_grad():
                    image = self.render_fn(convert_camera(self.camera, height))
            self.client.scene.set_background_image(image)
            logging.info(f"Rendered in {time.time() - start_time:.3f} seconds.")

        # Clean up.
        self.client = None
        self.lock = None

    def stop(self) -> None:
        self.should_stop = True
