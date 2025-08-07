from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from viser import CameraHandle
from viser.transforms import SO3


@dataclass(frozen=True)
class ViewerCamera:
    extrinsics: Float[np.ndarray, "4 4"]
    intrinsics: Float[np.ndarray, "3 3"]
    image_shape: tuple[int, int]


def convert_fov(fov: float, image_height: int) -> float:
    return image_height / (2 * np.tan(fov / 2))


def convert_camera(handle: CameraHandle, height: int) -> ViewerCamera:
    h = height
    w = int(h * handle.aspect)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = SO3(handle.wxyz).as_matrix()
    c2w[:3, 3] = handle.position

    f = convert_fov(handle.fov, h)
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[:2, :2] *= f
    intrinsics[0, 2] = w / 2
    intrinsics[1, 2] = h / 2

    return ViewerCamera(np.linalg.inv(c2w), intrinsics, (h, w))
