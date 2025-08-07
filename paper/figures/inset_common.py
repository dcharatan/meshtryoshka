from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class Camera:
    angle: Float[np.ndarray, "xy=2"]
    scale: Float[np.ndarray, ""]
    offset: Float[np.ndarray, "xy=2"]


DEFAULT_CAMERA = Camera(
    angle=np.array((35.0, 19.0), dtype=np.float32),
    scale=np.array(200.0 * 120 / 270),
    offset=np.array((7.0, 126.0)),
)


def project(
    points: Float[np.ndarray, "*batch xyz=3"],
    camera: Camera,
) -> Float[np.ndarray, "*batch xy=2"]:
    *batch, _ = points.shape
    points = points.reshape(-1, 3)

    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    offsets = [90, 0]
    for axis in range(2):
        axes = np.zeros(3, dtype=np.float32)
        axes[axis] = camera.angle[axis] + offsets[axis]
        points = R.from_euler("xyz", axes, degrees=True).apply(points)
        z = R.from_euler("xyz", axes, degrees=True).apply(z)

    z = z[:2]
    points = R.from_euler(
        "xyz", [0, 0, np.rad2deg(np.atan2(z[0], z[1])) + 180], degrees=True
    ).apply(points)

    points = points * camera.scale
    points = points.reshape(*batch, 3)
    return points[..., :2] + camera.offset


CUBE_VERTICES: Float[np.ndarray, "vertex xyz=3"] = np.array(
    (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
    )
)

CUBE_EDGES: Int[np.ndarray, "edge vertex=2"] = np.array(
    (
        (0, 2),
        (2, 3),
        (2, 6),
        (0, 1),
        (0, 4),
        (1, 3),
        (1, 5),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
)

AXES: Int[np.ndarray, "edge vertex=2"] = np.array(
    (
        (0, 4),
        (0, 2),
        (0, 1),
    )
)
