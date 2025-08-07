from dataclasses import dataclass
from functools import wraps
from math import prod
from typing import TypeVar

from torch.profiler import record_function as record_function_torch

T = TypeVar("T")


def ceildiv(numerator: T, denominator: T) -> T:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class TileGrid:
    # One tile's shape in pixels as (height, width).
    tile_shape: tuple[int, int]

    # The entire image's shape in pixels as (height, width).
    image_shape: tuple[int, int]

    # The number of cameras being rendered.
    num_cameras: int

    # The number of shells being rendered.
    num_shells: int

    @property
    def grid_shape(self) -> tuple[int, int]:
        # The shape of the grid in terms of (rows, columns).
        return tuple(
            ceildiv(image_length, tile_length)
            for image_length, tile_length in zip(self.image_shape, self.tile_shape)
        )

    @property
    def num_tiles_per_image(self) -> int:
        return prod(self.grid_shape)

    @property
    def tile_numel(self) -> int:
        h, w = self.tile_shape
        return h * w


def record_function(tag: str):
    """This functions the same way as torch.profiler.record_function, except that it
    properly maintains the function signature for VS Code autocomplete.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with record_function_torch(tag):
                return f(*args, **kwargs)

        return wrapper

    return decorator
