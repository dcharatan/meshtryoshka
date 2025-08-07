from pathlib import Path

import slangtorch
import torch
from jaxtyping import Int64
from torch import Tensor

from .common import TileGrid, ceildiv, record_function
from .compilation import wrap_compilation
from .generate_keys import PairedKeys

slang = wrap_compilation(
    lambda key_defines: slangtorch.loadModule(
        str(Path(__file__).parent / "delineate_tiles.slang"),
        verbose=True,
        defines=dict(key_defines),
    )
)


BLOCK_SIZE = 256


@record_function("delineate_tiles")
def delineate_tiles(
    sorted_paired_keys: PairedKeys,
    grid: TileGrid,
    key_defines: tuple[tuple[str, int], ...],
    separate_shells: bool,
    epsilon: float = 1e-8,
) -> Int64[Tensor, "batch shell tile boundary=2"]:
    device = sorted_paired_keys.keys.device

    # For the sake of simplicity, the shell dimension in the delineated tiles becomes
    # a singleton dimension if the shells shouldn't be separated.
    shape = [grid.num_cameras, grid.num_shells, grid.num_tiles_per_image, 2]
    if not separate_shells:
        shape[1] = 1

    tile_boundaries = torch.zeros(shape, dtype=torch.int64, device=device)
    num_keys = sorted_paired_keys.num_keys

    if num_keys > 0:
        slang(key_defines).delineate_tiles(
            sorted_keys=sorted_paired_keys.keys,
            num_keys_container=torch.tensor(num_keys, dtype=torch.int64, device=device),
            out_tile_boundaries=tile_boundaries,
            separate_shells=separate_shells,
            epsilon=epsilon,
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(num_keys, BLOCK_SIZE), 1, 1),
        )

    return tile_boundaries
