from dataclasses import dataclass
from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float32, Int32, Int64
from torch import Tensor

from .common import TileGrid, ceildiv, record_function
from .compilation import wrap_compilation

slang = wrap_compilation(
    lambda key_defines: slangtorch.loadModule(
        str(Path(__file__).parent / "generate_keys.slang"),
        verbose=True,
        defines=dict(key_defines),
    )
)


BLOCK_SIZE = 256


@dataclass
class PairedKeys:
    keys: Int64[Tensor, " key"]
    triangle_indices: Int32[Tensor, " key"]

    @property
    def num_keys(self) -> int:
        return self.keys.shape[0]


@record_function("generate_keys")
def generate_keys(
    depths: Float32[Tensor, "batch vertex"],
    faces: Int32[Tensor, "triangle corner=3"],
    shell_face_boundaries: Int32[Tensor, " shell"],
    tile_minima: Int32[Tensor, "batch triangle xy=2"],
    tile_maxima: Int32[Tensor, "batch triangle xy=2"],
    num_tiles_overlapped: Int32[Tensor, "batch triangle"],
    grid: TileGrid,
    key_defines: tuple[tuple[str, int], ...],
) -> PairedKeys:
    device = depths.device

    # Allocate space for the outputs and compute the offsets needed to index them.
    offsets = num_tiles_overlapped.view(-1).cumsum(dim=0, dtype=torch.int64)
    num_keys = offsets[-1].item()
    out_keys = torch.zeros(num_keys, dtype=torch.int64, device=device)
    out_triangle_indices = torch.zeros(num_keys, dtype=torch.int32, device=device)

    # Call the Slang shader to generate the keys.
    num_faces, _ = faces.shape
    slang(key_defines).generate_keys(
        depths=depths,
        faces=faces,
        shell_face_boundaries=shell_face_boundaries,
        tile_minima=tile_minima,
        tile_maxima=tile_maxima,
        offsets=offsets,
        grid_width=grid.grid_shape[1],
        out_keys=out_keys,
        out_triangle_indices=out_triangle_indices,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(num_faces, BLOCK_SIZE), 1, 1),
    )

    return PairedKeys(out_keys, out_triangle_indices)
