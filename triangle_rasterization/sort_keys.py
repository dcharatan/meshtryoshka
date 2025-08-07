from pathlib import Path

import torch
from torch.utils import cpp_extension

from .common import TileGrid, record_function
from .compilation import wrap_compilation
from .generate_keys import PairedKeys

cuda = wrap_compilation(
    lambda: cpp_extension.load(
        name="sort_keys",
        sources=[Path(__file__).parent / "sort_keys.cu"],
    )
)


@record_function("sort_keys")
def sort_keys(paired_keys: PairedKeys, grid: TileGrid) -> PairedKeys:
    keys, triangle_indices = cuda().sort_keys(
        paired_keys.keys,
        paired_keys.triangle_indices,
        paired_keys.num_keys,
    )

    # The sorting may happen in a different stream, so this may be necessary.
    torch.cuda.synchronize(paired_keys.keys.device)

    return PairedKeys(keys, triangle_indices)
