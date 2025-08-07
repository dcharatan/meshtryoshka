from pathlib import Path
from typing import NamedTuple

import slangtorch
import torch
from jaxtyping import Float32, Int32, UInt8
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

BLOCK_SIZE = 256


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "count_triangle_faces.slang"),
        verbose=True,
    )
)


class CountTriangleFacesResult(NamedTuple):
    cell_codes: UInt8[Tensor, "level_set voxel"]
    face_counts: Int32[Tensor, " level_set*voxel+1"]


@record_function("count_triangle_faces")
def count_triangle_faces(
    signed_distances: Float32[Tensor, " sample"],
    lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    upper_corners: Int32[Tensor, "corner=4 voxel"],
    level_sets: Float32[Tensor, " level_set"],
) -> CountTriangleFacesResult:
    device = lower_corners.device
    _, num_voxels = upper_corners.shape
    (num_level_sets,) = level_sets.shape

    voxel_cell_codes = torch.empty(
        (num_level_sets, num_voxels),
        dtype=torch.uint8,
        device=device,
    )
    face_counts = torch.empty(
        (num_level_sets * num_voxels + 1,),
        dtype=torch.int32,
        device=device,
    )
    face_counts[-1] = 0

    slang().count_triangle_faces(
        signed_distances=signed_distances,
        lower_corners=lower_corners,
        upper_corners=upper_corners,
        level_sets=level_sets,
        face_counts=face_counts[:-1].view(num_level_sets, -1),
        voxel_cell_codes=voxel_cell_codes,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
    )

    return CountTriangleFacesResult(face_counts, voxel_cell_codes)
