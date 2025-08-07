from pathlib import Path

import slangtorch
import torch
from jaxtyping import Bool, Float32
from torch import Tensor
from torch.profiler import record_function

from .common import ceildiv
from .compilation import wrap_compilation

slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "composite.slang"),
        verbose=True,
    )
)

BLOCK_SIZE = 256


class Composite(torch.autograd.Function):
    @record_function("composite_forward")
    @staticmethod
    def forward(
        ctx,
        signed_distances: Float32[Tensor, "pixel layer"],
        colors: Float32[Tensor, "pixel layer rgb=3"],
        mask: Bool[Tensor, "pixel layer"],
        sharpness: Float32[Tensor, ""],
    ) -> tuple[
        Float32[Tensor, "pixel rgb=3"],  # color
        Float32[Tensor, " pixel"],  # transmittance
    ]:
        device = signed_distances.device
        num_pixels, _ = signed_distances.shape
        out_colors = torch.empty((num_pixels, 3), dtype=torch.float32, device=device)
        out_transmittances = torch.empty(num_pixels, dtype=torch.float32, device=device)

        slang().composite_forward(
            signed_distances=signed_distances,
            colors=colors,
            mask=mask,
            sharpness=sharpness,
            out_colors=out_colors,
            out_transmittances=out_transmittances,
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(num_pixels, BLOCK_SIZE), 1, 1),
        )

        ctx.save_for_backward(
            signed_distances,
            colors,
            mask,
            sharpness,
            out_colors,
            out_transmittances,
        )

        return out_colors, out_transmittances

    @record_function("composite_backward")
    @staticmethod
    def backward(
        ctx,
        out_colors_grad: Float32[Tensor, "pixel rgb=3"],
        out_transmittances_grad: Float32[Tensor, " pixel"],
    ) -> tuple[
        Float32[Tensor, "pixel layer"],  # gradient for signed_distance
        Float32[Tensor, "pixel layer rgb=3"],  # gradient for color
        None,
        Float32[Tensor, ""],  # gradient for sharpness
    ]:
        (
            signed_distances,
            colors,
            mask,
            sharpness,
            out_colors,
            out_transmittances,
        ) = ctx.saved_tensors
        num_pixels, _ = signed_distances.shape

        signed_distances_grad = torch.zeros_like(signed_distances)
        colors_grad = torch.zeros_like(colors)
        sharpness_grad = torch.zeros_like(sharpness)

        slang().composite_backward(
            signed_distances=(signed_distances, signed_distances_grad),
            colors=(colors, colors_grad),
            mask=mask,
            sharpness=(sharpness, sharpness_grad),
            out_colors=(out_colors, out_colors_grad),
            out_transmittances=(out_transmittances, out_transmittances_grad),
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(num_pixels, BLOCK_SIZE), 1, 1),
        )

        return (signed_distances_grad, colors_grad, None, sharpness_grad)


def composite(
    signed_distances: Float32[Tensor, "pixel layer"],
    colors: Float32[Tensor, "pixel layer rgb=3"],
    mask: Bool[Tensor, "pixel layer"],
    background: Float32[Tensor, "rgb=3"],
    sharpness: Float32[Tensor, ""],
) -> Float32[Tensor, "pixel rgb=3"]:
    out_colors, out_transmittances = Composite.apply(
        signed_distances,
        colors,
        mask,
        sharpness,
    )
    return out_colors + background * out_transmittances[:, None]
