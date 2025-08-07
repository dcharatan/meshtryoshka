from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float32, Int32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "interpolate.slang"), verbose=False
    )
)

BLOCK_HEIGHT: int = 16
BLOCK_WIDTH: int = 16


class Interpolate(torch.autograd.Function):
    @record_function("interpolate_forward")
    @staticmethod
    def forward(
        ctx,
        uv: Float32[Tensor, "batch shell height width uv=2"],
        index: Int32[Tensor, "batch shell height width"],
        faces: Int32[Tensor, "face corner=3"],
        signed_distances: Float32[Tensor, " vertex"],
        colors: Float32[Tensor, "batch vertex rgb=3"],
    ) -> tuple[
        Float32[Tensor, "batch shell height width"],  # signed distance
        Float32[Tensor, "batch shell height width rgb=3"],  # color
    ]:
        device = uv.device
        b, s, h, w, _ = uv.shape
        bs = b * s

        kwargs = dict(device=device, dtype=torch.float32)
        out_signed_distances = torch.empty((bs, h, w), **kwargs)
        out_colors = torch.empty((bs, h, w, 3), **kwargs)

        slang().interpolate(
            uv=uv.view((bs, h, w, 2)),
            index=index.view((bs, h, w)),
            faces=faces,
            signed_distances=signed_distances,
            colors=colors,
            out_signed_distances=out_signed_distances,
            out_colors=out_colors,
            num_shells=s,
        ).launchRaw(
            blockSize=(BLOCK_WIDTH, BLOCK_HEIGHT, 1),
            gridSize=(
                ceildiv(w, BLOCK_WIDTH),
                ceildiv(h, BLOCK_HEIGHT),
                bs,
            ),
        )

        ctx.save_for_backward(
            uv,
            index,
            faces,
            signed_distances,
            colors,
            out_signed_distances,
            out_colors,
        )

        return (
            out_signed_distances.view(b, s, h, w),
            out_colors.view(b, s, h, w, 3),
        )

    @record_function("interpolate_backward")
    @staticmethod
    def backward(
        ctx,
        out_signed_distances_grad: Float32[Tensor, "batch shell height width"],
        out_colors_grad: Float32[Tensor, "batch shell height width rgb=3"],
    ):
        (
            uv,
            index,
            faces,
            signed_distances,
            colors,
            out_signed_distances,
            out_colors,
        ) = ctx.saved_tensors
        b, s, h, w, _ = uv.shape
        bs = b * s

        out_signed_distances_grad = out_signed_distances_grad.contiguous().view(
            bs, h, w
        )
        out_colors_grad = out_colors_grad.contiguous().view(bs, h, w, 3)

        signed_distances_grad = torch.zeros_like(signed_distances)
        colors_grad = torch.zeros_like(colors)

        slang().interpolate.bwd(
            uv=uv.view((bs, h, w, 2)),
            index=index.view((bs, h, w)),
            faces=faces,
            signed_distances=(signed_distances, signed_distances_grad),
            colors=(colors, colors_grad),
            out_signed_distances=(out_signed_distances, out_signed_distances_grad),
            out_colors=(out_colors, out_colors_grad),
            num_shells=s,
        ).launchRaw(
            blockSize=(BLOCK_WIDTH, BLOCK_HEIGHT, 1),
            gridSize=(
                ceildiv(w, BLOCK_WIDTH),
                ceildiv(h, BLOCK_HEIGHT),
                bs,
            ),
        )

        return None, None, None, signed_distances_grad, colors_grad


def interpolate(
    uv: Float32[Tensor, "batch shell height width uv=2"],
    index: Int32[Tensor, "batch shell height width"],
    faces: Int32[Tensor, "face corner=3"],
    signed_distances: Float32[Tensor, " vertex"],
    colors: Float32[Tensor, "batch vertex rgb=3"],
) -> tuple[
    Float32[Tensor, "batch shell height width"],  # signed distance
    Float32[Tensor, "batch shell height width rgb=3"],  # color
]:
    return Interpolate.apply(uv, index, faces, signed_distances, colors)
