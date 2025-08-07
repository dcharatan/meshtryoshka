from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float32
from torch import Tensor

from .common import ceildiv, record_function
from .compilation import wrap_compilation

slang = wrap_compilation(
    lambda sh_degree: slangtorch.loadModule(
        str(Path(__file__).parent / "evaluate_spherical_harmonics.slang"),
        defines={"SH_DEGREE": sh_degree},
        verbose=True,
    )
)


BLOCK_SIZE = 256


class EvaluateSphericalHarmonics(torch.autograd.Function):
    @record_function("evaluate_spherical_harmonics_forward")
    @staticmethod
    def forward(
        ctx,
        vertices: Float32[Tensor, "vertex xyz=3"],
        spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
        extrinsics: Float32[Tensor, "batch 4 4"],
        sh_degree: int,
    ) -> Float32[Tensor, "batch vertex rgb=3"]:
        device = vertices.device
        b, _, _ = extrinsics.shape
        v, _ = vertices.shape

        # TODO: Get rid of this inverse.
        camera_positions = torch.inverse(extrinsics)[:, :3, 3]

        out_colors = torch.empty((b, v, 3), device=device, dtype=torch.float32)

        slang(sh_degree).evaluate_spherical_harmonics_forward(
            vertices=vertices,
            spherical_harmonics=spherical_harmonics,
            camera_positions=camera_positions,
            out_colors=out_colors,
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(v, BLOCK_SIZE), 1, 1),
        )

        ctx.save_for_backward(
            vertices,
            spherical_harmonics,
            camera_positions,
            out_colors,
        )
        ctx.sh_degree = sh_degree

        return out_colors

    @record_function("evaluate_spherical_harmonics_backward")
    @staticmethod
    def backward(
        ctx,
        out_colors_grad: Float32[Tensor, "batch vertex rgb=3"],
    ) -> tuple[
        None,
        Float32[Tensor, "sh vertex rgb=3"],
        None,
        None,
    ]:
        # Slang doesn't support non-contiguous input gradients.
        out_colors_grad = out_colors_grad.contiguous()

        # Retrieve the saved tensors and non-tensor metadata.
        (
            vertices,
            spherical_harmonics,
            camera_positions,
            out_colors,
        ) = ctx.saved_tensors

        v, _ = vertices.shape
        spherical_harmonics_grad = torch.zeros_like(spherical_harmonics)

        slang(ctx.sh_degree).evaluate_spherical_harmonics_backward(
            vertices=vertices,
            spherical_harmonics=(spherical_harmonics, spherical_harmonics_grad),
            camera_positions=camera_positions,
            out_colors=(out_colors, out_colors_grad),
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(v, BLOCK_SIZE), 1, 1),
        )

        return None, spherical_harmonics_grad, None, None


def evaluate_spherical_harmonics(
    vertices: Float32[Tensor, "vertex xyz=3"],
    spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
    extrinsics: Float32[Tensor, "batch 4 4"],
    sh_degree: int,
) -> Float32[Tensor, "batch vertex rgb=3"]:
    return EvaluateSphericalHarmonics.apply(
        vertices,
        spherical_harmonics,
        extrinsics,
        sh_degree,
    )
