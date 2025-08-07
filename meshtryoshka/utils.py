from typing import TypeVar

from jaxtyping import Float
from torch import Tensor

from .metrics import compute_lpips, compute_psnr, compute_ssim


def get_metrics(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> dict[str, float]:
    metrics = {
        "psnr": compute_psnr(ground_truth, predicted).mean().item(),
        "lpips": compute_lpips(ground_truth, predicted).mean().item(),
        "ssim": compute_ssim(ground_truth, predicted).mean().item(),
    }

    return metrics


def composite_background(
    rgb: Float[Tensor, "batch rgb=3 height width"],
    alpha: Float[Tensor, "batch alpha=1 height width"],
    color: Float[Tensor, "rgb=3"],
):
    # Composite using alpha blending onto white (white = 1.0)
    composite = rgb * alpha + (1.0 - alpha) * color.view(1, 3, 1, 1)  # b 3 h w
    return composite


T = TypeVar("T")


def get_value_for_step(step: int, schedule: dict[int, T]) -> T:
    newest_step = None
    newest_value = None

    # Find the newest step that is less than or equal to the current step.
    for schedule_step, value in schedule.items():
        if step < schedule_step:
            continue
        if newest_step is None or schedule_step > newest_step:
            newest_step = schedule_step
            newest_value = value

    assert newest_value is not None
    return newest_value
