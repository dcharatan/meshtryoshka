from pathlib import Path

import numpy as np
import torch
import torchvision
from einops import rearrange, repeat
from jaxtyping import Float, Shaped
from PIL import Image
from torch import Tensor


def load_image(
    path: Path | str,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "channel height width"]:
    image = np.array(Image.open(path), dtype=np.float32) / 255
    image = rearrange(image, "h w c -> c h w")
    return torch.tensor(image, dtype=torch.float32, device=device)


def save_image(
    image: (
        Shaped[Tensor | np.ndarray, "channel height width"]
        | Shaped[Tensor | np.ndarray, "height width"]
    ),
    path: Path | str,
) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy()
    image = image.astype(np.float32)
    image = np.clip(image, a_min=0, a_max=1)
    if image.ndim == 2:
        image = repeat(image, "h w -> h w c", c=3)
    else:
        image = rearrange(image, "c h w -> h w c")
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def save_video(
    video: Float[Tensor, "time rgb=3 height width"],
    path: Path | str,
    fps: int = 30,
) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Convert the video to byte THWC.
    video = rearrange(video, "t c h w -> t h w c").clip(min=0, max=1)
    video = (video * 255).byte()

    # Expects shape frames, resolution, resolution, channels
    def crop_height_even(video):
        f, h, w, c = video.shape
        if h % 2 == 1:  # If height is odd, crop by removing the last row
            video = video[:, :-1, :, :]
        return video

    video = crop_height_even(video)
    torchvision.io.write_video(path, video, fps=fps)
