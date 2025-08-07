import codecs
from io import BytesIO
from typing import Literal, Union

import numpy as np
import svg
import torch
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]

BLUE = [
    "#4565E3",
    "#718BE9",
    "#9FAFEF",
    "#CCD4F5",
    "#E8EBF9",
]

RED = [
    "#DD3836",
    "#E46968",
    "#EB9999",
    "#F2C9CA",
    "#F7E7E7",
]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def encode_image(
    image: Float[Tensor, "3 height width"],
    image_format: Literal["png", "jpeg"] = "png",
) -> str:
    stream = BytesIO()
    Image.fromarray(prep_image(image)).save(stream, image_format)
    stream.seek(0)
    base64str = codecs.encode(stream.read(), "base64").rstrip()
    return f"data:image/{image_format};base64,{base64str.decode('ascii')}"


def draw_arrow(
    x1: float | int,
    y1: float | int,
    x2: float | int,
    y2: float | int,
    radius: float | int,
    stroke_width: float | int,
) -> svg.G:
    p1 = np.array([x1, y1], dtype=np.float32)
    p2 = np.array([x2, y2], dtype=np.float32)
    delta = p2 - p1
    ax, ay = (delta / np.linalg.norm(delta) * radius).tolist()

    return svg.G(
        elements=[
            svg.Line(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke="black",
                stroke_width=stroke_width,
                stroke_linecap="round",
            ),
            svg.Line(
                x1=x2,
                y1=y2,
                x2=x2 - ax + ay,
                y2=y2 - ay - ax,
                stroke="black",
                stroke_width=stroke_width,
                stroke_linecap="round",
            ),
            svg.Line(
                x1=x2,
                y1=y2,
                x2=x2 - ax - ay,
                y2=y2 - ay + ax,
                stroke="black",
                stroke_width=stroke_width,
                stroke_linecap="round",
            ),
        ],
    )


def flexible_crop(
    image: Image.Image,
    aspect_ratio: float,
    fraction: float = 1.0,
    dx: float = 0.5,
    dy: float = 0.5,
    max_resolution: int | None = None,
) -> Image.Image:
    """
    Crop an image with specified aspect ratio, size fraction, and position offsets.

    Args:
        image: PIL Image object
        aspect_ratio: float, desired width/height ratio
        fraction: float, fraction of maximum possible crop size (0.0 to 1.0)
        dx: float, horizontal position offset (0.0 = left, 1.0 = right)
        dy: float, vertical position offset (0.0 = top, 1.0 = bottom)
        max_resolution: Optional[int], maximum pixels on the longer axis for final
        downscaling

    Returns:
        PIL Image object (cropped and optionally downscaled)
    """
    img_width, img_height = image.size

    # Calculate maximum crop dimensions that fit in the image
    if img_width / img_height > aspect_ratio:
        # Image is wider than desired aspect ratio
        max_crop_height = img_height
        max_crop_width = int(max_crop_height * aspect_ratio)
    else:
        # Image is taller than desired aspect ratio
        max_crop_width = img_width
        max_crop_height = int(max_crop_width / aspect_ratio)

    # Scale by the fraction parameter
    crop_width = int(max_crop_width * fraction)
    crop_height = int(max_crop_height * fraction)

    # Calculate the range of possible positions for the crop
    max_left = img_width - crop_width
    max_top = img_height - crop_height

    # Calculate actual position based on dx, dy parameters
    left = int(max_left * dx)
    top = int(max_top * dy)

    # Ensure we don't go out of bounds
    left = max(0, min(left, max_left))
    top = max(0, min(top, max_top))

    # Calculate crop box coordinates
    right = left + crop_width
    bottom = top + crop_height

    # Perform the crop
    cropped_image = image.crop((left, top, right, bottom))

    # Apply optional downscaling
    if max_resolution is not None:
        current_width, current_height = cropped_image.size
        longer_axis = max(current_width, current_height)

        if longer_axis > max_resolution:
            # Calculate scaling factor
            scale_factor = max_resolution / longer_axis
            new_width = int(current_width * scale_factor)
            new_height = int(current_height * scale_factor)

            # Downscale using Lanczos resampling
            cropped_image = cropped_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

    return cropped_image
