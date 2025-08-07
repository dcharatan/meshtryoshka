"""This file contains useful layout utilities for images. They are:

- add_border: Add a border to an image.
- cat/hcat/vcat: Join images by arranging them in a line. If the images have different
  sizes, they are aligned as specified (start, end, center). Allows you to specify a gap
  between images.
- add_label: Add a label above an image.

Images are assumed to be float32 NumPy arrays with range 0 to 1 and shape
(*batch, channel, height, width).
"""

from pathlib import Path
from string import ascii_letters, digits, punctuation
from typing import Generator, Iterable, Literal, TypeVar

import ffmpeg
import numpy as np
from einops import reduce
from jaxtyping import Float, Int, UInt8
from PIL import Image, ImageDraw, ImageFont

Alignment = Literal["start", "center", "end"]
Direction = Literal["horizontal", "vertical"]
Color = (
    Float[np.ndarray | list, "#channel"]
    | Int[np.ndarray | list, "#channel"]
    | float
    | int
)


EXPECTED_CHARACTERS = digits + punctuation + ascii_letters

T = TypeVar("T")
D = TypeVar("D")


def _pad_color(color: Color) -> Float[np.ndarray, "#channel 1 1"]:
    return np.reshape(np.array(color, dtype=np.float32), (-1, 1, 1))


def intersperse(iterable: Iterable[T], delimiter: D) -> Generator[T | D, None, None]:
    try:
        it = iter(iterable)
        yield next(it)
        for item in it:
            yield delimiter
            yield item
    except StopIteration:
        return


def direction_to_axis(direction: Direction) -> Literal[-1, -2]:
    return {
        "horizontal": -1,
        "vertical": -2,
    }[direction]


def pad(
    image: Float[np.ndarray, "*batch channel original_height original_width"],
    direction: Direction,
    align: Alignment,
    target: int,
    color: Color,
) -> Float[np.ndarray, "*batch channel padded_height padded_width"]:
    """Pad the image to the desired length on the target axis."""

    axis = direction_to_axis(direction)
    delta = target - image.shape[axis]
    before = {
        "start": 0,
        "center": delta // 2,
        "end": delta,
    }[align]

    # Create an image with the padded shape and desired color.
    padded_shape = list(image.shape)
    padded_shape[axis] = target
    padded_image = np.empty(padded_shape, dtype=image.dtype)
    padded_image[:] = _pad_color(color)

    # Insert the original image into the padded image at the correct location.
    selector = [slice(None, None) for _ in padded_shape]
    selector[axis] = slice(before, before + image.shape[axis])
    padded_image[tuple(selector)] = image

    return padded_image


def cat(
    images: Iterable[Float[np.ndarray, "*#batch #channel _ _"]],
    direction: Direction,
    align: Alignment = "center",
    gap: int = 8,
    color: Color = 1,
    border: int = 0,
) -> Float[np.ndarray, "*batch channel height width"]:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""

    # Ensure that there's at least one image.
    images = list(images)
    assert images

    # Find the axis and cross axis.
    axis = direction_to_axis(direction)
    cross_direction = {
        "horizontal": "vertical",
        "vertical": "horizontal",
    }[direction]
    cross_axis = direction_to_axis(cross_direction)

    # Pad the images along the cross axis.
    target = max(image.shape[cross_axis] for image in images)
    images = [pad(image, cross_direction, align, target, color) for image in images]

    # Intersperse separators to create gaps.
    if gap > 0:
        # Create a separator with the desired size.
        *_, channel, _, _ = images[0].shape
        separator_shape = [channel, gap, gap]
        separator_shape[cross_axis] = target
        separator = np.empty(separator_shape, dtype=images[0].dtype)
        separator[:] = _pad_color(color)

        # Insert the separator.
        images = list(intersperse(images, separator))

    # Broadcast and concatenate the images.
    broad = [image.shape[:-2] for image in images]
    broad = np.broadcast_shapes(*broad)
    images = [np.broadcast_to(image, (*broad, *image.shape[-2:])) for image in images]
    images = np.concatenate(images, axis=axis)

    # Add a border if desired.
    if border > 0:
        images = add_border(images, border, color)

    return images


def hcat(
    images: Iterable[Float[np.ndarray, "*#batch #channel _ _"]],
    align: Literal["start", "center", "end", "top", "bottom"] = "start",
    gap: int = 8,
    color: Color = 1,
    border: int = 0,
) -> Float[np.ndarray, "*batch channel height width"]:
    """Shorthand for horizontal concatenation."""
    return cat(
        images,
        "horizontal",
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "top": "start",
            "bottom": "end",
        }[align],
        gap=gap,
        color=color,
        border=border,
    )


def vcat(
    images: Iterable[Float[np.ndarray, "*#batch #channel _ _"]],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    color: Color = 1,
    border: int = 0,
) -> Float[np.ndarray, "*batch channel height width"]:
    """Shorthand for vertical concatenation."""
    return cat(
        images,
        "vertical",
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        color=color,
        border=border,
    )


def add_border(
    image: Float[np.ndarray, "*batch channel height width"],
    border: int = 8,
    color: Color = 1,
) -> Float[np.ndarray, "*batch channel new_height new_width"]:
    """Add a border to the image."""
    *batch, channel, height, width = image.shape

    # Create an empty larger image with the border color.
    padded_shape = (*batch, channel, height + 2 * border, width + 2 * border)
    padded_image = np.empty(padded_shape, dtype=image.dtype)
    padded_image[:] = _pad_color(color)

    # Paste the original image intot he padded image.
    padded_image[..., border : border + height, border : border + width] = image

    return padded_image


def draw_label(
    text: str,
    font: Path,
    font_size: int,
) -> Float[np.ndarray, "height width"]:
    """Draw a monochrome white label on a black background with no border."""
    try:
        font = ImageFont.truetype(str(font), font_size)
    except OSError:
        font = ImageFont.load_default()
    left, _, right, _ = font.getbbox(text)
    width = right - left
    _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
    height = bottom - top
    image = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="white")
    return reduce(np.array(image, dtype=np.float32) / 255, "h w c -> h w", "mean")


def add_label(
    image: Float[np.ndarray, "*batch channel width height"],
    label: str,
    font: Path = Path("assets/Inter-Regular.otf"),
    font_size: int = 24,
    font_color: Color = 0,
    background: Color = 1,
    gap: int = 4,
    align: Alignment = "left",
    border: int = 0,
) -> Float[np.ndarray, "*batch channel width_with_label height_with_label"]:
    label = draw_label(label, font, font_size)
    label = label * _pad_color(font_color) + (1 - label) * _pad_color(background)
    return vcat((label, image), align=align, gap=gap, color=background, border=border)


def write_lossless_video(
    frames: list[UInt8[np.ndarray, "height width rgb=3"]],
    output_file: Path,
    fps=24,
) -> None:
    """
    Streams a list of RGB uint8 frames into FFmpeg for lossless encoding.

    Args:
        frames (List[np.ndarray]): Each array is shape [H, W, 3], dtype uint8.
        output_file (str): Path to the generated MP4.
        fps (int): Frame rate for the output video.
    """
    # Derive frame size from the first frame
    height, width, _ = frames[0].shape

    # Launch FFmpeg process, reading raw RGB frames from stdin
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            framerate=fps,
        )
        .output(
            str(output_file),
            vcodec="libx264",  # YUV-based H.264
            crf=18,  # visually lossless
            preset="medium",  # balanced speed
        )
        .overwrite_output()  # Overwrite existing file if necessary
        .run_async(pipe_stdin=True)  # Open stdin pipe for frame streaming
    )
    # Feed each frame into FFmpeg’s stdin
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())

    process.stdin.close()
    process.wait()
