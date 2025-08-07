import codecs
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Callable, Literal

import svg
from jaxtyping import UInt8, install_import_hook
from PIL import Image
from torch import Tensor

from .results_paths import MIP360_RESULTS, SYNTHETIC_RESULTS

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper",),
    ("beartype", "beartype"),
):
    from .common import flexible_crop

WIDTH = 510
MAX_HEIGHT = 615
MARGIN = 2


def encode_image_raw(
    image: UInt8[Tensor, "height width 3"],
    image_format: Literal["png", "jpeg"] = "png",
) -> str:
    stream = BytesIO()
    Image.fromarray(image).save(stream, image_format)
    stream.seek(0)
    base64str = codecs.encode(stream.read(), "base64").rstrip()
    return f"data:image/{image_format};base64,{base64str.decode('ascii')}"


def draw_grid(
    tag: str,
    width: int,
    rows: int,
    headings: list[str],
    aspect_ratio: float,
    load_fn: Callable[[str, int], Path],
    row_crops: dict[int, tuple[float, float, float]] = {},
) -> tuple[svg.G, float]:
    label_margin = 15
    cols = len(headings)
    image_width = (width - (cols - 1) * MARGIN) / cols
    image_height = image_width / aspect_ratio
    temp_path = Path(f"generated_paper_components/figure_results_images/{tag}")
    shutil.rmtree(temp_path, ignore_errors=True)
    temp_path.mkdir(parents=True, exist_ok=True)

    elements = []
    for col in range(cols):
        label = svg.Text(
            x=col * (image_width + MARGIN) + image_width / 2,
            y=10,
            text=headings[col],
            font_size=9,
            text_anchor="middle",
            fill="black",
        )
        elements.append(label)

    offset = label_margin
    for row in range(rows):
        scale, dx, dy = row_crops.get(row, (1.0, 0.0, 0.0))
        for col in range(cols):
            # Attempt to load the image.
            try:
                modified_image_path = temp_path / f"{headings[col]}_{row}.png"
                path = load_fn(headings[col], row)
                image = Image.open(path).convert("RGB")
                image = flexible_crop(image, aspect_ratio, scale, dx, dy)
                image.save(modified_image_path)
                image = svg.Image(
                    x=col * (image_width + MARGIN),
                    y=offset,
                    width=image_width,
                    height=image_height,
                    href=str(Path(modified_image_path).absolute()),
                )
                elements.append(image)
            except Exception as e:
                print(e)
                # If the image can't be loaded, use a placeholder.
                image = svg.Rect(
                    x=col * (image_width + MARGIN),
                    y=offset,
                    width=image_width,
                    height=image_height,
                    fill="#eeeeee",
                )
                elements.append(image)
        offset += image_height + MARGIN

    return svg.G(elements=elements), offset - MARGIN


def load_synthetic(method: str, row: int) -> Path:
    return SYNTHETIC_RESULTS[method][row]


def load_mip360(method: str, row: int) -> Path:
    assert MIP360_RESULTS[method][row].exists()
    return MIP360_RESULTS[method][row]


if __name__ == "__main__":
    mip360_grid, mip360_height = draw_grid(
        "mip360",
        WIDTH,
        3,
        ["Ground Truth", "Meshtryoshka (Ours)", "Zip-NeRF", "3D Gaussian Splatting"],
        1.3,
        load_mip360,
        row_crops={
            0: (0.7, 0.4, 0.2),
            1: (0.7, 0.5, 0.5),
            2: (0.3, 0.5, 0.2),
        },
    )
    spacing = 8
    synthetic_grid, synthetic_height = draw_grid(
        "synthetic",
        WIDTH,
        3,
        [
            "Ground Truth",
            "Meshtryoshka (Ours)",
            "nvdiffrec (Flexi.)",
            "nvdiffrec (DMTet)",
            "Zip-NeRF",
            "NeuS2",
            "Vol. Surfaces",
        ],
        1.0,
        load_synthetic,
        row_crops={
            0: (0.2, 0.55, 0.43),
            1: (0.3, 0.35, 0.5),
            2: (0.2, 0.5, 0.5),
        },
    )
    synthetic_grid.transform = f"translate(0, {mip360_height + spacing})"

    height = mip360_height + synthetic_height + spacing
    print(f"{height / MAX_HEIGHT * 100:.1f}% of possible height used")
    canvas = svg.SVG(
        width=WIDTH,
        height=MAX_HEIGHT,
        elements=[
            mip360_grid,
            synthetic_grid,
        ],
    )

    base = "generated_paper_components/figure_results"
    path = Path(f"{base}.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(canvas.as_str())

    os.system(f"rsvg-convert -f pdf -o {base}.pdf {base}.svg")
