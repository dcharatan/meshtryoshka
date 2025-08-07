import os
import shutil
from pathlib import Path
from typing import Callable

import svg
from jaxtyping import install_import_hook
from PIL import Image

from .results_paths import ABLATIONS_PATHS

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper",),
    ("beartype", "beartype"),
):
    from .common import flexible_crop

WIDTH = 240
MARGIN = 2


def draw_grid(
    width: int,
    cols: int,
    headings: list[str],
    aspect_ratio: float,
    load_fn: Callable[[str], Path],
    crops: tuple[float, float, float],
) -> tuple[svg.G, float]:
    label_margin = 15
    image_width = (width - (cols - 1) * MARGIN) / cols
    image_height = image_width / aspect_ratio
    temp_path = Path("generated_paper_components/figure_ablations_images")
    shutil.rmtree(temp_path, ignore_errors=True)
    temp_path.mkdir(parents=True, exist_ok=True)

    elements = []
    scale, dx, dy = crops
    for index, heading in enumerate(headings):
        row = index // cols
        col = index % cols
        offset = row * (image_height + MARGIN + label_margin)

        label = svg.Text(
            x=col * (image_width + MARGIN) + image_width / 2,
            y=offset + 10,
            text=heading,
            font_size=9,
            text_anchor="middle",
            fill="black",
        )
        elements.append(label)

        try:
            modified_image_path = temp_path / f"{heading}_{row}.png"
            path = load_fn(heading)
            image = Image.open(path).convert("RGB")
            image = flexible_crop(
                image,
                aspect_ratio,
                scale,
                dx,
                dy,
                max_resolution=400,
            )
            image.save(modified_image_path)
            image = svg.Image(
                x=col * (image_width + MARGIN),
                y=offset + label_margin,
                width=image_width,
                height=image_height,
                href=str(Path(modified_image_path).absolute()),
            )
            elements.append(image)
        except Exception:
            # If the image can't be loaded, use a placeholder.
            image = svg.Rect(
                x=col * (image_width + MARGIN),
                y=offset + label_margin,
                width=image_width,
                height=image_height,
                fill="#eeeeee",
            )
            elements.append(image)

    num_rows = (len(headings) - 1) // cols + 1
    height = num_rows * (image_height + label_margin) + (num_rows - 1) * MARGIN
    return svg.G(elements=elements), height


def load_ablation(method: str) -> Path:
    return ABLATIONS_PATHS[method]


ABLATIONS = [
    "Ours",
    "3 Shells",
    "11 Shells",
    "No Exponential",
    "No Sparsity",
    "No Regularizers",
    "No Frustums",
    "No SH Coeffs.",
]

if __name__ == "__main__":
    grid, height = draw_grid(
        WIDTH,
        4,
        ABLATIONS,
        1.0,
        load_ablation,
        (0.25, 0.6, 0.28),
    )

    canvas = svg.SVG(
        width=WIDTH,
        height=height,
        elements=[grid],
    )

    base = "generated_paper_components/figure_ablations"
    path = Path(f"{base}.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(canvas.as_str())

    os.system(f"rsvg-convert -f pdf -o {base}.pdf {base}.svg")
