import os
import random
from itertools import product
from pathlib import Path

import svg
import torch
from jaxtyping import Bool, Float, install_import_hook
from skimage import measure
from torch import Tensor

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper",),
    ("beartype", "beartype"),
):
    from .common import draw_arrow

WIDTH = 512
GRID_LENGTH = 120
HEIGHT = GRID_LENGTH + 2
ARROW = 5

GRID_RESOLUTION = 12

LEVEL_SETS = [-0.05, 0.0, 0.05]
COLORS = ["#4565E3", "#718AEA", "#9DAEF0"]

random.seed(1)


def evaluate_sdf(
    xy: Float[Tensor, "*batch xy=2"],
    radius: float = 0.2,
    distance: float = 0.1,
) -> Float[Tensor, "*batch z="]:
    a = (xy - 0.5 - distance).norm(dim=-1) - radius
    b = (xy - 0.5 + distance).norm(dim=-1) - radius
    return torch.minimum(a, b)


def sample_grid(
    resolution: int,
    device: torch.device,
) -> Float[Tensor, "height width xy=2"]:
    xy = torch.linspace(0, 1, resolution + 1, device=device, dtype=torch.float32)
    return torch.stack(torch.meshgrid(xy, xy, indexing="xy"), dim=-1)


def draw_grid() -> svg.G:
    horizontal = [
        svg.Line(
            x1=0,
            y1=GRID_LENGTH / GRID_RESOLUTION * y,
            x2=GRID_LENGTH,
            y2=GRID_LENGTH / GRID_RESOLUTION * y,
            stroke_width=1 if (y == 0 or y == GRID_RESOLUTION) else 0.25,
            stroke="black",
            stroke_linecap="square",
        )
        for y in range(0, GRID_RESOLUTION + 1)
    ]
    vertical = [
        svg.Line(
            x1=GRID_LENGTH / GRID_RESOLUTION * x,
            y1=0,
            x2=GRID_LENGTH / GRID_RESOLUTION * x,
            y2=GRID_LENGTH,
            stroke_width=1 if (x == 0 or x == GRID_RESOLUTION) else 0.25,
            stroke="black",
            stroke_linecap="square",
        )
        for x in range(0, GRID_RESOLUTION + 1)
    ]
    return svg.G(elements=horizontal + vertical)


def draw_shells() -> svg.G:
    elements = [draw_grid()]

    xy = sample_grid(GRID_RESOLUTION, torch.device("cuda:0"))
    sdf = evaluate_sdf(xy)

    for level_set, color in zip(LEVEL_SETS[::-1], COLORS):
        contours = measure.find_contours(sdf.cpu().numpy(), level_set)
        for contour in contours:
            contour *= GRID_LENGTH / GRID_RESOLUTION
            contour = contour.tolist()

            x, y = contour[0]
            path = [f"M {x} {y}"]
            for x, y in contour[1:]:
                path.append(f"L {x} {y}")

            elements.append(
                svg.Path(
                    d=" ".join(path) + " Z",
                    fill=color,
                    stroke="black",
                    stroke_width=1.0,
                )
            )

    return svg.G(elements=elements)


def compute_occupancy() -> Bool[Tensor, "height width"]:
    xy = sample_grid(GRID_RESOLUTION, torch.device("cuda:0"))
    sdf = evaluate_sdf(xy)
    occupancy = torch.zeros(
        (GRID_RESOLUTION, GRID_RESOLUTION),
        dtype=torch.bool,
        device="cuda",
    )
    for level_set in LEVEL_SETS:
        contours = measure.find_contours(sdf.cpu().numpy(), level_set)
        for lines in contours:
            centers = 0.5 * (lines[:-1] + lines[1:])
            centers = torch.tensor(centers, dtype=torch.float32, device="cuda")
            x, y = centers.type(torch.int32).unbind(dim=-1)
            occupancy[y, x] = True
    return occupancy


def draw_occupancy() -> svg.G:
    elements = [draw_grid()]

    occupancy = compute_occupancy()
    for row, row_values in enumerate(occupancy):
        for col, value in enumerate(row_values):
            if value:
                x = GRID_LENGTH / GRID_RESOLUTION * col
                y = GRID_LENGTH / GRID_RESOLUTION * row
                elements.append(
                    svg.Rect(
                        x=x,
                        y=y,
                        width=GRID_LENGTH / GRID_RESOLUTION,
                        height=GRID_LENGTH / GRID_RESOLUTION,
                        fill="black",
                    )
                )

    return svg.G(elements=elements)


def draw_extracted_voxels() -> svg.G:
    elements = []

    occupancy = compute_occupancy()
    vertex_occupancy = occupancy.clone()
    vertex_occupancy[:-1] |= occupancy[1:]
    vertex_occupancy[:, 1:] |= occupancy[:, :-1]
    vertex_occupancy[:-1, 1:] |= occupancy[1:, :-1]

    h, w = occupancy.shape

    for row in range(h):
        for col in range(w):
            if not vertex_occupancy[row, col]:
                continue
            x = GRID_LENGTH / GRID_RESOLUTION * col
            y = GRID_LENGTH / GRID_RESOLUTION * row
            elements.append(
                svg.Rect(
                    x=x,
                    y=y,
                    width=GRID_LENGTH / GRID_RESOLUTION,
                    height=GRID_LENGTH / GRID_RESOLUTION,
                    fill="#4565E3" if occupancy[row, col] else "#BDC8F5",
                )
            )

    for row in range(h):
        for col in range(w):
            if not vertex_occupancy[row, col]:
                continue
            x = GRID_LENGTH / GRID_RESOLUTION * col
            y = GRID_LENGTH / GRID_RESOLUTION * row
            elements.append(
                svg.Circle(
                    cx=x,
                    cy=y + GRID_LENGTH / GRID_RESOLUTION,
                    r=2,
                    fill="black",
                )
            )

    return svg.G(elements=elements + [draw_grid()])


def draw_label(
    text: str,
    x: float,
    y: float,
    width: float | None = None,
    rotation: float = 0,
) -> svg.G:
    elements = []
    height = 16
    if width is not None:
        elements.append(
            svg.Rect(
                x=-width / 2,
                y=-height / 2,
                width=width,
                height=height,
                fill="white",
                stroke="black",
                stroke_width=1.0,
            )
        )
    return svg.G(
        elements=[
            *elements,
            svg.Text(
                x=0,
                y=11 - height / 2,
                text=text,
                font_size=9,
                text_anchor="middle",
                fill="black",
            ),
        ],
        transform=f"translate({x}, {y}) rotate({rotation})",
    )


def draw_parameters(
    x: float,
    y: float,
    rows: int,
    cols: int,
    color_fn,
    length: float,
) -> svg.G:
    return svg.G(
        elements=[
            svg.Rect(
                x=x - (cols * length / 2) + j * length,
                y=y - (rows * length / 2) + i * length,
                width=length,
                height=length,
                fill=color_fn(i, j),
                stroke="black",
                stroke_width=0.25,
            )
            for i, j in product(range(rows), range(cols))
        ]
    )


if __name__ == "__main__":
    LEFT_OF_OCCUPANCY = 1
    RIGHT_OF_OCCUPANCY = LEFT_OF_OCCUPANCY + GRID_LENGTH
    LEFT_OF_VOXELS = RIGHT_OF_OCCUPANCY + 40
    RIGHT_OF_VOXELS = LEFT_OF_VOXELS + GRID_LENGTH
    RIGHT_OF_SHELLS = WIDTH - 1
    LEFT_OF_SHELLS = RIGHT_OF_SHELLS - GRID_LENGTH

    occupancy = draw_occupancy()
    occupancy.transform = f"translate({LEFT_OF_OCCUPANCY}, 1)"

    extracted_voxels = draw_extracted_voxels()
    extracted_voxels.transform = f"translate({LEFT_OF_VOXELS}, 1)"

    shells = draw_shells()
    shells.transform = f"translate({(LEFT_OF_SHELLS)}, 1)"

    canvas = svg.SVG(
        width=WIDTH,
        height=HEIGHT + 14,
        elements=[
            occupancy,
            extracted_voxels,
            shells,
            draw_arrow(
                RIGHT_OF_OCCUPANCY + ARROW,
                HEIGHT / 2,
                LEFT_OF_VOXELS - ARROW,
                HEIGHT / 2,
                3,
                1,
            ),
            draw_arrow(
                RIGHT_OF_VOXELS + ARROW,
                HEIGHT - 20,
                LEFT_OF_SHELLS - ARROW,
                HEIGHT - 20,
                3,
                1,
            ),
            draw_label(
                "Voxel Extraction",
                (RIGHT_OF_OCCUPANCY + LEFT_OF_VOXELS) / 2,
                HEIGHT / 2,
                75.0,
                -90.0,
            ),
            draw_label(
                "Marching Cubes",
                (RIGHT_OF_VOXELS + LEFT_OF_SHELLS) / 2,
                HEIGHT - 20,
                75.0,
                0.0,
            ),
            draw_label(
                "Explicit Parameters",
                (RIGHT_OF_VOXELS + LEFT_OF_SHELLS) / 2,
                17,
                None,
                0.0,
            ),
            draw_parameters(
                (RIGHT_OF_VOXELS + LEFT_OF_SHELLS) / 2 + 4,
                50,
                12,
                4,
                lambda i, j: random.choice(COLORS),
                4,
            ),
            draw_parameters(
                (RIGHT_OF_VOXELS + LEFT_OF_SHELLS) / 2 - 10,
                50,
                12,
                1,
                lambda i, j: random.choice(
                    ["#555555", "#666666", "#777777", "#888888"]
                ),
                4,
            ),
            draw_label(
                "SDF",
                (RIGHT_OF_VOXELS + LEFT_OF_SHELLS) / 2 - 20,
                50,
                None,
                -90.0,
            ),
            draw_label(
                "SH Coeff.",
                (RIGHT_OF_VOXELS + LEFT_OF_SHELLS) / 2 + 20,
                50,
                None,
                -90.0,
            ),
            draw_arrow(
                (RIGHT_OF_VOXELS + LEFT_OF_SHELLS) / 2,
                78,
                (RIGHT_OF_VOXELS + LEFT_OF_SHELLS) / 2,
                HEIGHT - 20 - 16 / 2 - ARROW,
                3,
                1,
            ),
            draw_label(
                "Bit-Packed Active Grid",
                (LEFT_OF_OCCUPANCY + RIGHT_OF_OCCUPANCY) / 2,
                HEIGHT + 8,
                None,
                0.0,
            ),
            draw_label(
                "Sparse Voxels and Vertices",
                (LEFT_OF_VOXELS + RIGHT_OF_VOXELS) / 2,
                HEIGHT + 8,
                None,
                0.0,
            ),
            draw_label(
                "Extracted Mesh Shells",
                (LEFT_OF_SHELLS + RIGHT_OF_SHELLS) / 2,
                HEIGHT + 8,
                None,
                0.0,
            ),
        ],
    )

    base = "generated_paper_components/figure_marching_cubes"
    path = Path(f"{base}.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(canvas.as_str())

    os.system(f"rsvg-convert -f pdf -o {base}.pdf {base}.svg")
