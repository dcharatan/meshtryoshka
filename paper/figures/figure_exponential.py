import os
import sys
from pathlib import Path

import numpy as np
import svg
import torch
from jaxtyping import Float, install_import_hook
from torch import Tensor

sys.path.append("/data/scene-rep/u/charatan/projects/triangle-rasterization")

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper", "meshtryoshka"),
    ("beartype", "beartype"),
):
    from meshtryoshka.model.contraction import Contraction, ContractionCfg

    from .common import BLUE, draw_arrow

WIDTH = 240 + 4  # shift figure slightly left in latex to make line match up
HEIGHT = 80

CENTER_RESOLUTION = 6
ARM_RESOLUTION = 9
ARM_RATIO = 1.5


def get_corners() -> Float[Tensor, "i=2 j=2 xy=2"]:
    xy = torch.arange(2, dtype=torch.float32) * 2 - 1
    return torch.stack(torch.meshgrid(xy, xy, indexing="xy"), dim=-1)


def draw_contraction_grid(length: float) -> svg.G:
    # Define the contraction function.
    CENTER_EXTENT = 0.5
    cfg = ContractionCfg(1.0, ARM_RATIO, False)
    contraction = Contraction(cfg, CENTER_EXTENT)

    def uncontract(xy: Float[Tensor, "*batch xy=2"]) -> Float[Tensor, "*batch xy=2"]:
        xyz = torch.cat((xy, torch.zeros_like(xy[..., :1])), dim=-1)
        xyz = xyz * 0.5 + 0.5
        return contraction.uncontract(xyz)[..., :2]

    def uncontract_to_pixels(
        xy: Float[Tensor, "*batch xy=2"],
    ) -> Float[Tensor, "*batch xy=2"]:
        outer = uncontract(get_corners())
        xy = uncontract(xy)
        minima = outer[0, 0]
        maxima = outer[1, 1]
        xy = (xy - minima) / (maxima - minima)
        return xy * length

    # Draw the outer border.
    elements = [
        svg.Rect(
            x=0,
            y=0,
            width=length,
            height=length,
            fill="none",
            stroke="black",
            stroke_width=1.0,
        )
    ]

    # Draw the diagonal lines.
    inner = uncontract_to_pixels(get_corners() * CENTER_EXTENT)
    outer = uncontract_to_pixels(get_corners())
    for (x1, y1), (x2, y2) in zip(inner.reshape(-1, 2), outer.reshape(-1, 2)):
        elements.append(
            svg.Line(
                x1=x1.item(),
                y1=y1.item(),
                x2=x2.item(),
                y2=y2.item(),
                stroke="black",
                stroke_width=1.0,
            )
        )

    # Draw the radial frustum lines.
    order = [0, 1, 3, 2]
    for i in range(4):
        a_inner = inner.reshape(-1, 2)[order[i]]
        b_inner = inner.reshape(-1, 2)[order[i - 1]]

        a_outer = outer.reshape(-1, 2)[order[i]]
        b_outer = outer.reshape(-1, 2)[order[i - 1]]

        for j in range(1, CENTER_RESOLUTION):
            t = j / CENTER_RESOLUTION
            c = a_inner * (1 - t) + b_inner * t
            d = a_outer * (1 - t) + b_outer * t

            elements.append(
                svg.Line(
                    x1=c[0].item(),
                    y1=c[1].item(),
                    x2=d[0].item(),
                    y2=d[1].item(),
                    stroke="black",
                    stroke_width=0.25,
                )
            )

    # Draw the circular frustum lines.
    for i in range(1, ARM_RESOLUTION):
        current = get_corners() * (
            (i / ARM_RESOLUTION) * (1 - CENTER_EXTENT) + CENTER_EXTENT
        )
        current = uncontract_to_pixels(current)
        elements.append(
            svg.Rect(
                x=current[0, 0, 0].item(),
                y=current[0, 0, 1].item(),
                width=current.diff(dim=1)[0, 0, 0].item(),
                height=current.diff(dim=0)[0, 0, 1].item(),
                fill="none",
                stroke="black",
                stroke_width=0.25,
            )
        )

    # Draw the inner border.
    elements.append(
        svg.Rect(
            x=inner[0, 0, 0].item(),
            y=inner[0, 0, 1].item(),
            width=inner.diff(dim=1)[0, 0, 0].item(),
            height=inner.diff(dim=0)[0, 0, 1].item(),
            fill="none",
            stroke="black",
            stroke_width=1.0,
        )
    )

    return svg.G(elements=elements)


if __name__ == "__main__":
    TOP = 1
    BOTTOM = HEIGHT - 4
    LEFT = 4
    RIGHT = WIDTH - 1
    grid_length = BOTTOM - TOP
    GRID_LEFT = RIGHT - grid_length
    GRAPH_RIGHT = GRID_LEFT - 10

    elements = []

    # Draw the contraction grid on the right.
    grid = draw_contraction_grid(grid_length)
    grid.transform = f"translate({GRID_LEFT}, 1)"
    elements.append(grid)

    # Just draw the graph here.
    x = np.linspace(0, 1, 100)
    y = np.exp(2 * ARM_RATIO * x)
    LINE_RIGHT = GRAPH_RIGHT - 10
    LINE_TOP = TOP + 10
    x = (LEFT + x * (LINE_RIGHT - LEFT)).tolist()
    y = (BOTTOM - y / y.max() * (BOTTOM - LINE_TOP)).tolist()
    elements.append(
        svg.Polyline(
            points=" ".join(f"{xx},{yy}" for xx, yy in zip(x, y)),
            fill="none",
            stroke=BLUE[0],
            stroke_width=1.0,
            stroke_linecap="round",
        )
    )
    elements.append(draw_arrow(LEFT, BOTTOM, GRAPH_RIGHT, BOTTOM, 3.0, 1.0))
    elements.append(draw_arrow(LEFT, BOTTOM, LEFT, TOP, 3.0, 1.0))
    elements.append(
        svg.Text(
            x=GRAPH_RIGHT - 6,
            y=BOTTOM - 3,
            text="Radial Index",
            font_size=9,
            fill="black",
            text_anchor="end",
        )
    )
    elements.append(
        svg.Text(
            x=0,
            y=0,
            text="Radial Distance",
            font_size=9,
            fill="black",
            text_anchor="end",
            transform=f"translate({LEFT + 10}, {TOP + 10}) rotate(-90)",
        )
    )

    canvas = svg.SVG(
        width=WIDTH,
        height=HEIGHT,
        elements=elements,
    )

    base = "generated_paper_components/figure_exponential"
    path = Path(f"{base}.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(canvas.as_str())

    os.system(f"rsvg-convert -f pdf -o {base}.pdf {base}.svg")
