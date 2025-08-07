import os
from pathlib import Path

import svg
from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper", "meshtryoshka"),
    ("beartype", "beartype"),
):
    from .common import BLUE


WIDTH = 510
HEIGHT = 300


def draw_frustum(
    x: float,
    y: float,
    top_width: float,
    bottom_width: float,
    n: int,
) -> svg.G:
    elements = []
    height = (top_width - bottom_width) / 2

    # Add the center line.
    elements.append(
        svg.Line(
            x1=top_width / 2,
            x2=top_width / 2,
            y1=0,
            y2=height,
            stroke="black",
            stroke_width=0.25,
            stroke_linecap="round",
        )
    )

    # Add the other lines.
    bottom_left = (top_width - bottom_width) / 2
    for i in range(n - 1):
        t = (i + 1) / n

        # Left line.
        elements.append(
            svg.Line(
                x1=top_width / 2 * t,
                x2=bottom_left + bottom_width / 2 * t,
                y1=0,
                y2=height,
                stroke="black",
                stroke_width=0.25,
                stroke_linecap="round",
            )
        )

        # Right line.
        elements.append(
            svg.Line(
                x1=top_width / 2 + top_width / 2 * t,
                x2=top_width / 2 + bottom_width / 2 * t,
                y1=0,
                y2=height,
                stroke="black",
                stroke_width=0.25,
                stroke_linecap="round",
            )
        )

    scan = 0
    current_width = bottom_width / (2 * n)
    multiplier = 1 + 1 / n

    while scan < height:
        next_scan = scan + current_width

        t = next_scan / height
        line_y = (1 - t) * height
        if t < 1:
            line_offset = (1 - t) * (top_width - bottom_width) / 2
            elements.append(
                svg.Line(
                    x1=line_offset,
                    x2=top_width - line_offset,
                    y1=line_y,
                    y2=line_y,
                    stroke="black",
                    stroke_width=0.25,
                    stroke_linecap="round",
                )
            )
            elements.append(
                svg.Rect(
                    x=top_width / 2,
                    y=line_y,
                    width=current_width,
                    height=current_width,
                    stroke="black",
                    stroke_width=0.5,
                    stroke_linejoin="round",
                    fill=BLUE[0],
                )
            )
        else:
            yy = current_width + line_y
            elements.append(
                svg.Rect(
                    x=top_width / 2,
                    y=0,
                    width=current_width,
                    height=yy,
                    fill=BLUE[0],
                )
            )
            xy = (
                (top_width / 2, 0),
                (top_width / 2, yy),
                (top_width / 2 + current_width, yy),
                (top_width / 2 + current_width, 0),
            )
            elements.append(
                svg.Polyline(
                    points=" ".join(f"{xx},{yy}" for xx, yy in xy),
                    fill="none",
                    stroke="black",
                    stroke_width=0.5,
                    stroke_linecap="round",
                )
            )

        # Step to the next row.
        current_width *= multiplier
        scan = next_scan

    # Add the outlines.
    xy = (
        (0, 0),
        ((top_width - bottom_width) / 2, height),
        (top_width - (top_width - bottom_width) / 2, height),
        (top_width, 0),
    )
    elements.append(
        svg.Polyline(
            points=" ".join(f"{xx},{yy}" for xx, yy in xy),
            fill="none",
            stroke="black",
            stroke_width=1.0,
            stroke_linecap="round",
        )
    )

    return svg.G(elements=elements, transform=f"translate({x}, {y})")


if __name__ == "__main__":
    NUM_ROWS = 1
    NUM_COLS = 3
    MARGIN = 10
    MAT = 5
    SHORT_SIDE = 20
    NS = (1, 2, 4)
    FRUSTUM_WIDTH = (WIDTH - 2 * MAT - (NUM_COLS - 1) * MARGIN) / NUM_COLS
    FRUSTUM_HEIGHT = (FRUSTUM_WIDTH - SHORT_SIDE) / 2
    HEIGHT = 2 * MAT + NUM_ROWS * FRUSTUM_HEIGHT + (NUM_ROWS - 1) * MARGIN

    elements = []

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            index = row * NUM_COLS + col
            frustum = draw_frustum(
                MAT + col * (FRUSTUM_WIDTH + MARGIN),
                MAT + row * (FRUSTUM_HEIGHT + MARGIN),
                FRUSTUM_WIDTH,
                SHORT_SIDE,
                NS[index],
            )
            elements.append(frustum)

    canvas = svg.SVG(
        width=WIDTH,
        height=HEIGHT,
        elements=elements,
    )

    base = "generated_paper_components/figure_exponential_math"
    path = Path(f"{base}.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(canvas.as_str())

    os.system(f"rsvg-convert -f pdf -o {base}.pdf {base}.svg")
