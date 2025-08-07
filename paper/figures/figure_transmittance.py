import os
from pathlib import Path

import numpy as np
import svg
from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper", "meshtryoshka"),
    ("beartype", "beartype"),
):
    from .common import BLUE, draw_arrow

WIDTH = 240 + 4  # shift figure slightly left in latex to make line match up
HEIGHT = 120


if __name__ == "__main__":
    TOP = 1
    LEFT = 1 + 14
    RIGHT = WIDTH - 1
    BOTTOM = HEIGHT - 14

    def compute_transmittance(d):
        return 1 / (1 + np.exp(-d))

    def inverse(t):
        return np.log(t / (1 - t))

    elements = []

    # Just draw the graph here.
    X_MIN = -6
    X_MAX = 6
    LINE_TOP = TOP + 10
    LINE_RIGHT = RIGHT - 10

    for i, level_set in enumerate([0.9, 0.7, 0.5, 0.3, 0.1, 0.03]):
        y = BOTTOM - level_set * (BOTTOM - LINE_TOP)
        d = inverse(level_set)
        x = LEFT + (d - X_MIN) / (X_MAX - X_MIN) * (LINE_RIGHT - LEFT)
        elements.append(
            svg.Line(
                x1=LEFT,
                y1=y,
                x2=x,
                y2=y,
                stroke="#dddddd",
                stroke_width=0.25,
                stroke_linecap="round",
            )
        )
        elements.append(
            svg.Line(
                x1=x,
                y1=y,
                x2=x,
                y2=BOTTOM,
                stroke="#dddddd",
                stroke_width=0.25,
                stroke_linecap="round",
            )
        )
        elements.append(
            svg.Text(
                x=x,
                y=BOTTOM + 10,
                text=f"d_{i}",
                font_size=9,
                fill="black",
                text_anchor="middle",
            )
        )
        elements.append(
            svg.Text(
                x=0,
                y=0,
                text=f"T_{i}",
                font_size=9,
                fill="black",
                text_anchor="middle",
                transform=f"translate({LEFT - 3}, {y}) rotate(-90)",
            )
        )

    x = np.linspace(X_MIN, X_MAX, 100)
    y = compute_transmittance(x)
    x = (LEFT + (x - X_MIN) / (X_MAX - X_MIN) * (LINE_RIGHT - LEFT)).tolist()
    y = (BOTTOM - y * (BOTTOM - LINE_TOP)).tolist()
    elements.append(
        svg.Polyline(
            points=" ".join(f"{xx},{yy}" for xx, yy in zip(x, y)),
            fill="none",
            stroke=BLUE[0],
            stroke_width=1.0,
            stroke_linecap="round",
        )
    )
    elements.append(draw_arrow(LEFT, BOTTOM, RIGHT, BOTTOM, 3.0, 1.0))
    elements.append(draw_arrow(LEFT, BOTTOM, LEFT, TOP, 3.0, 1.0))
    elements.append(
        svg.Text(
            x=RIGHT - 6,
            y=BOTTOM - 3,
            text="\\text{Signed Distance}",
            font_size=9,
            fill="black",
            text_anchor="end",
        )
    )
    elements.append(
        svg.Text(
            x=0,
            y=0,
            text="\\text{Transmittance}",
            font_size=9,
            fill="black",
            text_anchor="end",
            transform=f"translate({LEFT + 9}, {TOP + 6}) rotate(-90)",
        )
    )

    canvas = svg.SVG(
        width=WIDTH,
        height=HEIGHT,
        elements=elements,
    )

    base = "generated_paper_components/figure_transmittance"
    path = Path(f"{base}.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(canvas.as_str())
    os.system(
        f"svg2tikz {base}.svg "
        f"--output {base}_tikz.tex -t math --figonly --output-unit px"
    )
