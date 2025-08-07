from pathlib import Path

import svg
from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper",),
    ("beartype", "beartype"),
):
    from .inset_common import CUBE_EDGES, CUBE_VERTICES, DEFAULT_CAMERA, project


if __name__ == "__main__":

    def inner_fn(x, center=0.5):
        return x * center + (1 - center) / 2

    canvas = svg.SVG(width=130, height=152, elements=[])

    projected_cube = project(CUBE_VERTICES[CUBE_EDGES], DEFAULT_CAMERA)
    for index, ((ax, ay), (bx, by)) in enumerate(projected_cube[:3]):
        stroke_width = [2] * 3 + [2] * 9
        stroke_color = ["#e6e6e6"] * 3 + ["#000000"] * 9
        canvas.elements.append(
            svg.Line(
                x1=ax,
                y1=ay,
                x2=bx,
                y2=by,
                stroke=stroke_color[index],
                stroke_width=stroke_width[index],
                stroke_linecap="round",
            )
        )

    projected_cube = project(inner_fn(CUBE_VERTICES[CUBE_EDGES]), DEFAULT_CAMERA)
    for index, ((ax, ay), (bx, by)) in enumerate(projected_cube):
        stroke_width = [2] * 3 + [2] * 9
        stroke_color = ["#e6e6e6"] * 3 + ["#000000"] * 9
        canvas.elements.append(
            svg.Line(
                x1=ax,
                y1=ay,
                x2=bx,
                y2=by,
                stroke=stroke_color[index],
                stroke_width=stroke_width[index],
                stroke_linecap="round",
            )
        )

    inner = project(inner_fn(CUBE_VERTICES), DEFAULT_CAMERA)
    outer = project(CUBE_VERTICES, DEFAULT_CAMERA)
    for index, ((ax, ay), (bx, by)) in enumerate(zip(inner, outer)):
        canvas.elements.append(
            svg.Line(
                x1=ax,
                y1=ay,
                x2=bx,
                y2=by,
                stroke="black" if index != 2 else "#e6e6e6",
                stroke_width=1,
                stroke_linecap="round",
            )
        )

    projected_cube = project(CUBE_VERTICES[CUBE_EDGES], DEFAULT_CAMERA)
    for index, ((ax, ay), (bx, by)) in enumerate(projected_cube[3:]):
        stroke_width = [2] * 3 + [2] * 9
        stroke_color = ["#e6e6e6"] * 3 + ["#000000"] * 9
        canvas.elements.append(
            svg.Line(
                x1=ax,
                y1=ay,
                x2=bx,
                y2=by,
                stroke=stroke_color[index + 3],
                stroke_width=stroke_width[index + 3],
                stroke_linecap="round",
            )
        )

    path = Path("generated_paper_components/inset_tesseract.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(canvas.as_str())
