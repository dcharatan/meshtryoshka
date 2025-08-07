from pathlib import Path

import svg
from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper",),
    ("beartype", "beartype"),
):
    from .inset_common import AXES, CUBE_EDGES, CUBE_VERTICES, DEFAULT_CAMERA, project


if __name__ == "__main__":

    def inner_fn(x, center=0.5):
        return x * center + (1 - center) / 2

    canvas = svg.SVG(width=130, height=152, elements=[])

    projected_cube = project(CUBE_VERTICES[CUBE_EDGES], DEFAULT_CAMERA)
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

    projected_axes = project(CUBE_VERTICES[AXES] * 0.4 + 0.08, DEFAULT_CAMERA)
    axes = []
    for axis, ((ax, ay), (bx, by)) in enumerate(projected_axes):
        axes.append(
            svg.Line(
                x1=ax,
                y1=ay,
                x2=bx,
                y2=by,
                stroke=["#DD3836", "#52AF65", "#4565E3"][axis],
                stroke_width=4,
                stroke_linecap="round",
            )
        )
    x, y, z = axes
    canvas.elements.extend([y, x, z])

    for index, (x, y) in enumerate(project(CUBE_VERTICES, DEFAULT_CAMERA)):
        color_indices = [1, 1, 1, 0, 1, 0, 0, 0]
        colors = ["#4565E3", "#DD3836"]
        canvas.elements.append(
            svg.Circle(
                cx=x,
                cy=y,
                r=6,
                fill=colors[color_indices[index]],
                stroke="black",
                stroke_width=1,
            )
        )

    path = Path("generated_paper_components/inset_voxel_representation.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(canvas.as_str())
