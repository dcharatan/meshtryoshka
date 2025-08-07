import os
import random
from collections import defaultdict
from functools import cache
from itertools import combinations
from pathlib import Path

import numpy as np
import svg
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float32, install_import_hook
from torch import Tensor

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper",),
    ("beartype", "beartype"),
):
    from .common import BLUE, draw_arrow, encode_image


WIDTH = 510
HEIGHT = 110
BOX_WIDTH = 145
CIRCLE_RADIUS = 35


def rgba(r, g, b, a):
    return (r, g, b)


COLORS = [
    rgba(222, 135, 239, 1),
    rgba(171, 65, 196, 1),
    rgba(69, 101, 227, 1),
    rgba(79, 160, 236, 1),
    rgba(239, 173, 89, 1),
    rgba(222, 108, 44, 1),
    rgba(29, 145, 107, 1),
    rgba(82, 175, 101, 1),
    rgba(245, 122, 121, 1),
    rgba(221, 56, 54, 1),
]


def get_triangles():
    # This is exported from Figma.
    path = "M50.5 104.5L34 68M50.5 104.5L88 100M50.5 104.5L76.5 141.5M50.5 104.5L34 132.5M50.5 104.5L1 88.5M34 68L88 100M34 68L67 40M34 68L1 88.5M34 68L13.5 34.5M88 100L67 40M88 100L143 101.5M88 100L76.5 141.5M88 100L101.5 62.5M67 40L125 16.5M67 40L13.5 34.5M67 40L101.5 62.5M67 40L70 1M143 101.5L125 16.5M143 101.5L76.5 141.5M143 101.5L101.5 62.5M125 16.5L101.5 62.5M125 16.5L70 1M76.5 141.5L34 132.5M34 132.5L1 88.5M1 88.5L13.5 34.5M13.5 34.5L70 1"  # noqa: E501

    # Parse the SVG very non-robustly.
    points = {}
    lines = []
    for start, end in [x.split("L") for x in path.split("M") if x != ""]:
        if start not in points:
            points[start] = len(points)
        if end not in points:
            points[end] = len(points)
        lines.append((points[start], points[end]))

    lines = np.array(lines, dtype=np.int32)

    points = [(v, k) for k, v in points.items()]
    points = [point.split(" ") for _, point in sorted(points)]
    points = [(float(x), float(y)) for x, y in points]
    points = np.array(list(points), dtype=np.float32)

    # Find all of the triangles (very messy).
    line_lookup = defaultdict(list)
    for a, b in lines.tolist():
        if a < b:
            line_lookup[a].append(b)
        else:
            line_lookup[b].append(a)
    line_lookup = dict(line_lookup)
    triangles = set()
    for a, others in line_lookup.items():
        for b, c in combinations(others, 2):
            if b in line_lookup.get(c, []) or c in line_lookup.get(b, []):
                triangles.add(tuple(sorted((a, b, c))))

    triangles = np.array(list(triangles), dtype=np.int32)

    # Add the offset (needs to be done manually based on the exported SVG).
    points = (points - np.array((18, 23))) / 100

    return points, lines, triangles


def get_parameter_values():
    points, _, _ = get_triangles()
    g = np.random.Generator(np.random.PCG64(0))
    sdf = g.normal(size=points.shape[0]) * 0.1 + 0.5

    random.seed(12345)
    color = random.choices(BLUE, k=points.shape[0])

    return sdf, color


def make_vertex_sdf_circle(x: float, y: float, id: str):
    points, lines, _ = get_triangles()
    points *= CIRCLE_RADIUS * 2

    sdf, _ = get_parameter_values()

    g_lines = svg.G(
        elements=[
            svg.Line(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke="black",
                stroke_width=0.5,
                stroke_linecap="round",
            )
            for (x1, y1), (x2, y2) in points[lines]
        ],
        mask=f"url(#mask_{id})",
    )

    g_points = svg.G(
        elements=[
            svg.Circle(
                cx=x,
                cy=y,
                r=3,
                fill=f"rgb({int(255 * sdf[i])}, {int(255 * sdf[i])}, {int(255 * sdf[i])})",  # noqa: E501
                stroke="black",
                stroke_width=0.5,
            )
            for i, (x, y) in enumerate(points)
        ],
        mask=f"url(#mask_{id})",
    )

    mask = svg.Mask(
        id=f"mask_{id}",
        elements=[
            svg.Circle(
                cx=CIRCLE_RADIUS,
                cy=CIRCLE_RADIUS,
                r=CIRCLE_RADIUS,
                fill="white",
            )
        ],
    )

    circle = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="none",
        stroke="black",
        stroke_width=1,
    )

    bg = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="#f5f5f5",
    )

    group = svg.G(
        elements=[bg, mask, g_lines, g_points, circle],
        transform=f"translate({x}, {y})",
    )

    return group


def make_vertex_color_circle(x: float, y: float, id: str):
    points, lines, _ = get_triangles()
    points *= CIRCLE_RADIUS * 2

    _, color = get_parameter_values()

    g_lines = svg.G(
        elements=[
            svg.Line(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke="black",
                stroke_width=0.5,
                stroke_linecap="round",
            )
            for (x1, y1), (x2, y2) in points[lines]
        ],
        mask=f"url(#mask_{id})",
    )

    g_points = svg.G(
        elements=[
            svg.Circle(
                cx=x,
                cy=y,
                r=3,
                fill=color[i],
                stroke="black",
                stroke_width=0.5,
            )
            for i, (x, y) in enumerate(points)
        ],
        mask=f"url(#mask_{id})",
    )

    mask = svg.Mask(
        id=f"mask_{id}",
        elements=[
            svg.Circle(
                cx=CIRCLE_RADIUS,
                cy=CIRCLE_RADIUS,
                r=CIRCLE_RADIUS,
                fill="white",
            )
        ],
    )

    circle = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="none",
        stroke="black",
        stroke_width=1,
    )

    bg = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="#f5f5f5",
    )

    group = svg.G(
        elements=[bg, mask, g_lines, g_points, circle],
        transform=f"translate({x}, {y})",
    )

    return group


def make_triangle_id_circle(x: float, y: float, id: str):
    points, lines, triangles = get_triangles()
    points *= CIRCLE_RADIUS * 2

    g_lines = svg.G(
        elements=[
            svg.Line(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke="black",
                stroke_width=0.5,
                stroke_linecap="round",
            )
            for (x1, y1), (x2, y2) in points[lines]
        ],
        mask=f"url(#mask_{id})",
    )

    centroids = points[triangles].mean(axis=1)

    ids = [
        svg.Text(
            x=xy[0],
            y=xy[1] + 4,
            text=str(i),
            font_size=9,
            text_anchor="middle",
            fill="black",
            mask=f"url(#mask_{id})",
        )
        for i, xy in enumerate(centroids)
    ]

    mask = svg.Mask(
        id=f"mask_{id}",
        elements=[
            svg.Circle(
                cx=CIRCLE_RADIUS,
                cy=CIRCLE_RADIUS,
                r=CIRCLE_RADIUS,
                fill="white",
            )
        ],
    )

    circle = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="none",
        stroke="black",
        stroke_width=1,
    )

    bg = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="#f5f5f5",
    )

    group = svg.G(
        elements=[bg, mask, g_lines, *ids, circle],
        transform=f"translate({x}, {y})",
    )

    return group


def compute_barycentric_coordinates(
    points: Float32[Tensor, "*#batch xy=2"],
    triangles: Float32[Tensor, "*#batch corner=3 xy=2"],
) -> Float32[Tensor, "*batch uvw=3"]:
    a, b, c = triangles.unbind(dim=-2)

    v0_x, v0_y = (b - a).unbind(dim=-1)
    v1_x, v1_y = (c - a).unbind(dim=-1)
    v2_x, v2_y = (points - a).unbind(dim=-1)
    coefficient = 1 / (v0_x * v1_y - v1_x * v0_y)
    v = (v2_x * v1_y - v1_x * v2_y) * coefficient
    w = (v0_x * v2_y - v2_x * v0_y) * coefficient
    u = 1 - v - w
    return torch.stack((u, v, w), dim=-1)


@cache
def get_bary():
    points, _, triangles = get_triangles()
    RESOLUTION = 400
    cc = (
        torch.arange(RESOLUTION, dtype=torch.float32, device="cuda") + 0.5
    ) / RESOLUTION
    xy = torch.meshgrid(cc, cc, indexing="ij")
    xy = torch.stack(xy, dim=-1)
    tri = torch.tensor(points[triangles], device="cuda", dtype=torch.float32)
    bary = compute_barycentric_coordinates(xy[:, :, None], tri)
    valid = ((bary <= 1) & (bary >= 0)).all(dim=-1)
    tri_index = valid.int().argmax(dim=-1)
    arange = torch.arange(RESOLUTION, device="cuda", dtype=torch.int32)
    bary = (
        bary[
            repeat(arange, "i -> i j", j=RESOLUTION),
            repeat(arange, "i -> j i", j=RESOLUTION),
            tri_index,
        ]
        * valid.any(dim=-1)[..., None]
    )
    return tri_index, bary


def make_barycentric_circle(x: float, y: float, id: str):
    points, lines, _ = get_triangles()

    _, bary = get_bary()
    bary = 1 - bary

    bg = svg.Image(
        width=2 * CIRCLE_RADIUS,
        height=2 * CIRCLE_RADIUS,
        href=encode_image(rearrange(bary, "h w c -> c w h"), "jpeg"),
        x=0,
        y=0,
        mask=f"url(#mask_{id})",
    )

    points *= CIRCLE_RADIUS * 2
    g_lines = svg.G(
        elements=[
            svg.Line(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke="black",
                stroke_width=0.5,
                stroke_linecap="round",
            )
            for (x1, y1), (x2, y2) in points[lines]
        ],
        mask=f"url(#mask_{id})",
    )

    mask = svg.Mask(
        id=f"mask_{id}",
        elements=[
            svg.Circle(
                cx=CIRCLE_RADIUS,
                cy=CIRCLE_RADIUS,
                r=CIRCLE_RADIUS,
                fill="white",
            )
        ],
    )

    circle = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="none",
        stroke="black",
        stroke_width=1,
    )

    group = svg.G(
        elements=[mask, bg, g_lines, circle],
        transform=f"translate({x}, {y})",
    )

    return group


def make_sdf_circle(x: float, y: float, id: str):
    points, lines, triangles = get_triangles()

    sdf, _ = get_parameter_values()
    tri_idx, bary = get_bary()

    corner_sdf = sdf[triangles][tri_idx.cpu().numpy()]

    vis = einsum(bary.cpu().numpy(), corner_sdf, "h w uvw, h w uvw -> h w")
    vis = torch.tensor(vis)

    bg = svg.Image(
        width=2 * CIRCLE_RADIUS,
        height=2 * CIRCLE_RADIUS,
        href=encode_image(repeat(vis, "h w -> c w h", c=3), "jpeg"),
        x=0,
        y=0,
        mask=f"url(#mask_{id})",
    )

    points *= CIRCLE_RADIUS * 2
    g_lines = svg.G(
        elements=[
            svg.Line(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke="black",
                stroke_width=0.5,
                stroke_linecap="round",
            )
            for (x1, y1), (x2, y2) in points[lines]
        ],
        mask=f"url(#mask_{id})",
    )

    mask = svg.Mask(
        id=f"mask_{id}",
        elements=[
            svg.Circle(
                cx=CIRCLE_RADIUS,
                cy=CIRCLE_RADIUS,
                r=CIRCLE_RADIUS,
                fill="white",
            )
        ],
    )

    circle = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="none",
        stroke="black",
        stroke_width=1,
    )

    group = svg.G(
        elements=[mask, bg, g_lines, circle],
        transform=f"translate({x}, {y})",
    )

    return group


def make_color_circle(x: float, y: float, id: str):
    points, lines, triangles = get_triangles()

    _, color = get_parameter_values()
    tri_idx, bary = get_bary()

    def hex_to_rgb_float(h):
        h = h.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    color = np.array([hex_to_rgb_float(c) for c in color], dtype=np.float32)

    corner_color = color[triangles][tri_idx.cpu().numpy()]

    vis = einsum(bary.cpu().numpy(), corner_color, "h w uvw, h w uvw rgb -> rgb h w")
    vis = torch.tensor(vis)

    bg = svg.Image(
        width=2 * CIRCLE_RADIUS,
        height=2 * CIRCLE_RADIUS,
        href=encode_image(rearrange(vis, "c h w -> c w h"), "jpeg"),
        x=0,
        y=0,
        mask=f"url(#mask_{id})",
    )

    points *= CIRCLE_RADIUS * 2
    g_lines = svg.G(
        elements=[
            svg.Line(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke="black",
                stroke_width=0.5,
                stroke_linecap="round",
            )
            for (x1, y1), (x2, y2) in points[lines]
        ],
        mask=f"url(#mask_{id})",
    )

    mask = svg.Mask(
        id=f"mask_{id}",
        elements=[
            svg.Circle(
                cx=CIRCLE_RADIUS,
                cy=CIRCLE_RADIUS,
                r=CIRCLE_RADIUS,
                fill="white",
            )
        ],
    )

    circle = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="none",
        stroke="black",
        stroke_width=1,
    )

    group = svg.G(
        elements=[mask, bg, g_lines, circle],
        transform=f"translate({x}, {y})",
    )

    return group


def make_circle_base(x: float, y: float, id: str):
    points, lines, triangles = get_triangles()
    points *= CIRCLE_RADIUS * 2

    from random import choice

    g_triangles = g_lines = svg.G(
        elements=[
            svg.Polygon(
                points=f"{x1},{y1} {x2},{y2} {x3},{y3}",
                fill=choice(
                    ["gray", "blue", "red", "green", "yellow", "pink", "purple"]
                ),
            )
            for (x1, y1), (x2, y2), (x3, y3) in points[triangles]
        ],
        mask=f"url(#mask_{id})",
    )

    g_lines = svg.G(
        elements=[
            svg.Line(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke="black",
                stroke_width=0.5,
                stroke_linecap="round",
            )
            for (x1, y1), (x2, y2) in points[lines]
        ],
        mask=f"url(#mask_{id})",
    )

    g_points = svg.G(
        elements=[
            svg.Circle(
                cx=x,
                cy=y,
                r=3,
                fill="red",
            )
            for x, y in points
        ],
        mask=f"url(#mask_{id})",
    )

    mask = svg.Mask(
        id=f"mask_{id}",
        elements=[
            svg.Circle(
                cx=CIRCLE_RADIUS,
                cy=CIRCLE_RADIUS,
                r=CIRCLE_RADIUS,
                fill="white",
            )
        ],
    )

    circle = svg.Circle(
        cx=CIRCLE_RADIUS,
        cy=CIRCLE_RADIUS,
        r=CIRCLE_RADIUS,
        fill="none",
        stroke="black",
        stroke_width=1,
    )

    group = svg.G(
        elements=[mask, g_triangles, g_lines, g_points, circle],
        transform=f"translate({x}, {y})",
    )

    return group


def create_curly_bracket(x, y, width, height, stroke_width=1, stroke_color="black"):
    # Calculate dimensions
    r = height / 2
    horizontal_segment = (width - 4 * r) / 2

    # Start path at top left
    path = f"M 0,{height}"

    # First quarter circle (left)
    path += f" A {r},{r} 0 0 1 {r},{r}"

    # First horizontal line
    path += f" H {r + horizontal_segment}"

    # Two quarter circles in the middle (down then up)
    path += f" A {r},{r} 0 0 0 {r * 2 + horizontal_segment},{0}"
    path += f" A {r},{r} 0 0 0 {r * 3 + horizontal_segment},{r}"

    # Second horizontal line
    path += f" H {width - r}"

    # Last quarter circle (right)
    path += f" A {r},{r} 0 0 1 {width},{height}"

    # Create the SVG
    return svg.Path(
        d=path,
        stroke=stroke_color,
        stroke_width=stroke_width,
        fill="none",
        transform=f"translate({x}, {y})",
        stroke_linecap="round",
        stroke_linejoin="round",
    )


def make_box(
    x: float,
    label: str,
    id: str,
    left_fn,
    right_fn,
    left_label: str,
    right_label: str,
):
    BRACKET_WIDTH = BOX_WIDTH - 20
    margin = BOX_WIDTH - 4 * CIRCLE_RADIUS
    CIRCLE_Y = 25
    group = svg.G(
        elements=[
            left_fn(0, CIRCLE_Y, f"{id}_left"),
            right_fn(margin + 2 * CIRCLE_RADIUS, CIRCLE_Y, f"{id}_right"),
            svg.Text(
                x=BOX_WIDTH / 2,
                y=10,
                text=label,
                font_size=9,
                text_anchor="middle",
                fill="black",
            ),
            create_curly_bracket(
                0.5 * (BOX_WIDTH - BRACKET_WIDTH), 16.5, BRACKET_WIDTH, 5, 0.5
            ),
            # labels
            svg.Text(
                x=CIRCLE_RADIUS,
                y=CIRCLE_Y + 2 * CIRCLE_RADIUS + 12,
                text=left_label,
                font_size=9,
                text_anchor="middle",
                fill="black",
            ),
            svg.Text(
                x=BOX_WIDTH - CIRCLE_RADIUS,
                y=CIRCLE_Y + 2 * CIRCLE_RADIUS + 12,
                text=right_label,
                font_size=9,
                text_anchor="middle",
                fill="black",
            ),
        ],
        transform=f"translate({x}, 0)",
    )
    return group


def create_plus(
    x,
    y,
    radius,
    stroke_width=2,
    stroke_color="black",
):
    return svg.G(
        elements=[
            svg.Line(
                x1=x - radius,
                y1=y,
                x2=x + radius,
                y2=y,
                stroke=stroke_color,
                stroke_width=stroke_width,
                stroke_linecap="round",
            ),
            svg.Line(
                x1=x,
                y1=y - radius,
                x2=x,
                y2=y + radius,
                stroke=stroke_color,
                stroke_width=stroke_width,
                stroke_linecap="round",
            ),
        ],
    )


if __name__ == "__main__":
    canvas = svg.SVG(
        width=WIDTH,
        height=HEIGHT,
        elements=[
            make_box(
                1,
                "Vertex Attributes (Differentiable)",
                "differentiable_inputs",
                make_vertex_sdf_circle,
                make_vertex_color_circle,
                "Signed Distance",
                "Color",
            ),
            make_box(
                1 + BOX_WIDTH + 30,
                "Rendered (Non-Differentiable)",
                "rendered",
                make_triangle_id_circle,
                make_barycentric_circle,
                "Triangle ID",
                "Barycentric Coords.",
            ),
            make_box(
                WIDTH - BOX_WIDTH - 1,
                "Interpolated (Differentiable)",
                "interpolated",
                make_sdf_circle,
                make_color_circle,
                "Signed Distance",
                "Color",
            ),
            create_plus(1 + BOX_WIDTH + 30 / 2, 25 + CIRCLE_RADIUS, 6, 1),
            draw_arrow(
                1 + 2 * BOX_WIDTH + 30 + 9,
                25 + CIRCLE_RADIUS,
                (WIDTH - BOX_WIDTH - 1) - 9,
                25 + CIRCLE_RADIUS,
                3,
                stroke_width=1,
            ),
        ],
    )

    base = "generated_paper_components/figure_rendering"
    path = Path(f"{base}.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(canvas.as_str())

    os.system(f"rsvg-convert -f pdf -o {base}.pdf {base}.svg")
