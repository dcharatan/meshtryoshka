import os
from pathlib import Path
from typing import Callable

import svg
from jaxtyping import install_import_hook
from PIL import Image

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper",),
    ("beartype", "beartype"),
):
    from .common import draw_arrow, flexible_crop

WIDTH = 510
MARGIN = 2


def draw_progression(
    tag: str,
    image_fn: Callable[[int], Path],
    n: int,
    width: float,
    aspect_ratio: float,
    fraction: float = 1.0,
    dx: float = 0.5,
    dy: float = 0.5,
    max_resolution: int | None = None,
    border: bool = False,
) -> tuple[svg.G, float, float]:
    temp_path = Path(f"generated_paper_components/figure_representative_images/{tag}")
    temp_path.mkdir(parents=True, exist_ok=True)

    image_width = (width - (n - 1) * MARGIN) / n
    height = image_width / aspect_ratio

    elements = []
    for i in range(n):
        modified_image_path = temp_path / f"{i}.png"
        path = image_fn(i)
        image = Image.open(path).convert("RGB")
        image = flexible_crop(image, aspect_ratio, fraction, dx, dy, max_resolution)
        image.save(modified_image_path)
        elements.append(
            svg.Image(
                x=i * (image_width + MARGIN),
                y=0,
                width=image_width,
                height=image_width / aspect_ratio,
                href=str(modified_image_path.absolute()),
            )
        )
        if border:
            elements.append(
                svg.Rect(
                    x=i * (image_width + MARGIN) + 0.5,
                    y=0.5,
                    width=image_width - 1,
                    height=image_width / aspect_ratio - 1,
                    fill="none",
                    stroke="black",
                    stroke_width=0.5,
                )
            )

    return svg.G(elements=elements), height, image_width


def get_flower_image(index: int) -> Path:
    return {
        0: Path(
            "/data/scene-rep/u/charatan/jobs/triangle_splatting/special/val_render_0_garden/surface_1.5/DSC08084.JPG",
        ),
        1: Path(
            "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_43_59.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_garden/workspace/tensorboard/val_render_1599/surface_2.0/DSC08084.JPG",
        ),
        2: Path(
            "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_43_59.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_garden/workspace/tensorboard/val_render_3198/surface_2.0/DSC08084.JPG",
        ),
        3: Path(
            "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_43_59.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_garden/workspace/tensorboard/val_render_23985/surface_2.0/DSC08084.JPG",
        ),
    }[index]


def get_mip360_image(index: int) -> Path:
    return {
        0: Path(
            "/data/scene-rep/u/charatan/jobs/triangle_splatting/special/val_render_0/layered_1.5/_DSC9269.JPG"
        ),
        1: Path(
            "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_44_31.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_stump/workspace/tensorboard/val_render_1599/surface_2.0/_DSC9269.JPG"
        ),
        2: Path(
            "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_44_31.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_stump/workspace/tensorboard/val_render_3198/surface_2.0/_DSC9269.JPG"
        ),
        3: Path(
            "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_44_31.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_stump/workspace/tensorboard/val_render_22386/surface_2.0/_DSC9269.JPG"
        ),
    }[index]


if __name__ == "__main__":
    MESH_WIDTH = 100
    LEFT = MARGIN
    RIGHT = WIDTH - MESH_WIDTH - MARGIN - 1
    ASPECT = 0.6313
    TOP = MARGIN
    flowers, flower_height, image_width = draw_progression(
        "flowers",
        get_mip360_image,  # switcheroo
        n=4,
        width=RIGHT - LEFT,
        aspect_ratio=ASPECT,
        fraction=0.8,
        dx=0.5,
        dy=0.3,
    )
    flowers.transform = f"translate({LEFT}, {TOP})"
    FLOWER_BOTTOM = TOP + flower_height
    ARROW = FLOWER_BOTTOM + 10
    MIP_TOP = ARROW + 10
    mip360, mip360_height, _ = draw_progression(
        "mip360",
        get_flower_image,  # switcheroo
        n=4,
        width=RIGHT - LEFT,
        aspect_ratio=ASPECT,
        fraction=0.9,
        dx=0.5,
        dy=1.0,
    )
    mip360.transform = f"translate({LEFT}, {MIP_TOP})"
    HEIGHT = MIP_TOP + mip360_height + MARGIN

    # Build the callouts.
    image = Image.open("/data/scene-rep/u/chandratreya/vis_stump.png").convert("RGB")
    image = flexible_crop(image, MESH_WIDTH / flower_height, 0.4, 0.5, 0.8, 512)
    top = Path("generated_paper_components/figure_representative_images/top_thing.png")
    image.save(top)
    flowers_callout = svg.G(
        elements=[
            svg.Image(
                x=0,
                y=0,
                width=image_width,
                height=flower_height,
                href=str(top.absolute()),
            ),
            svg.Rect(
                x=0,
                y=0,
                width=MESH_WIDTH,
                height=flower_height,
                fill="none",
            ),
        ]
    )
    flowers_callout.transform = f"translate({RIGHT + MARGIN}, {TOP})"
    image = Image.open(
        "/data/scene-rep/u/chandratreya/vis_render_brighter.png"
    ).convert("RGB")
    image = flexible_crop(image, MESH_WIDTH / flower_height, 0.4, 0.5, 0.33, 512)
    bottom = Path(
        "generated_paper_components/figure_representative_images/bottom_thing.png"
    )
    image.save(bottom)
    mip360_callout = svg.G(
        elements=[
            svg.Image(
                x=0,
                y=0,
                width=image_width,
                height=flower_height,
                href=str(bottom.absolute()),
            ),
            svg.Rect(
                x=0,
                y=0,
                width=MESH_WIDTH,
                height=mip360_height,
                fill="none",
            ),
        ]
    )
    mip360_callout.transform = f"translate({RIGHT + MARGIN}, {MIP_TOP})"

    # Build combined canvas
    canvas = svg.SVG(
        width=WIDTH,
        height=HEIGHT,
        elements=[
            svg.Rect(
                x=0,
                y=0,
                width=WIDTH,
                height=HEIGHT,
                fill="white",
            ),
            flowers,
            mip360,
            draw_arrow(
                LEFT + image_width + MARGIN / 2 - 20,
                ARROW,
                LEFT + image_width + MARGIN / 2 + 20,
                ARROW,
                radius=3,
                stroke_width=1,
            ),
            draw_arrow(
                RIGHT - image_width - MARGIN / 2 - 20,
                ARROW,
                RIGHT - image_width - MARGIN / 2 + 20,
                ARROW,
                radius=3,
                stroke_width=1,
            ),
            mip360_callout,
            flowers_callout,
            svg.Text(
                x=LEFT + image_width / 2,
                y=ARROW + 3,
                text="Initialization",
                font_size=9,
                text_anchor="middle",
                fill="black",
            ),
            svg.Text(
                x=RIGHT - image_width / 2,
                y=ARROW + 3,
                text="Final Model",
                font_size=9,
                text_anchor="middle",
                fill="black",
            ),
            svg.Text(
                x=LEFT + (RIGHT - LEFT) / 2,
                y=ARROW + 3,
                text="Mesh-Based Differentiable Rendering",
                font_size=9,
                text_anchor="middle",
                font_weight="bold",
                fill="black",
            ),
        ],
    )
    print(WIDTH)
    print(HEIGHT)
    print(WIDTH / HEIGHT)

    # Write out the final SVG and convert it to PDF.
    base = Path("generated_paper_components/figure_representative")
    base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = base.with_suffix(".svg")
    with svg_path.open("w") as f:
        f.write(canvas.as_str())
    os.system(f"rsvg-convert -f pdf -o {base.with_suffix('.pdf')} {svg_path}")
    os.system(f"pdftoppm {base}.pdf {base}.jpg -jpeg -rx 600 -ry 600")
