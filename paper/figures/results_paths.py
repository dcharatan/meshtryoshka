import json
from pathlib import Path

# fmt: off
SYNTHETIC_RESULTS = {
    "Ground Truth": [
        Path("/data/scene-rep/u/danielxu/triangle/volsurfs/runs/nerf/base/lego/2025-05-11-180029/0199999/renders/test/volumetric/gt/071.png"),
        Path("/data/scene-rep/u/danielxu/triangle/volsurfs/runs/nerf/base/chair/2025-05-12-001537/0199999/renders/test/volumetric/gt/041.png"),
        Path("/data/scene-rep/u/danielxu/triangle/volsurfs/runs/nerf/base/ship/2025-05-12-102806/0199999/renders/test/volumetric/gt/047.png"),
    ],
    "Meshtryoshka (Ours)": [
        Path("/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-21/lego_4_shells/2025_05_21.19_12_55.python3_-m_triangle_splatting__experiment__nerf_synthetic__dataset.scene_lego/workspace/tensorboard/test_render_7182/surface/r_71.png"),
        Path("/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-21/lego_4_shells/2025_05_21.19_12_00.python3_-m_triangle_splatting__experiment__nerf_synthetic__dataset.scene_chair/workspace/tensorboard/test_render_7182/surface/r_41.png"),
        Path("/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-21/lego_4_shells/2025_05_21.19_13_35.python3_-m_triangle_splatting__experiment__nerf_synthetic__dataset.scene_ship/workspace/tensorboard/test_render_7182/surface/r_47.png"),
    ],
    "nvdiffrec (DMTet)": [
        Path("/data/scene-rep/u/danielxu/triangle/nvdiffrec/out_nerf/nerf_lego/validate/val_000071_opt.png"),
        Path("/data/scene-rep/u/danielxu/triangle/nvdiffrec/out_nerf/nerf_chair/validate/val_000041_opt.png"),
        Path("/data/scene-rep/u/danielxu/triangle/nvdiffrec/out_nerf/nerf_ship/validate/val_000047_opt.png"),
    ],
    "nvdiffrec (Flexi.)": [
        Path("/data/scene-rep/u/danielxu/triangle/nvdiffrec/out_flexi/nerf_lego/validate/val_000071_opt.png"),
        Path("/data/scene-rep/u/danielxu/triangle/nvdiffrec/out_flexi/nerf_chair/validate/val_000041_opt.png"),
        Path("/data/scene-rep/u/danielxu/triangle/nvdiffrec/out_flexi/nerf_ship/validate/val_000047_opt.png"),
    ],
    "NeuS2": [
        Path("/data/scene-rep/u/danielxu/triangle/NeuS2/output_base_config_renders/lego/test_screenshots/r_71.png"),
        Path("/data/scene-rep/u/danielxu/triangle/NeuS2/output_base_config_renders/chair/test_screenshots/r_41.png"),
        Path("/data/scene-rep/u/danielxu/triangle/NeuS2/output_base_config_renders/ship/test_screenshots/r_47.png"),
    ],
    "Vol. Surfaces": [
        Path("/data/scene-rep/u/danielxu/triangle/volsurfs/runs/nerf/base/lego/2025-05-11-180029/0199999/renders/test/volumetric/rgb/071.png"),
        Path("/data/scene-rep/u/danielxu/triangle/volsurfs/runs/nerf/base/chair/2025-05-12-001537/0199999/renders/test/volumetric/rgb/041.png"),
        Path("/data/scene-rep/u/danielxu/triangle/volsurfs/runs/nerf/base/ship/2025-05-12-102806/0199999/renders/test/volumetric/rgb/047.png"),
    ],
    "Zip-NeRF": [
        Path("/data/scene-rep/u/danielxu/triangle/zipnerf-results/lego/test_preds/color_071.png"),
        Path("/data/scene-rep/u/danielxu/triangle/zipnerf-results/chair/test_preds/color_041.png"),
        Path("/data/scene-rep/u/danielxu/triangle/zipnerf-results/ship/test_preds/color_047.png"),
    ]
}

# fmt: off
MIP360_RESULTS = {
    "Ground Truth": [
        Path("/data/scene-rep/u/danielxu/datasets/mipnerf360/bicycle/images_4/_DSC8816.JPG"),
        Path("/data/scene-rep/u/danielxu/datasets/mipnerf360/garden/images_4/DSC08028.JPG"),
        Path("/data/scene-rep/u/danielxu/datasets/mipnerf360/bonsai/images_2/DSCF5597.JPG"),
    ],
    "Meshtryoshka (Ours)": [
        Path("/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360/2025_05_22.14_19_49.python3_-m_triangle_splatting__experiment__mip360__dataset.scene_bicycle/workspace/tensorboard/val_render_22386/layered_0.7/_DSC8816.JPG"),
        Path("/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360/2025_05_22.14_20_29.python3_-m_triangle_splatting__experiment__mip360__dataset.scene_garden/workspace/tensorboard/val_render_23985/layered_0.7/DSC08028.JPG"),
        Path("/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360/2025_05_22.14_19_59.python3_-m_triangle_splatting__experiment__mip360__dataset.scene_bonsai/workspace/tensorboard/val_render_23985/layered_0.7/DSCF5597.JPG"),
    ],
    "Zip-NeRF": [
        Path("/data/scene-rep/u/danielxu/triangle/zipnerf-results/bicycle/test_preds/color_017.png"),
        Path("/data/scene-rep/u/danielxu/triangle/zipnerf-results/gardenvase/test_preds/color_009.png"),
        Path("/data/scene-rep/u/danielxu/triangle/zipnerf-results/officebonsai/test_preds/color_004.png"),
    ],
    "3D Gaussian Splatting": [
        Path("/data/scene-rep/u/danielxu/triangle/3dgs-results/bicycle/test/ours_30000/renders/_DSC8816.png"),
        Path("/data/scene-rep/u/danielxu/triangle/3dgs-results/garden/test/ours_30000/renders/DSC08028.png"),
        Path("/data/scene-rep/u/danielxu/triangle/3dgs-results/bonsai/test/ours_30000/renders/DSCF5597.png"),
    ],

}


LIMITATIONS_OURS = [
    Path(
        "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_43_17.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_bicycle/workspace/tensorboard/val_render_22386/surface_1.5/_DSC8679.JPG"
    ),
    Path(
        "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_43_49.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_flowers/workspace/tensorboard/val_render_22386/surface_0.2/_DSC9072.JPG"
    ),
    Path(
        "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_44_20.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_room/workspace/tensorboard/val_render_23985/surface_1.5/DSCF4923.JPG"
    ),
    Path(
        "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_44_41.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_treehill/workspace/tensorboard/val_render_20787/surface_1.5/_DSC8898.JPG"
    ),
    Path(
        "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_44_10.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_kitchen/workspace/tensorboard/val_render_23985/surface_1.5/DSCF0680.JPG"
    ),
]

LIMITATIONS_3DGS = [
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/bicycle/test/ours_30000/renders/_DSC8679.png"
    ),
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/flowers/test/ours_30000/renders/_DSC9072.png"
    ),
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/room/test/ours_30000/renders/DSCF4923.png"
    ),
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/treehill/test/ours_30000/renders/_DSC8898.png"
    ),
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/kitchen/test/ours_30000/renders/DSCF0680.png"
    ),
]

LIMITATIONS_GT = [
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/bicycle/test/ours_30000/gt/_DSC8679.png"
    ),
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/flowers/test/ours_30000/gt/_DSC9072.png"
    ),
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/room/test/ours_30000/gt/DSCF4923.png"
    ),
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/treehill/test/ours_30000/gt/_DSC8898.png"
    ),
    Path(
        "/data/scene-rep/u/danielxu/triangle/3dgs-results/kitchen/test/ours_30000/gt/DSCF0680.png"
    ),
]


# fmt: off
ABLATIONS_PATHS = {
    "Ours": Path(
        "/data/scene-rep/u/danielxu/jobs/triangle_splatting/outputs/5-22/mip360_daniel/2025_05_22.23_43_17.python3_-m_triangle_splatting__experiment__mip360_daniel__dataset.scene_bicycle/workspace/tensorboard/val_render_22386/layered_1.5/_DSC8727.JPG"
    ),
    "3 Shells": Path(
        "/data/scene-rep/u/charatan/sweeps/2025_05_23_ablations/run_3_shells/val_render_22386/layered_1.5/_DSC8727.JPG"
    ),
    "11 Shells": Path(
        "/data/scene-rep/u/charatan/sweeps/2025_05_23_ablations/run_11_shells/val_render_20787/layered_1.5/_DSC8727.JPG"
    ),
    "No Exponential": Path(
        "/data/scene-rep/u/charatan/sweeps/2025_05_23_ablations/run_no_exponential/val_render_22386/layered_1.5/_DSC8727.JPG"
    ),
    "No Regularizers": Path(
        "/data/scene-rep/u/charatan/sweeps/2025_05_23_ablations/run_no_regularizers/val_render_11193/layered_1.5/_DSC8727.JPG"
    ),
    "No Sparsity": Path(
        "/data/scene-rep/u/charatan/sweeps/2025_05_23_ablations/run_no_sparsity/val_render_7995/layered_1.5/_DSC8727.JPG"
    ),
    "No SH Coeffs.": Path(
        "/data/scene-rep/u/charatan/sweeps/2025_05_23_ablations/run_no_spherical_harmonics/val_render_12792/layered_1.5/_DSC8727.JPG"
    ),
    "No Frustums": Path(
        "/data/scene-rep/u/charatan/sweeps/2025_05_23_ablations/run_no_tesseract/val_render_22386/layered_1.5/_DSC8727.JPG"
    ),
}
# fmt: on


def get_metrics_from_image_path(image_path: Path):
    """
    Navigate to directory of image_path, go up one level, and
    load json "surface_1.5.json" as dict.
    """
    # Resolve the parent directory of the image_path
    parent_dir = image_path.parent.parent

    # Construct the path to 'surface_1.5.json' in the parent directory
    json_path = parent_dir / "surface_1.5.json"

    # Read and parse the JSON file
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data
