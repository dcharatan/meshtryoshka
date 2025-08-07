from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper",),
    ("beartype", "beartype"),
):
    from ..figures.figure_ablations import ABLATIONS
    from ..figures.results_paths import ABLATIONS_PATHS, get_metrics_from_image_path
    from .table import make_latex_table

if __name__ == "__main__":
    results = {}
    for k in ABLATIONS:
        results_json = get_metrics_from_image_path(ABLATIONS_PATHS[k])
        results[k] = [
            results_json["psnr"],
            results_json["ssim"],
            results_json["lpips"],
        ]

    table = make_latex_table(
        results,
        ["PSNR", "SSIM", "LPIPS"],
        [2, 3, 3],
        [1, 1, -1],
    )
    print(table)
