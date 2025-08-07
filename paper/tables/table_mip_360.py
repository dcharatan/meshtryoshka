from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("paper",),
    ("beartype", "beartype"),
):
    from .table import make_latex_table

if __name__ == "__main__":
    results = {
        "Ours (All Layers)": [
            24.26892472105705,
            0.6697918708135583,
            0.3681565161997995,
        ],
        "Ours (Zero Level Set)": [
            24.234259660331034,
            0.6686156314373483,
            0.36896022751681007,
        ],
        "Zip-NeRF": [
            28.54,
            0.828,
            0.189,
        ],
        "3D Gaussian Splatting": [27.21, 0.815, 0.214],
    }
    table = make_latex_table(
        results,
        ["PSNR", "SSIM", "LPIPS"],
        [2, 3, 3],
        [1, 1, -1],
    )
    print(table)
