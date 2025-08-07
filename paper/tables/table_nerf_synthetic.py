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
            29.36418696761131,
            0.9386499463394284,
            0.07683064026408828,
        ],
        "Ours (Zero Level Set)": [
            29.37223597764969,
            0.9386499463394284,
            0.07683061218471265,
        ],
        "NVDiffrec (DMTet)": [
            28.804346112012862,
            0.9376245177537204,
            0.07844879605807364,
        ],
        "NVDiffrec (Flexicubes)": [
            29.218404756784434,
            0.9404703509435057,
            0.07605997295817361,
        ],
        "Neus2": [29.716569925546647, 0.9433680655807256, 0.06838822739489842],
        "Zip-NeRF": [33.10, 0.971, 0.031],
        "Volumetric Surfaces": [
            27.326965482234957,
            0.9194892188161611,
            0.11024817766039632,
        ],
    }
    table = make_latex_table(
        results,
        ["PSNR", "SSIM", "LPIPS"],
        [2, 3, 3],
        [1, 1, -1],
    )
    print(table)
