import os
import shutil
from pathlib import Path

if __name__ == "__main__":
    # Specify the Slang kernels' location.
    slang_location = Path("triangle_rasterization")

    # Delete the Slang cache and lock files.
    shutil.rmtree(slang_location / ".slangtorch_cache", True)
    os.system(f"rm -r {slang_location}/*.slang*.lock")

    # Clear the PyTorch extension cache.
    shutil.rmtree(Path.home() / ".cache/torch_extensions", True)
