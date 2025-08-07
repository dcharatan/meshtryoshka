import os
import shutil
import sys
from pathlib import Path

if __name__ == "__main__":
    executable_path = Path(sys.executable)
    library_path = executable_path.parents[1] / "lib"

    # Find the Python version.
    library_python_path = next(iter(library_path.iterdir()))

    # Specify the Slang kernels' location.
    slang_location = library_python_path / "site-packages/triangle_rasterization/slang"

    # Delete the Slang cache and lock files.
    shutil.rmtree(slang_location / ".slangtorch_cache", True)
    os.system(f"rm -r {slang_location}/*.slang*.lock")

    # Clear the PyTorch extension cache.
    shutil.rmtree(Path.home() / ".cache/torch_extensions", True)
