import logging
import os
import platform
import shutil
from contextlib import nullcontext
from pathlib import Path

from jaxtyping import install_import_hook

# Set up logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info(f"Host: {platform.node()}")

import_context = (
    install_import_hook(("meshtryoshka", "triangle_rasterization"), "beartype.beartype")
    if os.environ.get("USE_JAXTYPING", None)
    else nullcontext()
)
with import_context:
    from meshtryoshka.config import get_typed_config
    from meshtryoshka.trainer import Trainer

# Read the configuration.
cfg = get_typed_config()

# Read the workspace directory.
if os.environ.get("WORKSPACE", None) is None:
    raise ValueError("You must specify the WORKSPACE environment variable.")
workspace = Path(os.environ["WORKSPACE"])
if cfg.overwrite:
    shutil.rmtree(workspace, ignore_errors=True)

# Start the training loop.
trainer = Trainer(workspace, cfg)
if cfg.benchmark_rendering_speed:
    trainer.benchmark_rendering_speed()
else:
    trainer.train()
