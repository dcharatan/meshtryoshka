from pathlib import Path

from jaxtyping import Int32
from torch import Tensor
from torch.utils import cpp_extension

from .common import record_function
from .compilation import wrap_compilation

cuda = wrap_compilation(
    lambda: cpp_extension.load(
        name="compute_exclusive_cumsum",
        sources=[Path(__file__).parent / "compute_exclusive_cumsum.cu"],
        verbose=True,
    )
)


@record_function("compute_exclusive_cumsum")
def compute_exclusive_cumsum(x: Int32[Tensor, " entry"]) -> None:
    cuda().compute_exclusive_cumsum(x)
