import torch
from jaxtyping import Float, Int, Int64
from torch import Tensor


def index_vertices(
    points: Float[Tensor, "vertex 3"],
    tol: float = 1e-5,
) -> Int[Tensor, " vertex"]:
    """
    Returns a 0-based LongTensor `inv` of shape (V,), where:
      - inv[i] == inv[j] if points[i] and points[j] are within `tol`.
      - New IDs follow *first-seen* order in the original tensor.
    """

    # 1) Quantize to integer grid & pack into 1D keys
    q = torch.round(points / tol).to(torch.int64)  # (V,3)
    max_abs = int(q.abs().max().item())
    bits = max_abs.bit_length() + 1
    if bits * 3 > 63:
        raise RuntimeError(f"need {bits*3} bits, exceed 63 bits total")

    shift = bits
    x, y, z = q.unbind(dim=1)
    keys = (x << (2 * shift)) | (y << shift) | z  # (V,)

    return index_keys(keys)


def index_keys(
    keys: Int64[Tensor, " vertex"],
) -> Int[Tensor, " vertex"]:
    device = keys.device
    (num_keys,) = keys.shape

    # Create permutations for sorting and un-sorting the keys.
    sorted_keys, sorted_to_arange = keys.sort()
    arange_to_sorted = torch.empty_like(sorted_to_arange)
    arange_to_sorted[sorted_to_arange] = torch.arange(num_keys, device=device)

    # Use the following mappings:
    # - original index -> sorted index
    # - sorted index -> "collapsed" sorted index (index of first same key)
    # - "collapsed" sorted index -> original index
    _, unique, counts = torch.unique_consecutive(
        sorted_keys,
        return_inverse=True,
        return_counts=True,
    )
    collapse_sorted = (counts.cumsum(dim=0) - counts)[unique]
    return sorted_to_arange[collapse_sorted[arange_to_sorted]].type(torch.int32)
