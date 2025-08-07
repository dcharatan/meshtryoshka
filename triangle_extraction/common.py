from functools import wraps
from typing import TypeVar

import torch
from einops import rearrange
from jaxtyping import Bool, Int32
from torch import Tensor
from torch.profiler import record_function as record_function_torch

T = TypeVar("T")


def ceildiv(numerator: T, denominator: T) -> T:
    return (numerator + denominator - 1) // denominator


def pack_occupancy(
    occupancy: Bool[Tensor, "i j k"],
) -> Int32[Tensor, "i j k//32"]:
    # The last side length must be divisible by 32.
    assert occupancy.shape[-1] % 32 == 0

    # Match each boolean value with its corresponding power of two.
    occupancy = rearrange(occupancy, "i j (k p) -> i j k p", p=32)
    powers = 1 << torch.arange(32, device=occupancy.device, dtype=torch.int64)
    return (occupancy * powers).sum(dim=-1).type(torch.uint32).view(torch.int32)


def unpack_occupancy(
    packed_occupancy: Int32[Tensor, "i j k_packed"],
) -> Bool[Tensor, "i j k_packed*32"]:
    occupancy = packed_occupancy.view(torch.int32)
    powers = 1 << torch.arange(32, device=occupancy.device, dtype=torch.int32)
    occupancy = occupancy[..., None] & powers
    return rearrange(occupancy, "i j k p -> i j (k p)").type(torch.bool)


def record_function(tag: str):
    """This functions the same way as torch.profiler.record_function, except that it
    properly maintains the function signature for VS Code autocomplete.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with record_function_torch(tag):
                return f(*args, **kwargs)

        return wrapper

    return decorator
