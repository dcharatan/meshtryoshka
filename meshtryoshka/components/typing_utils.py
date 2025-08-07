from dataclasses import fields, is_dataclass
from typing import Any

import torch


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move all torch.Tensor objects contained in obj to the given device.
    If obj is a dataclass, list, dict, or tuple containing tensors (or nested such
    objects), this will return a new object with all tensors moved to the target device.
    """
    # If it's a tensor, move it.
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    # If it's a dataclass, recursively move each field.
    if is_dataclass(obj):
        # Build a dict of field values after moving them to the device.
        field_values = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            field_values[field.name] = to_device(value, device)
        return type(obj)(**field_values)

    # If it's a dictionary, apply to_device on both keys and values.
    if isinstance(obj, dict):
        return {to_device(k, device): to_device(v, device) for k, v in obj.items()}

    # If it's a list, process each element.
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]

    # If it's a tuple, process each element and return a tuple.
    if isinstance(obj, tuple):
        return tuple(to_device(x, device) for x in obj)

    # Otherwise, return the object as is.
    return obj
